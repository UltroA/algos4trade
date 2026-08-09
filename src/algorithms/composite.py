"""
Композитные алгоритмы: собирают уже реализованные "строительные блоки"
бенчмарка в единый пайплайн генерация сигнала -> риск-оверлей -> сайзинг
позиции / аллокация капитала, чтобы задокументированный недостаток одного
компонента был закрыт задокументированной сильной стороной другого:

  - Isolation Forest / One-Class SVM сами по себе не генерируют альфу
    (торгуют примитивным SMA(10)/SMA(50)-кроссовером), но по итогам
    кросс-секционной валидации на 97 тикерах оказались лучшим риск-оверлеем
    (снижают частоту сделок в нестабильные периоды, mean SR 0.49/0.39,
    доля тикеров с положительным SR 78%/66%). `AnomalyRiskOverlay` берёт их
    детектор аномалий, но применяет его к сигналу ЛЮБОГО другого
    single-asset алгоритма проекта вместо жёстко зашитого SMA-кроссовера.
  - VAE и N-BEATS дают содержательный, но зашумлённый сигнал без какого-либо
    риск-контроля (VAE: mean SR 0.12, mean MDD -10.5%; N-BEATS: mean SR
    0.19, mean MDD -2.8%) - их недостаток (нет защиты от нестабильных
    режимов) закрывается риск-оверлеем выше.
  - PPO - лучший протестированный сайзер позиции, но переобучение RL с нуля
    на каждом тикере/композите вычислительно дорого; `VolTargetSizer`
    реализует ту же идею (не давать резких по размеру ставок) дешёвым
    таргетированием волатильности вместо RL.
  - Thompson Sampling Allocator сам по себе не создаёт альфу и, по выводам
    расширенной валидации, "нуждается в более сильных базовых стратегиях,
    чем сырой momentum" - `ThompsonWithStrongArms` заменяет momentum-"руки"
    на произвольный композитный пайплайн (например, Elastic Net + risk
    overlay), это прямая проверка того вывода.

Все классы реализуют стандартный интерфейс BaseTradingAlgorithm, поэтому
встраиваются в BenchmarkRunner/Backtester наравне с обычными алгоритмами.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from core.base import AlgorithmCategory, MultiAssetAlgorithm, SingleAssetAlgorithm
from core.features import make_features, split_features_target

_DETECTOR_LABELS = {"isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM"}


class AnomalyRiskOverlay(SingleAssetAlgorithm):
    """Базовый сигнал произвольного алгоритма, обнуляемый риск-оверлеем на аномальных днях."""

    category = AlgorithmCategory.COMPOSITE

    def __init__(
        self,
        base_algo_factory: Callable[[], SingleAssetAlgorithm],
        detector: str = "isolation_forest",
        contamination: float = 0.05,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(detector=detector, contamination=contamination, seed=seed, **kwargs)
        if detector not in _DETECTOR_LABELS:
            raise ValueError(f"unknown detector {detector!r}, expected one of {list(_DETECTOR_LABELS)}")
        self.base_algo = base_algo_factory()
        self.detector_type = detector
        self.contamination = contamination
        self.seed = seed
        self.scaler = StandardScaler() if detector == "one_class_svm" else None
        self.detector_model = (
            IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
            if detector == "isolation_forest"
            else OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
        )
        self.name = f"{self.base_algo.name} + {_DETECTOR_LABELS[detector]} overlay"
        self.description = (
            f"Композит: базовый сигнал '{self.base_algo.name}', обнуляемый риск-оверлеем "
            f"({_DETECTOR_LABELS[detector]}) на днях, помеченных как аномальные."
        )

    def fit(self, train_data: pd.DataFrame) -> "AnomalyRiskOverlay":
        self.base_algo.fit(train_data)
        feat_df = make_features(train_data, horizon=1)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty or not self.base_algo.is_fitted:
            self.is_fitted = False
            return self
        Xt = self.scaler.fit_transform(X) if self.scaler is not None else X
        self.detector_model.fit(Xt)
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        base_signal = self.base_algo.generate_signals(data).reindex(data.index).fillna(0.0)

        feat_df = make_features(data, horizon=1)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty:
            return pd.Series(0.0, index=data.index)

        Xt = self.scaler.transform(X) if self.scaler is not None else X
        anomaly_flag = self.detector_model.predict(Xt)  # -1 = аномалия, 1 = норма
        is_normal = pd.Series(anomaly_flag == 1, index=X.index)
        risk_switch = is_normal.reindex(data.index).fillna(False).astype(float)

        return (base_signal * risk_switch).clip(-1.0, 1.0)


class VolTargetSizer(SingleAssetAlgorithm):
    """Масштабирует размер позиции базового алгоритма обратно пропорционально реализованной волатильности."""

    category = AlgorithmCategory.COMPOSITE

    def __init__(
        self,
        base_algo_factory: Callable[[], SingleAssetAlgorithm],
        vol_window: int = 20,
        target_vol: float = 0.02,
        max_scale: float = 1.5,
        **kwargs,
    ):
        super().__init__(vol_window=vol_window, target_vol=target_vol, max_scale=max_scale, **kwargs)
        self.base_algo = base_algo_factory()
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_scale = max_scale
        self.name = f"{self.base_algo.name} (vol-targeted sizing)"
        self.description = (
            f"Композит: сигнал '{self.base_algo.name}', размер позиции масштабирован обратно "
            f"пропорционально реализованной волатильности ({vol_window}-дневное окно) вместо "
            f"фиксированного +-1 - дешёвая замена RL-сайзеру (PPO)."
        )

    def fit(self, train_data: pd.DataFrame) -> "VolTargetSizer":
        self.base_algo.fit(train_data)
        self.is_fitted = self.base_algo.is_fitted
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)
        base_signal = self.base_algo.generate_signals(data).reindex(data.index).fillna(0.0)
        realized_vol = data["close"].pct_change().rolling(self.vol_window).std()
        scale = (self.target_vol / realized_vol.replace(0, np.nan)).clip(upper=self.max_scale).fillna(1.0)
        return (base_signal * scale).clip(-1.0, 1.0)


class ThompsonWithStrongArms(MultiAssetAlgorithm):
    """Thompson sampling аллокатор капитала (как в thompson_bandits.py), но "рукой" на каждом
    тикере служит произвольный композитный пайплайн вместо сырого momentum-сигнала."""

    category = AlgorithmCategory.CAPITAL_ALLOCATION

    def __init__(
        self,
        arm_algo_factory: Callable[[], SingleAssetAlgorithm],
        seed: int = 0,
        max_weight: float = 1.0,
        arm_label: str = "custom arm",
        **kwargs,
    ):
        super().__init__(seed=seed, max_weight=max_weight, **kwargs)
        self.arm_algo_factory = arm_algo_factory
        self.max_weight = max_weight
        self._rng = np.random.default_rng(seed)
        self._arms: dict[str, SingleAssetAlgorithm] = {}
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}
        self.name = f"Thompson Sampling ({arm_label})"
        self.description = (
            f"Байесовский многорукий бандит (Thompson sampling) распределяет капитал между "
            f"тикерами, где каждая 'рука' - не сырой momentum, а полноценный пайплайн ({arm_label})."
        )

    def fit(self, train_data: dict[str, pd.DataFrame]) -> "ThompsonWithStrongArms":
        for ticker, df in train_data.items():
            arm = self.arm_algo_factory()
            arm.fit(df)
            self._arms[ticker] = arm
            if not arm.is_fitted:
                self._alpha[ticker] = 1.0
                self._beta[ticker] = 1.0
                continue
            arm_signal = arm.generate_signals(df).reindex(df.index).fillna(0.0)
            arm_return = arm_signal.shift(1) * df["close"].pct_change()
            wins = int((arm_return > 0).sum())
            losses = int((arm_return < 0).sum())
            self._alpha[ticker] = wins + 1.0
            self._beta[ticker] = losses + 1.0
        self.is_fitted = True
        return self

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        if not self.is_fitted:
            return {t: pd.Series(0.0, index=df.index) for t, df in data.items()}

        tickers = list(data.keys())
        base_signals = {}
        for t in tickers:
            arm = self._arms.get(t)
            if arm is not None and arm.is_fitted:
                base_signals[t] = arm.generate_signals(data[t]).reindex(data[t].index).fillna(0.0)
            else:
                base_signals[t] = pd.Series(0.0, index=data[t].index)

        common_index = base_signals[tickers[0]].index
        for s in base_signals.values():
            common_index = common_index.union(s.index)
        common_index = common_index.sort_values()

        alpha = {t: self._alpha.get(t, 1.0) for t in tickers}
        beta_param = {t: self._beta.get(t, 1.0) for t in tickers}

        weights_over_time: dict[str, list[float]] = {t: [] for t in tickers}
        for _ in common_index:
            samples = np.array([self._rng.beta(alpha[t], beta_param[t]) for t in tickers])
            exp_samples = np.exp((samples - samples.max()) * 5.0)
            weights = exp_samples / exp_samples.sum()
            for t, w in zip(tickers, weights):
                weights_over_time[t].append(min(w, self.max_weight))

        signals: dict[str, pd.Series] = {}
        for t in tickers:
            direction = base_signals[t].reindex(common_index).fillna(0.0)
            weight = pd.Series(weights_over_time[t], index=common_index)
            signals[t] = (direction * weight).reindex(data[t].index).fillna(0.0).clip(-1.0, 1.0)
        return signals


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from algorithms.elastic_net import ElasticNetFactorModel
    from algorithms.vae import VAEFactorModel
    from core.features import chronological_split

    rng = np.random.default_rng(0)
    n = 400
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    price[200] *= 1.15
    dummy = pd.DataFrame(
        {"open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
         "volume": rng.integers(1_000_000, 5_000_000, n)},
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    train_df, test_df = chronological_split(dummy, 0.7)

    combo = AnomalyRiskOverlay(lambda: VAEFactorModel(), detector="isolation_forest")
    combo.fit(train_df)
    print(combo.name, "->", combo.generate_signals(test_df).describe())

    sized = VolTargetSizer(lambda: ElasticNetFactorModel())
    sized.fit(train_df)
    print(sized.name, "->", sized.generate_signals(test_df).describe())

    data = {"A": dummy, "B": dummy * 1.01}
    train_data = {k: v.iloc[:280] for k, v in data.items()}
    test_data = {k: v.iloc[280:] for k, v in data.items()}
    allocator = ThompsonWithStrongArms(
        arm_algo_factory=lambda: AnomalyRiskOverlay(lambda: ElasticNetFactorModel(), detector="isolation_forest"),
        arm_label="Elastic Net + Isolation Forest",
    )
    allocator.fit(train_data)
    signals = allocator.generate_signals(test_data)
    for ticker, s in signals.items():
        print(ticker, s.describe())
