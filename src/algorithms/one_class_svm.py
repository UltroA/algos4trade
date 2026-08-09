"""
One-Class SVM - детекция аномалий, риск-выключатель перед крахом.

как и isolation_forest.py, модель не формирует сигнал сама, а обнаруживает аномальный рыночный режим и
модулирует им базовый трендовый сигнал (SMA(10) vs SMA(50)), обнуляя позицию
в аномальные дни ("риск-выключатель"). В отличие от Isolation Forest, One-Class
SVM строит гиперповерхность вокруг "нормальных" точек в признаковом пространстве
через RBF-ядро, поэтому чувствителен к масштабу признаков - обязательна
стандартизация (StandardScaler, обученный на train).

Реалистичный уровень: ловит 60-80% стрессовых дней. Главная слабость -
много ложных срабатываний (модель реагирует на любые статистические выбросы
в признаках, а не только на действительно неблагоприятные для стратегии дни).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target


class OneClassSVMRiskSwitch(SingleAssetAlgorithm):
    name = "One-Class SVM Risk Switch"
    category = AlgorithmCategory.ANOMALY_DETECTION
    description = (
        "One-Class SVM с RBF-ядром детектирует аномальные рыночные режимы и обнуляет позицию "
        "базовой трендовой стратегии (SMA crossover) в такие дни - риск-выключатель перед крахом."
    )

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 50,
        nu: float = 0.05,
        gamma: str | float = "scale",
        **kwargs,
    ):
        super().__init__(
            fast_window=fast_window,
            slow_window=slow_window,
            nu=nu,
            gamma=gamma,
            **kwargs,
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

    @staticmethod
    def _base_trend_signal(data: pd.DataFrame, fast: int, slow: int) -> pd.Series:
        sma_fast = data["close"].rolling(fast).mean()
        sma_slow = data["close"].rolling(slow).mean()
        return (sma_fast > sma_slow).astype(float) * 2 - 1

    def fit(self, train_data: pd.DataFrame) -> "OneClassSVMRiskSwitch":
        feat_df = make_features(train_data, horizon=1)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty:
            self.is_fitted = False
            return self
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        base_signal = self._base_trend_signal(data, self.fast_window, self.slow_window)

        feat_df = make_features(data, horizon=1)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty:
            return pd.Series(0.0, index=data.index)

        X_scaled = self.scaler.transform(X)
        anomaly_flag = self.model.predict(X_scaled)  # -1 = аномалия, 1 = норма
        is_normal = pd.Series(anomaly_flag == 1, index=X.index)

        risk_switch = is_normal.reindex(data.index).fillna(False).astype(float)
        signal = base_signal.fillna(0.0) * risk_switch
        return signal.clip(-1.0, 1.0)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    price[200] *= 1.15
    price[250] *= 0.85
    dummy = pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    train_df, test_df = chronological_split(dummy, 0.7)
    algo = OneClassSVMRiskSwitch()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
