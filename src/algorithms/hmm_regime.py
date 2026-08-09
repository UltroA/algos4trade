"""
HMM (скрытая марковская модель) - определение режима рынка.

гауссова HMM с 3 скрытыми состояниями пытается разделить
рынок на устойчивые режимы (тренд-вверх / флэт / стресс) по доходности и
волатильности. Реалистичный уровень - 2-4 устойчивых режима удаётся выделить
в среднем; главная слабость (см. таблицу) - режимы переопределяются задним
числом: Viterbi-декодирование, применённое к уже свершившемуся отрезку, не
эквивалентно онлайн-детекции в реальном времени. Здесь для простоты бэктеста
`model.predict()` вызывается сразу на всём test-окне (см. generate_signals) -
это оптимистичное допущение, в проде уместнее скользящее online-декодирование.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split


class HMMRegimeDetector(SingleAssetAlgorithm):
    name = "HMM Regime Detector"
    category = AlgorithmCategory.REGIME_DETECTION
    description = (
        "Гауссова скрытая марковская модель с 3 состояниями классифицирует рыночный режим "
        "(бычий тренд / флэт / стрессовая просадка) по доходности и волатильности и торгует по нему."
    )

    def __init__(self, n_states: int = 3, n_iter: int = 100, vol_window: int = 10, seed: int = 0, **kwargs):
        super().__init__(n_states=n_states, n_iter=n_iter, vol_window=vol_window, seed=seed, **kwargs)
        self.n_states = n_states
        self.vol_window = vol_window
        self.model = GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=n_iter, random_state=seed
        )
        self._bull_state: int | None = None
        self._bear_state: int | None = None

    def _state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df["close"].pct_change()
        vol = ret.rolling(self.vol_window).std()
        feat = pd.DataFrame({"return": ret, "volatility": vol}, index=df.index)
        return feat

    def fit(self, train_data: pd.DataFrame) -> "HMMRegimeDetector":
        feat = self._state_features(train_data).dropna()
        if len(feat) < self.n_states * 10:
            self.is_fitted = False
            return self

        X = feat.to_numpy()
        self.model.fit(X)

        # Определяем "бычий" (наибольшая средняя доходность) и "медвежий/стрессовый"
        # (наименьшая средняя доходность) режимы по means_ обученной модели.
        mean_returns = self.model.means_[:, 0]
        self._bull_state = int(np.argmax(mean_returns))
        self._bear_state = int(np.argmin(mean_returns))
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        feat = self._state_features(data).dropna()
        if feat.empty:
            return pd.Series(0.0, index=data.index)

        states = self.model.predict(feat.to_numpy())
        signal = pd.Series(0.0, index=feat.index)
        signal[states == self._bull_state] = 1.0
        signal[states == self._bear_state] = -1.0

        return signal.reindex(data.index).fillna(0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    # Синтетика с переключением режимов: тренд-вверх, флэт, стрессовая просадка.
    regimes = np.random.default_rng(1).choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    drift_by_regime = {0: 0.002, 1: 0.0, 2: -0.004}
    vol_by_regime = {0: 0.008, 1: 0.006, 2: 0.02}
    returns = np.array([rng.normal(drift_by_regime[r], vol_by_regime[r]) for r in regimes])
    price = 100 * np.exp(np.cumsum(returns))
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
    algo = HMMRegimeDetector()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
    print(signals.value_counts())
