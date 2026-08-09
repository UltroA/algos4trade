"""
Isolation Forest - anomaly detection, a risk switch ahead of a crash.

Isolation Forest (One-Class SVM): the model does not generate
a trading signal on its own, but detects anomalous market regimes
(sharp spikes in volatility/volume/returns) and uses them to modulate a
simple base trend signal (SMA(10) vs SMA(50)) - on anomalous days the
position is forcibly zeroed out (a "risk switch").

Realistic level: catches 60-80% of stress days. Main weakness -
many false positives (the model detects statistical outliers, not
specifically days that are "bad for the strategy", so some signal
zero-outs happen for no reason, reducing returns during calm but
atypical periods).
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import FEATURE_COLUMNS, chronological_split, make_features, split_features_target


class IsolationForestRiskSwitch(SingleAssetAlgorithm):
    name = "Isolation Forest Risk Switch"
    category = AlgorithmCategory.ANOMALY_DETECTION
    description = (
        "Isolation Forest detects anomalous market regimes and zeroes out the position "
        "of a base trend strategy (SMA crossover) on such days - a risk switch ahead of a crash."
    )

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 50,
        contamination: float = 0.05,
        n_estimators: int = 200,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            fast_window=fast_window,
            slow_window=slow_window,
            contamination=contamination,
            n_estimators=n_estimators,
            seed=seed,
            **kwargs,
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
        )

    @staticmethod
    def _base_trend_signal(data: pd.DataFrame, fast: int, slow: int) -> pd.Series:
        sma_fast = data["close"].rolling(fast).mean()
        sma_slow = data["close"].rolling(slow).mean()
        return (sma_fast > sma_slow).astype(float) * 2 - 1

    def fit(self, train_data: pd.DataFrame) -> "IsolationForestRiskSwitch":
        feat_df = make_features(train_data, horizon=1)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty:
            self.is_fitted = False
            return self
        self.model.fit(X)
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

        anomaly_flag = self.model.predict(X)  # -1 = anomaly, 1 = normal
        is_normal = pd.Series(anomaly_flag == 1, index=X.index)

        risk_switch = is_normal.reindex(data.index).fillna(False).astype(float)
        signal = base_signal.fillna(0.0) * risk_switch
        return signal.clip(-1.0, 1.0)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # add a few anomalous days (sharp jumps) to test the risk switch
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
    algo = IsolationForestRiskSwitch()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
