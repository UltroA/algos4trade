"""
Logistic regression - breakout/pullback probability forecast, calibration.

A linear classifier of price direction (fwd_direction)
on top of the same engineered features as the project's other supervised
models. Unlike trees/boosting, it gives a well-calibrated class probability
(predict_proba closer to the true event frequency), which is convenient for
a threshold decision like "only trade when confidence is high".

Realistic level (see docs/Алгоритмы.md): direction accuracy ~51-53%.
Requires manual feature engineering - the model does not build nonlinear
feature interactions on its own.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target


class LogisticRegressionDirection(SingleAssetAlgorithm):
    name = "Logistic Regression Direction"
    category = AlgorithmCategory.SUPERVISED_RANKING
    description = (
        "Logistic regression predicts a calibrated probability of price increase from technical "
        "features - a simple linear baseline for breakout/pullback signals."
    )

    def __init__(
        self,
        horizon: int = 1,
        C: float = 1.0,
        signal_strength: float = 4.0,
        **kwargs,
    ):
        super().__init__(horizon=horizon, C=C, signal_strength=signal_strength, **kwargs)
        self.horizon = horizon
        self.signal_strength = signal_strength
        self.scaler = StandardScaler()
        self.model = SkLogisticRegression(
            C=C,
            max_iter=1000,
            **{k: v for k, v in kwargs.items() if k not in {"horizon", "C", "signal_strength"}},
        )

    def fit(self, train_data: pd.DataFrame) -> "LogisticRegressionDirection":
        feat_df = make_features(train_data, horizon=self.horizon)
        X, y = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty or y.nunique() < 2:
            self.is_fitted = False
            return self
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data, horizon=self.horizon)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty:
            return pd.Series(0.0, index=data.index)

        X_scaled = self.scaler.transform(X)
        proba_up = self.model.predict_proba(X_scaled)[:, 1]
        raw_signal = (proba_up - 0.5) * self.signal_strength
        signal = pd.Series(raw_signal, index=X.index).clip(-1.0, 1.0)
        return signal.reindex(data.index).fillna(0.0)


if __name__ == "__main__":
    # Quick self-check on synthetic data (no network).
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
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
    algo = LogisticRegressionDirection()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
