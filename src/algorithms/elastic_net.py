"""
Elastic Net - factor model, feature selection from a set of alphas.

Table row #3 (Elastic Net / Lasso): linear regression with a combination
of L1/L2 regularization predicts future return (fwd_return, regression,
not direction classification) as a linear combination of engineered features.
The L1 part of the regularization zeroes out weights of irrelevant features (factor selection),
the L2 part stabilizes the solution under correlated features.

Realistic level (see docs/Алгоритмы.md): low accuracy but high
coefficient stability. Captures only linear dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, select_features, split_features_target


class ElasticNetFactorModel(SingleAssetAlgorithm):
    name = "Elastic Net Factor Model"
    category = AlgorithmCategory.LINEAR_FACTOR
    description = (
        "Linear factor model (Elastic Net) predicts future return as a combination of "
        "technical features, with L1 selection of significant factors and L2 stabilization of coefficients."
    )

    def __init__(
        self,
        horizon: int = 1,
        alpha: float = 0.001,
        l1_ratio: float = 0.5,
        signal_strength: float = 20.0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon, alpha=alpha, l1_ratio=l1_ratio, signal_strength=signal_strength, **kwargs
        )
        self.horizon = horizon
        self.signal_strength = signal_strength
        self.scaler = StandardScaler()
        # fwd_return's natural scale shrinks with bar length (~1e-4 for 5min
        # bars vs ~1e-2 for daily bars); alpha is an absolute penalty on
        # y_scaled's coefficients (which are O(1) regardless of bar length -
        # X is already standardized), so without also scaling y, the same
        # alpha that's reasonable for daily bars swamps every coefficient to
        # exactly 0 on intraday bars (verified: 0/19 nonzero on live 5min
        # data), collapsing the model to a constant (its intercept) that
        # never meaningfully changes the position.
        self.y_scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10_000,
            **{k: v for k, v in kwargs.items() if k not in {"horizon", "alpha", "l1_ratio", "signal_strength"}},
        )

    def fit(self, train_data: pd.DataFrame) -> "ElasticNetFactorModel":
        feat_df = make_features(train_data, horizon=self.horizon)
        X, y = split_features_target(feat_df, target_col="fwd_return")
        if X.empty or y.empty:
            self.is_fitted = False
            return self
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data, horizon=self.horizon)
        X = select_features(feat_df)
        if X.empty:
            return pd.Series(0.0, index=data.index)

        X_scaled = self.scaler.transform(X)
        predicted_return_scaled = self.model.predict(X_scaled)
        predicted_return = self.y_scaler.inverse_transform(predicted_return_scaled.reshape(-1, 1)).ravel()
        raw_signal = np.tanh(predicted_return * self.signal_strength)
        signal = pd.Series(raw_signal, index=X.index).clip(-1.0, 1.0)
        return signal.reindex(data.index).fillna(0.0)


if __name__ == "__main__":
    # Quick self-check on synthetic data (no network).
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
    algo = ElasticNetFactorModel()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
