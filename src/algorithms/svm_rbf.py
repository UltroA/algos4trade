"""
SVM with an RBF kernel - direction classification on small samples.

Support vectors with an RBF kernel predict the probability
of the price rising over the next horizon using the same technical features as
the project's other supervised models (core.features.make_features).

Realistic level: 51-53% directional accuracy. The main weakness is
poor scalability: SVM training scales as O(n^2..n^3) in the number of
samples, so this model is suitable primarily for small/medium
train samples (hundreds to a few thousand rows), typical of a single ticker.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target


class SVMDirectionClassifier(SingleAssetAlgorithm):
    name = "SVM RBF Direction Classifier"
    category = AlgorithmCategory.SUPERVISED_RANKING
    description = (
        "An SVM with an RBF kernel classifies the direction of price movement over the next horizon "
        "using technical features. Works well on small samples, scales poorly on large ones."
    )

    def __init__(
        self,
        horizon: int = 1,
        C: float = 1.0,
        gamma: str | float = "scale",
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon,
            C=C,
            gamma=gamma,
            signal_strength=signal_strength,
            seed=seed,
            **kwargs,
        )
        self.horizon = horizon
        self.signal_strength = signal_strength
        self.scaler = StandardScaler()
        self.model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed)

    def fit(self, train_data: pd.DataFrame) -> "SVMDirectionClassifier":
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
    algo = SVMDirectionClassifier()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
