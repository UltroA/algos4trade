"""
Random Forest - a base baseline and feature importance estimate.

An ensemble of independent decision trees (bagging) over the same
engineered features as the gradient boosting in lightgbm_ranker.py.
Serves as a baseline for comparison with boosting - usually weaker (smooths
out rare signals due to averaging across trees), but more robust to
overfitting and gives an easy-to-interpret feature importance estimate
(impurity-based).

Realistic level (see docs/Алгоритмы.md): direction accuracy ~51-54%.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import FEATURE_COLUMNS, chronological_split, make_features, split_features_target


class RandomForestBaseline(SingleAssetAlgorithm):
    name = "Random Forest Baseline"
    category = AlgorithmCategory.SUPERVISED_RANKING
    description = (
        "A base baseline using bagged decision trees for price direction forecasting and estimating "
        "the importance of technical features."
    )

    def __init__(
        self,
        horizon: int = 1,
        n_estimators: int = 300,
        max_depth: int = 6,
        min_samples_leaf: int = 10,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            signal_strength=signal_strength,
            seed=seed,
            **kwargs,
        )
        self.horizon = horizon
        self.signal_strength = signal_strength
        self._trained_columns: list[str] = []
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=seed,
            **{k: v for k, v in kwargs.items() if k not in {"horizon", "n_estimators", "max_depth", "min_samples_leaf", "signal_strength", "seed"}},
        )

    def fit(self, train_data: pd.DataFrame) -> "RandomForestBaseline":
        feat_df = make_features(train_data, horizon=self.horizon)
        X, y = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty or y.nunique() < 2:
            self.is_fitted = False
            return self
        self.model.fit(X, y)
        self._trained_columns = list(X.columns)
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(0.0, index=data.index)

        feat_df = make_features(data, horizon=self.horizon)
        X, _ = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty:
            return pd.Series(0.0, index=data.index)

        proba_up = self.model.predict_proba(X)[:, 1]
        raw_signal = (proba_up - 0.5) * self.signal_strength
        signal = pd.Series(raw_signal, index=X.index).clip(-1.0, 1.0)
        return signal.reindex(data.index).fillna(0.0)

    def feature_importances(self) -> pd.Series:
        """Impurity-based feature importance of the trained forest, in descending order."""
        if not self.is_fitted:
            return pd.Series(dtype=float)
        cols = self._trained_columns or [c for c in FEATURE_COLUMNS if c in self._trained_columns]
        return pd.Series(self.model.feature_importances_, index=cols).sort_values(ascending=False)


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
    algo = RandomForestBaseline()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
    print(algo.feature_importances().head())
