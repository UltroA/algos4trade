"""
LightGBM - cross-sectional ranking / return direction forecast.

Table row #1 (LightGBM/XGBoost/CatBoost): gradient boosting over
engineered features (momentum, volatility, RSI, MACD, volume) predicts
the probability of a price increase over a `horizon`-day horizon. Features are
built in core.features.make_features - uniformly for all boosting/linear
models in the project, so benchmark results are comparable.

Realistic level (see docs/Алгоритмы.md): direction accuracy ~52-56%,
IC 0.02-0.05. The model easily overfits noisy financial features, so
shallow trees (max_depth=4) and a moderate learning rate are used here.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target


class LightGBMRanker(SingleAssetAlgorithm):
    name = "LightGBM Ranker"
    category = AlgorithmCategory.SUPERVISED_RANKING
    description = (
        "Cross-sectional ranking of stocks by expected return via gradient boosting "
        "over technical features. Primary use case - scoring price direction."
    )

    def __init__(
        self,
        horizon: int = 1,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        signal_strength: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            horizon=horizon,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            signal_strength=signal_strength,
            seed=seed,
            **kwargs,
        )
        self.horizon = horizon
        self.signal_strength = signal_strength
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            verbosity=-1,
            random_state=seed,
            **{k: v for k, v in kwargs.items() if k not in {"horizon", "n_estimators", "max_depth", "learning_rate", "signal_strength", "seed"}},
        )

    def fit(self, train_data: pd.DataFrame) -> "LightGBMRanker":
        feat_df = make_features(train_data, horizon=self.horizon)
        X, y = split_features_target(feat_df, target_col="fwd_direction")
        if X.empty or y.nunique() < 2:
            self.is_fitted = False
            return self
        self.model.fit(X, y)
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
    algo = LightGBMRanker()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
