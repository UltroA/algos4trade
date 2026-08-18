"""
XGBoost - cross-sectional ranking/prediction of return direction.

Table row #1 (LightGBM/XGBoost/CatBoost): an alternative gradient boosting
implementation over the same engineered features (momentum,
volatility, RSI, MACD, volume) as in lightgbm_ranker.py. Features are built
in core.features.make_features - uniformly for all boosting/linear
models in the project, so benchmark results are comparable.

Realistic level (see docs/Алгоритмы.md): direction accuracy ~52-56%,
IC 0.02-0.05. Overfitting on noisy financial features is mitigated by shallow
trees (max_depth=4) and row/column subsampling.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split, make_features, split_features_target


class XGBoostRanker(SingleAssetAlgorithm):
    name = "XGBoost Ranker"
    category = AlgorithmCategory.SUPERVISED_RANKING
    description = (
        "Cross-sectional ranking of stocks by expected return via XGBoost gradient boosting "
        "over technical features. Main use case - scoring the direction of price movement."
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
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            **{k: v for k, v in kwargs.items() if k not in {"horizon", "n_estimators", "max_depth", "learning_rate", "signal_strength", "seed"}},
        )

    def fit(self, train_data: pd.DataFrame) -> "XGBoostRanker":
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
    algo = XGBoostRanker()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
