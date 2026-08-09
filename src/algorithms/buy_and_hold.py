"""
Buy-and-Hold - reference baseline: a constant maximum long
position over the entire test period, without any signal.

`AlgorithmCategory.BASELINE`, not one of the core algorithms.
Needed as the lower bound of comparison: any actively trading algorithm should
at minimum explain why it is better than simply holding the asset (accounting for
the single entry trade and its costs).
"""

from __future__ import annotations

import pandas as pd

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split


class BuyAndHoldBaseline(SingleAssetAlgorithm):
    name = "Buy and Hold Baseline"
    category = AlgorithmCategory.BASELINE
    description = (
        "Reference baseline: position +1 over the entire test period "
        "(one entry trade). Does not use training data."
    )

    def fit(self, train_data: pd.DataFrame) -> "BuyAndHoldBaseline":
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index)


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
    algo = BuyAndHoldBaseline()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
