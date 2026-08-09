"""
Random Position - reference baseline: position +-1 with equal
probability at each step, as the null hypothesis of "no predictive power
whatsoever".

`AlgorithmCategory.BASELINE`, not included in the total algorithm count.
The seed is fixed (seed=42) for reproducibility between runs - this is
intentionally the same random sequence on every ticker, not an
ensemble over seeds; interpret it as a single realization of the null
hypothesis, not as its expected value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split


class RandomPositionBaseline(SingleAssetAlgorithm):
    name = "Random Position Baseline"
    category = AlgorithmCategory.BASELINE
    description = (
        "Reference baseline: position +1/-1 with probability 0.5 at "
        "each step (fixed seed). The null hypothesis of no "
        "predictive power."
    )

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.seed = seed

    def fit(self, train_data: pd.DataFrame) -> "RandomPositionBaseline":
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        values = rng.choice([-1.0, 1.0], size=len(data))
        return pd.Series(values, index=data.index)


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
    algo = RandomPositionBaseline()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
