"""
SMA Crossover - a bare trend signal without a risk overlay.

A reference baseline (`AlgorithmCategory.BASELINE`), not a full-fledged
algorithm from docs/Алгоритмы.md. Reproduces exactly the same SMA(10 vs 50)
crossover that is baked into `IsolationForestRiskSwitch`/`OneClassSVMRiskSwitch`
as the "signal before the overlay" - needed so the summary tables can
directly verify whether a risk overlay on top of this signal is actually
better than the signal alone, rather than just assuming it without comparison.
"""

from __future__ import annotations

import pandas as pd

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.features import chronological_split


class SMACrossoverBaseline(SingleAssetAlgorithm):
    name = "SMA Crossover Baseline"
    category = AlgorithmCategory.BASELINE
    description = (
        "Reference baseline: SMA(10) above SMA(50) -> long, otherwise short. "
        "No risk overlay and no training on data - a direct comparison to "
        "check what an overlay on top of this same signal actually gives."
    )

    def __init__(self, fast_window: int = 10, slow_window: int = 50, **kwargs):
        super().__init__(fast_window=fast_window, slow_window=slow_window, **kwargs)
        self.fast_window = fast_window
        self.slow_window = slow_window

    def fit(self, train_data: pd.DataFrame) -> "SMACrossoverBaseline":
        self.is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        sma_fast = data["close"].rolling(self.fast_window).mean()
        sma_slow = data["close"].rolling(self.slow_window).mean()
        signal = (sma_fast > sma_slow).astype(float) * 2 - 1
        return signal.fillna(0.0).clip(-1.0, 1.0)


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
    algo = SMACrossoverBaseline()
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
