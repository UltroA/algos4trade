"""
Correlation clustering - simple alternative to HDBSCAN for finding asset baskets.

Table row #8 (second method for the same row): instead of density-based
clustering on raw returns, a direct threshold on pairwise
correlation of returns is used - a greedy algorithm groups assets whose correlation
with a cluster's "seed" asset is above `corr_threshold`. Simpler and more deterministic
than HDBSCAN, but equally prone to cluster instability over time (see table):
correlations computed on train may not hold in the test period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.base import AlgorithmCategory, MultiAssetAlgorithm


class CorrelationClustering(MultiAssetAlgorithm):
    name = "Correlation Clustering"
    category = AlgorithmCategory.CLUSTERING
    description = (
        "Greedy clustering of assets by pairwise return correlation (corr_threshold threshold); "
        "within each group, mean reversion of the asset's price relative to the group average is traded."
    )

    def __init__(
        self,
        corr_threshold: float = 0.6,
        zscore_window: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            corr_threshold=corr_threshold, zscore_window=zscore_window,
            entry_z=entry_z, exit_z=exit_z, **kwargs,
        )
        self.corr_threshold = corr_threshold
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._cluster_of: dict[str, int] = {}

    def fit(self, train_data: dict[str, pd.DataFrame]) -> "CorrelationClustering":
        tickers = list(train_data.keys())
        returns = {}
        common_index = None
        for t in tickers:
            r = train_data[t]["close"].pct_change().dropna()
            returns[t] = r
            common_index = r.index if common_index is None else common_index.intersection(r.index)

        if common_index is None or len(common_index) < 10 or len(tickers) < 2:
            self.is_fitted = False
            return self

        returns_df = pd.DataFrame({t: returns[t].reindex(common_index) for t in tickers})
        corr = returns_df.corr()

        # Sorted list, not a set: set iteration order depends on Python's
        # per-process string hash seed (PYTHONHASHSEED, randomized by default
        # and re-rolled in every worker subprocess spawned by
        # run_all_benchmarks.py / the market simulator), so unassigned.pop()
        # picked a different "seed" ticker on every run and produced
        # different clusters from identical data - verified by bisection
        # after cross-run Sharpe swung by ~0.7 with no code change.
        unassigned = sorted(tickers)
        cluster_of: dict[str, int] = {}
        next_cluster_id = 0
        while unassigned:
            seed = unassigned.pop(0)
            members = [seed]
            for other in list(unassigned):
                if corr.loc[seed, other] > self.corr_threshold:
                    members.append(other)
                    unassigned.remove(other)

            label = next_cluster_id if len(members) >= 2 else -1
            for m in members:
                cluster_of[m] = label
            if len(members) >= 2:
                next_cluster_id += 1

        self._cluster_of = cluster_of
        self.is_fitted = True
        return self

    def _cluster_members(self, label: int, available_tickers: list[str]) -> list[str]:
        return [t for t in available_tickers if self._cluster_of.get(t, -1) == label and label != -1]

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        signals = {t: pd.Series(0.0, index=df.index) for t, df in data.items()}
        if not self.is_fitted:
            return signals

        available = list(data.keys())
        labels_present = {self._cluster_of.get(t, -1) for t in available}

        for label in labels_present:
            members = self._cluster_members(label, available)
            if len(members) < 2:
                continue

            norm_prices = {}
            common_index = None
            for m in members:
                close = data[m]["close"]
                norm = close / close.iloc[0] * 100.0
                norm_prices[m] = norm
                common_index = norm.index if common_index is None else common_index.union(norm.index)
            common_index = common_index.sort_values()

            price_matrix = pd.DataFrame({m: norm_prices[m].reindex(common_index) for m in members}).ffill()
            cluster_mean = price_matrix.mean(axis=1)

            for m in members:
                spread = price_matrix[m] - cluster_mean
                spread_mean = spread.rolling(self.zscore_window).mean()
                spread_std = spread.rolling(self.zscore_window).std()
                zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

                position = pd.Series(0.0, index=zscore.index)
                position[zscore > self.entry_z] = -1.0
                position[zscore < -self.entry_z] = 1.0
                position[zscore.abs() < self.exit_z] = 0.0
                position = position.replace(0.0, np.nan).ffill().fillna(0.0)

                signals[m] = position.reindex(data[m].index).fillna(0.0)

        return signals


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="B")

    common1 = np.cumsum(rng.normal(0, 0.01, n))
    common2 = np.cumsum(rng.normal(0, 0.008, n))

    def make_df(base, scale, noise_scale=0.004):
        price = 100 * scale * np.exp(base + rng.normal(0, noise_scale, n))
        return pd.DataFrame(
            {"open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": 1_000_000},
            index=idx,
        )

    data = {
        "A1": make_df(common1, 1.0),
        "A2": make_df(common1, 0.8),
        "A3": make_df(common1, 1.2),
        "B1": make_df(common2, 1.0),
        "B2": make_df(common2, 0.5),
        "C1": make_df(np.cumsum(rng.normal(0, 0.02, n)), 1.0),
    }

    train_data = {k: v.iloc[:280] for k, v in data.items()}
    test_data = {k: v.iloc[280:] for k, v in data.items()}

    algo = CorrelationClustering()
    algo.fit(train_data)
    print("clusters:", algo._cluster_of)
    signals = algo.generate_signals(test_data)
    for ticker, s in signals.items():
        print(ticker, "nonzero:", (s != 0).sum())
