"""
HDBSCAN clustering - finding pairs/baskets for statistical arbitrage.

HDBSCAN clustering - assets are grouped by similarity of their log-return
time series using density-based HDBSCAN clustering (unlike k-means, it does
not require specifying the number of clusters in advance and can label
"noise" assets that do not belong to any cluster with -1). Within each
discovered cluster, the mean-reversion of an asset's spread relative to the
cluster average is traded.

Main weakness (see table): clusters are unstable over time - membership
changes from refit to refit, so here clustering is computed once on train
and stays fully static on test (no adaptation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN

from core.base import AlgorithmCategory, MultiAssetAlgorithm


class HDBSCANPairsClustering(MultiAssetAlgorithm):
    name = "HDBSCAN Pairs/Basket Clustering"
    category = AlgorithmCategory.CLUSTERING
    description = (
        "Density-based HDBSCAN clustering over log-returns finds groups of similar assets; "
        "within each group, the mean-reversion of an asset's price relative to the group average is traded."
    )

    def __init__(
        self,
        min_cluster_size: int = 2,
        lookback: int = 60,
        zscore_window: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            min_cluster_size=min_cluster_size, lookback=lookback, zscore_window=zscore_window,
            entry_z=entry_z, exit_z=exit_z, **kwargs,
        )
        self.min_cluster_size = min_cluster_size
        self.lookback = lookback
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._cluster_of: dict[str, int] = {}

    def fit(self, train_data: dict[str, pd.DataFrame]) -> "HDBSCANPairsClustering":
        tickers = list(train_data.keys())
        log_returns = {}
        common_index = None
        for t in tickers:
            lr = np.log(train_data[t]["close"]).diff().dropna()
            log_returns[t] = lr
            common_index = lr.index if common_index is None else common_index.intersection(lr.index)

        if common_index is None or len(common_index) < 10 or len(tickers) < self.min_cluster_size:
            self.is_fitted = False
            return self

        # cluster on a recent window rather than the full history: reduces dimensionality
        # (curse of dimensionality breaks HDBSCAN on long series with few assets)
        # and is also a partial answer to "clusters are unstable over time" -
        # cluster membership is recomputed from current, not stale, correlations.
        common_index = common_index.sort_values()[-self.lookback :]
        matrix = np.vstack([log_returns[t].reindex(common_index).to_numpy() for t in tickers])

        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(matrix)
        self._cluster_of = dict(zip(tickers, labels.tolist()))
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
                continue  # asset with no pair in its cluster -> signal stays 0

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
                position[zscore > self.entry_z] = -1.0   # asset pricier than its group -> short (wait for reversion down)
                position[zscore < -self.entry_z] = 1.0    # asset cheaper than its group -> long
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
        "C1": make_df(np.cumsum(rng.normal(0, 0.02, n)), 1.0),  # uncorrelated "noise" asset
    }

    train_data = {k: v.iloc[:280] for k, v in data.items()}
    test_data = {k: v.iloc[280:] for k, v in data.items()}

    algo = HDBSCANPairsClustering()
    algo.fit(train_data)
    print("clusters:", algo._cluster_of)
    signals = algo.generate_signals(test_data)
    for ticker, s in signals.items():
        print(ticker, s.describe())
