"""
HDBSCAN-кластеризация - поиск пар/корзин для статистического арбитража.

HDBSCAN-кластеризация -  активы группируются по схожести временных рядов
лог-доходностей плотностной кластеризацией HDBSCAN (в отличие от k-means,
не требует заранее задавать число кластеров и умеет помечать "шумовые",
не входящие ни в один кластер активы меткой -1). Внутри каждого найденного
кластера торгуется mean-reversion спреда актива относительно среднего
по кластеру.

Главная слабость (см. таблицу): кластеры нестабильны во времени - состав
меняется от переобучения к переобучению, поэтому здесь кластеризация
считается один раз на train и полностью статична на test (без адаптации).
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
        "Плотностная кластеризация HDBSCAN по лог-доходностям находит группы похожих активов; "
        "внутри каждой группы торгуется возврат к среднему цены актива относительно среднего по группе."
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

        # кластеризуем по недавнему окну, а не по всей истории: снижает размерность
        # (curse of dimensionality валит HDBSCAN на длинных рядах при малом числе активов)
        # и одновременно является частичным ответом на "кластеры нестабильны во времени" -
        # состав кластеров пересчитывается от актуальных, а не устаревших корреляций.
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
                continue  # актив без пары в своём кластере -> сигнал остаётся 0

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
                position[zscore > self.entry_z] = -1.0   # актив дороже своей группы -> шорт (ждём возврата вниз)
                position[zscore < -self.entry_z] = 1.0    # актив дешевле своей группы -> лонг
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
        "C1": make_df(np.cumsum(rng.normal(0, 0.02, n)), 1.0),  # некоррелированный "шумовой" актив
    }

    train_data = {k: v.iloc[:280] for k, v in data.items()}
    test_data = {k: v.iloc[280:] for k, v in data.items()}

    algo = HDBSCANPairsClustering()
    algo.fit(train_data)
    print("clusters:", algo._cluster_of)
    signals = algo.generate_signals(test_data)
    for ticker, s in signals.items():
        print(ticker, s.describe())
