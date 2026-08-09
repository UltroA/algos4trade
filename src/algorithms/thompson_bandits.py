"""
Многорукие бандиты / Thompson sampling - распределение капитала между стратегиями.

Сам по себе не генерирует торговый сигнал (главная
слабость из таблицы), а решает, какой из уже существующих "рукавов" (здесь -
простая momentum-стратегия по каждому тикеру из data) получает капитал в
следующем периоде. Каждый актив = одна "рука". Апостериорное распределение
ожидаемой доходности руки - Beta(успехи, неудачи) на основе знака дневной
доходности momentum-сигнала; на каждом шаге сэмплируется по одному значению
на руку (Thompson sampling), капитал распределяется на активы с наибольшим
сэмплом (soft allocation через softmax сэмплов, чтобы не концентрировать
100% в одном активе и не требовать целочисленных лотов).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.base import AlgorithmCategory, MultiAssetAlgorithm


class ThompsonSamplingAllocator(MultiAssetAlgorithm):
    name = "Thompson Sampling Capital Allocator"
    category = AlgorithmCategory.CAPITAL_ALLOCATION
    description = (
        "Байесовский многорукий бандит (Thompson sampling) распределяет капитал между "
        "momentum-стратегиями по разным активам, отдавая предпочтение тем, что чаще выигрывали."
    )

    def __init__(self, momentum_window: int = 10, seed: int = 0, max_weight: float = 1.0, **kwargs):
        super().__init__(momentum_window=momentum_window, seed=seed, max_weight=max_weight, **kwargs)
        self.momentum_window = momentum_window
        self.max_weight = max_weight
        self._rng = np.random.default_rng(seed)
        # Параметры Beta-апостериора по каждой "руке" (тикеру): alpha=успехи+1, beta=неудачи+1.
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}

    @staticmethod
    def _momentum_signal(df: pd.DataFrame, window: int) -> pd.Series:
        """Базовая стратегия внутри каждой руки: лонг, если моментум за window дней положительный."""
        momentum = df["close"].pct_change(window)
        return np.sign(momentum).fillna(0.0)

    def fit(self, train_data: dict[str, pd.DataFrame]) -> "ThompsonSamplingAllocator":
        for ticker, df in train_data.items():
            arm_signal = self._momentum_signal(df, self.momentum_window)
            arm_return = arm_signal.shift(1) * df["close"].pct_change()
            wins = int((arm_return > 0).sum())
            losses = int((arm_return < 0).sum())
            self._alpha[ticker] = wins + 1.0
            self._beta[ticker] = losses + 1.0
        self.is_fitted = True
        return self

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        if not self.is_fitted:
            return {t: pd.Series(0.0, index=df.index) for t, df in data.items()}

        tickers = list(data.keys())
        base_signals = {t: self._momentum_signal(data[t], self.momentum_window) for t in tickers}
        common_index = base_signals[tickers[0]].index
        for s in base_signals.values():
            common_index = common_index.union(s.index)
        common_index = common_index.sort_values()

        alpha = {t: self._alpha.get(t, 1.0) for t in tickers}
        beta_param = {t: self._beta.get(t, 1.0) for t in tickers}

        weights_over_time: dict[str, list[float]] = {t: [] for t in tickers}
        for _ in common_index:
            samples = np.array([self._rng.beta(alpha[t], beta_param[t]) for t in tickers])
            exp_samples = np.exp((samples - samples.max()) * 5.0)  # температура 5.0 заостряет выбор
            weights = exp_samples / exp_samples.sum()
            for t, w in zip(tickers, weights):
                weights_over_time[t].append(min(w, self.max_weight))

        signals: dict[str, pd.Series] = {}
        for t in tickers:
            direction = base_signals[t].reindex(common_index).fillna(0.0)
            weight = pd.Series(weights_over_time[t], index=common_index)
            signals[t] = (direction * weight).reindex(data[t].index).fillna(0.0).clip(-1.0, 1.0)
        return signals


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="B")

    def make_df(drift):
        price = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
        return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1_000_000}, index=idx)

    data = {"A": make_df(0.001), "B": make_df(-0.0005), "C": make_df(0.0002)}
    train_data = {k: v.iloc[:210] for k, v in data.items()}
    test_data = {k: v.iloc[210:] for k, v in data.items()}

    algo = ThompsonSamplingAllocator()
    algo.fit(train_data)
    signals = algo.generate_signals(test_data)
    for ticker, s in signals.items():
        print(ticker, s.describe())
