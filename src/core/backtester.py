"""
Backtester: runs any :class:`~core.base.BaseTradingAlgorithm` on
historical data, computes the strategy's returns net of transaction costs,
and metrics from :mod:`core.metrics`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .base import BaseTradingAlgorithm, InputMode
from .features import chronological_split
from .metrics import BacktestMetrics, compute_hit_rate, compute_metrics


@dataclass
class BacktestResult:
    algorithm_name: str
    category: str
    tickers: list[str]
    train_seconds: float
    inference_seconds: float
    n_train_rows: int
    n_test_rows: int
    metrics: BacktestMetrics
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_flat_dict(self) -> dict:
        d = {
            "algorithm_name": self.algorithm_name,
            "category": self.category,
            "tickers": ",".join(self.tickers),
            "train_seconds": round(self.train_seconds, 4),
            "inference_seconds": round(self.inference_seconds, 4),
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "error": self.error,
        }
        d.update(self.metrics.to_dict())
        return d


class Backtester:
    """
    transaction_cost_bps - cost of a position change, in basis points
    (typical MOEX commission+spread for blue-chip stocks is ~5-10 bp per trade).
    """

    # Stages of a single backtester run - pulled out into constants so that
    # calling code (e.g. the progress bar in scripts/run_all_benchmarks.py) can
    # check against them instead of hardcoding the strings again.
    STAGE_FIT = "training"
    STAGE_SIGNALS = "signal generation"
    STAGE_METRICS = "metrics"

    def __init__(self, transaction_cost_bps: float = 5.0, train_frac: float = 0.7):
        self.transaction_cost_bps = transaction_cost_bps
        self.train_frac = train_frac

    def run(
        self,
        algo: BaseTradingAlgorithm,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        on_stage: Callable[[str], None] | None = None,
    ) -> BacktestResult:
        """on_stage(stage) - optional callback, called before each stage
        (see STAGE_* above) with that stage's name; used only for progress
        indication, does not affect the backtest result."""
        try:
            if algo.input_mode == InputMode.SINGLE_ASSET:
                return self._run_single(algo, data, on_stage)
            return self._run_multi(algo, data, on_stage)
        except Exception as exc:
            return BacktestResult(
                algorithm_name=getattr(algo, "name", algo.__class__.__name__),
                category=str(getattr(algo, "category", "")),
                tickers=[],
                train_seconds=0.0,
                inference_seconds=0.0,
                n_train_rows=0,
                n_test_rows=0,
                metrics=compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float), hit_rate=float("nan")),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_single(
        self, algo: BaseTradingAlgorithm, data: pd.DataFrame, on_stage: Callable[[str], None] | None = None,
    ) -> BacktestResult:
        train_df, test_df = chronological_split(data, self.train_frac)

        if on_stage:
            on_stage(self.STAGE_FIT)
        t0 = time.perf_counter()
        algo.fit(train_df)
        train_seconds = time.perf_counter() - t0

        if on_stage:
            on_stage(self.STAGE_SIGNALS)
        t0 = time.perf_counter()
        signals = algo.generate_signals(test_df)
        inference_seconds = time.perf_counter() - t0

        if on_stage:
            on_stage(self.STAGE_METRICS)
        signals = signals.reindex(test_df.index).fillna(0.0).clip(-1.0, 1.0)
        strategy_returns = self._strategy_returns(test_df["close"], signals)
        asset_returns = test_df["close"].pct_change().fillna(0.0)
        position_lagged = signals.shift(1).fillna(0.0)
        hit_rate = compute_hit_rate(position_lagged, asset_returns)
        metrics = compute_metrics(strategy_returns, signals, hit_rate=hit_rate)

        return BacktestResult(
            algorithm_name=algo.name,
            category=str(algo.category.value if hasattr(algo.category, "value") else algo.category),
            tickers=[getattr(algo, "ticker", "N/A")],
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
            n_train_rows=len(train_df),
            n_test_rows=len(test_df),
            metrics=metrics,
        )

    def _run_multi(
        self, algo: BaseTradingAlgorithm, data: dict[str, pd.DataFrame], on_stage: Callable[[str], None] | None = None,
    ) -> BacktestResult:
        train_data, test_data = {}, {}
        for ticker, df in data.items():
            train_df, test_df = chronological_split(df, self.train_frac)
            train_data[ticker] = train_df
            test_data[ticker] = test_df

        if on_stage:
            on_stage(self.STAGE_FIT)
        t0 = time.perf_counter()
        algo.fit(train_data)
        train_seconds = time.perf_counter() - t0

        if on_stage:
            on_stage(self.STAGE_SIGNALS)
        t0 = time.perf_counter()
        signals_by_ticker = algo.generate_signals(test_data)
        inference_seconds = time.perf_counter() - t0

        if on_stage:
            on_stage(self.STAGE_METRICS)

        per_asset_returns = []
        hr_positions, hr_asset_returns = [], []
        for ticker, signals in signals_by_ticker.items():
            close = test_data[ticker]["close"]
            signals = signals.reindex(close.index).fillna(0.0).clip(-1.0, 1.0)
            per_asset_returns.append(self._strategy_returns(close, signals))
            asset_returns = close.pct_change().fillna(0.0)
            hr_positions.append(signals.shift(1).fillna(0.0).reset_index(drop=True))
            hr_asset_returns.append(asset_returns.reset_index(drop=True))

        portfolio_returns = pd.concat(per_asset_returns, axis=1).mean(axis=1) if per_asset_returns else pd.Series(dtype=float)
        # absolute value, not mean: hedged strategies (e.g. pairs trading) have
        # legs with opposite signs, and a naive mean would give a perpetual zero.
        all_signals = pd.concat(signals_by_ticker.values(), axis=1).abs().mean(axis=1) if signals_by_ticker else pd.Series(dtype=float)
        # HR is computed on observations (ticker, day) pooled across all tickers,
        # not on the aggregated portfolio signal - otherwise the formula loses
        # meaning for hedged strategies (see the abs() note above).
        hit_rate = (
            compute_hit_rate(pd.concat(hr_positions, ignore_index=True), pd.concat(hr_asset_returns, ignore_index=True))
            if hr_positions else float("nan")
        )
        metrics = compute_metrics(portfolio_returns, all_signals, hit_rate=hit_rate)

        n_train_rows = sum(len(df) for df in train_data.values())
        n_test_rows = sum(len(df) for df in test_data.values())

        return BacktestResult(
            algorithm_name=algo.name,
            category=str(algo.category.value if hasattr(algo.category, "value") else algo.category),
            tickers=list(data.keys()),
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
            n_train_rows=n_train_rows,
            n_test_rows=n_test_rows,
            metrics=metrics,
        )

    def _strategy_returns(self, close: pd.Series, signals: pd.Series) -> pd.Series:
        """Return = yesterday's position * today's price return - cost of the position change."""
        asset_returns = close.pct_change().fillna(0.0)
        position = signals.shift(1).fillna(0.0)
        gross_returns = position * asset_returns
        turnover = signals.diff().abs().fillna(signals.abs())
        costs = turnover * (self.transaction_cost_bps / 10_000.0)
        return gross_returns - costs
