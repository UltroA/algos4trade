"""
Single entry point for running benchmarks across all registered algorithms.

Collects :class:`~core.backtester.Backtester` results for each algorithm
into one table and saves it to results/benchmark_results.json and .md -
this is the "separate file with all benchmark results".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .backtester import Backtester, BacktestResult
from .base import BaseTradingAlgorithm, InputMode


@dataclass
class AlgorithmSpec:
    """Registration record: how to build the algorithm and what data it needs."""

    factory: Callable[[], BaseTradingAlgorithm]
    ticker: str | None = None            # for InputMode.SINGLE_ASSET
    tickers: list[str] | None = None     # for InputMode.MULTI_ASSET


class BenchmarkRunner:
    def __init__(
        self,
        single_asset_data: dict[str, pd.DataFrame],
        results_dir: str | Path = "results",
        transaction_cost_bps: float = 5.0,
        train_frac: float = 0.7,
    ):
        """
        single_asset_data - {ticker: OHLCV DataFrame}, the shared data pool
        from which single-asset algorithms take their ticker, and multi-asset
        ones take an arbitrary subset of tickers.
        """
        self._data = single_asset_data
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._backtester = Backtester(transaction_cost_bps=transaction_cost_bps, train_frac=train_frac)
        self._specs: dict[str, AlgorithmSpec] = {}

    def register(self, spec_name: str, factory: Callable[[], BaseTradingAlgorithm], ticker: str | None = None, tickers: list[str] | None = None) -> None:
        self._specs[spec_name] = AlgorithmSpec(factory=factory, ticker=ticker, tickers=tickers)

    def run_all(self, verbose: bool = True) -> pd.DataFrame:
        rows: list[dict] = []
        for spec_name, spec in self._specs.items():
            algo = spec.factory()
            if algo.input_mode == InputMode.SINGLE_ASSET:
                ticker = spec.ticker or next(iter(self._data))
                data = self._data[ticker]
                algo.ticker = ticker
            else:
                tickers = spec.tickers or list(self._data.keys())
                data = {t: self._data[t] for t in tickers}

            if verbose:
                print(f"[benchmark] running {spec_name} ...")
            result = self._backtester.run(algo, data)
            row = {"spec_name": spec_name}
            row.update(result.to_flat_dict())
            rows.append(row)
            if verbose:
                status = "ERROR: " + result.error if result.error else f"sharpe={result.metrics.sharpe_ratio:.2f}"
                print(f"[benchmark] {spec_name} done in {result.train_seconds + result.inference_seconds:.2f}s -> {status}")

        return pd.DataFrame(rows)

    def save(self, results_df: pd.DataFrame, basename: str = "benchmark_results") -> None:
        json_path = self._results_dir / f"{basename}.json"
        md_path = self._results_dir / f"{basename}.md"

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": results_df.to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        md_path.write_text(self._to_markdown(results_df), encoding="utf-8")

    @staticmethod
    def _to_markdown(results_df: pd.DataFrame) -> str:
        if results_df.empty:
            return "# Benchmark results\n\n(no results)\n"

        display_cols = [
            "spec_name", "algorithm_name", "category", "tickers",
            "train_seconds", "inference_seconds", "n_train_rows", "n_test_rows",
            "total_return", "annualized_return", "sharpe_ratio", "max_drawdown",
            "win_rate", "hit_rate", "n_trades", "error",
        ]
        display_cols = [c for c in display_cols if c in results_df.columns]
        df = results_df[display_cols].copy()

        for col in ("total_return", "annualized_return", "max_drawdown", "win_rate", "hit_rate"):
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        for col in ("sharpe_ratio", "train_seconds", "inference_seconds"):
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

        lines = ["# Trading algorithm benchmark results", ""]
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")

        ranked = results_df[results_df["error"].isna()].sort_values("sharpe_ratio", ascending=False)
        if not ranked.empty:
            lines.append("## Top 10 by Sharpe ratio (out-of-sample)")
            lines.append("")
            top = ranked[["algorithm_name", "sharpe_ratio", "total_return", "max_drawdown"]].head(10).copy()
            top["total_return"] = top["total_return"].map(lambda x: f"{x:.2%}")
            top["max_drawdown"] = top["max_drawdown"].map(lambda x: f"{x:.2%}")
            top["sharpe_ratio"] = top["sharpe_ratio"].map(lambda x: f"{x:.3f}")
            lines.append(top.to_markdown(index=False))
            lines.append("")

        failed = results_df[results_df["error"].notna()]
        if not failed.empty:
            lines.append("## Algorithms that finished with an error")
            lines.append("")
            lines.append(failed[["algorithm_name", "error"]].to_markdown(index=False))
            lines.append("")

        return "\n".join(lines)
