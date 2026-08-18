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
        starting_capital: float = 1_000_000.0,
    ):
        """
        single_asset_data - {ticker: OHLCV DataFrame}, the shared data pool
        from which single-asset algorithms take their ticker, and multi-asset
        ones take an arbitrary subset of tickers.
        starting_capital  - notional account size (RUB), used to express
        each algorithm's result as money won/lost, not just percentage return.
        """
        self._data = single_asset_data
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._backtester = Backtester(
            transaction_cost_bps=transaction_cost_bps, train_frac=train_frac, starting_capital=starting_capital,
        )
        self._specs: dict[str, AlgorithmSpec] = {}

    def register(self, spec_name: str, factory: Callable[[], BaseTradingAlgorithm], ticker: str | None = None, tickers: list[str] | None = None) -> None:
        self._specs[spec_name] = AlgorithmSpec(factory=factory, ticker=ticker, tickers=tickers)

    def run_all(self, verbose: bool = True, basename: str | None = None) -> pd.DataFrame:
        """
        basename - if given, autosaves results/<basename>.{json,md} after
        EVERY algorithm finishes (not just once at the end), so a long run
        that gets interrupted doesn't lose the algorithms already completed.
        """
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

            if basename:
                self.save(pd.DataFrame(rows), basename=basename)

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
    def _money(x: float) -> str:
        return f"{x:,.0f} ₽".replace(",", " ") if pd.notnull(x) else ""

    @staticmethod
    def _to_markdown(results_df: pd.DataFrame) -> str:
        if results_df.empty:
            return "# Benchmark results\n\n(no results)\n"
        lines = [
            "# Trading algorithm benchmark results", "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        ]
        return "\n".join(lines) + BenchmarkRunner._body_markdown(results_df)

    @staticmethod
    def _body_markdown(results_df: pd.DataFrame) -> str:
        """The results table + narrative ranking sections, without a title/
        generated-at header - shared with core.market_simulator, which wraps
        it in its own live-session-specific header."""
        if results_df.empty:
            return "(no results)\n"

        display_cols = [
            "spec_name", "algorithm_name", "category", "tickers",
            "train_seconds", "inference_seconds", "n_train_rows", "n_test_rows",
            "total_return", "annualized_return", "sharpe_ratio", "max_drawdown",
            "win_rate", "hit_rate", "n_trades", "starting_capital", "final_capital",
            "pnl_rub", "error",
        ]
        display_cols = [c for c in display_cols if c in results_df.columns]
        df = results_df[display_cols].copy()

        for col in ("total_return", "annualized_return", "max_drawdown", "win_rate", "hit_rate"):
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        for col in ("sharpe_ratio", "train_seconds", "inference_seconds"):
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        for col in ("starting_capital", "final_capital", "pnl_rub"):
            if col in df.columns:
                df[col] = df[col].map(BenchmarkRunner._money)

        lines = [df.to_markdown(index=False), ""]

        ranked = results_df[results_df["error"].isna()].sort_values("sharpe_ratio", ascending=False)

        lines.append("## Summary: who performed better, who worse")
        lines.append("")
        n_total = len(results_df)
        n_errors = int(results_df["error"].notna().sum()) if "error" in results_df.columns else 0
        n_ok = n_total - n_errors
        if not ranked.empty:
            n_profitable = int((ranked["pnl_rub"] > 0).sum()) if "pnl_rub" in ranked.columns else 0
            best_sharpe = ranked.iloc[0]
            worst_sharpe = ranked.iloc[-1]
            lines.append(
                f"- {n_ok}/{n_total} algorithms completed without errors "
                f"({n_errors} failed); of those, {n_profitable}/{n_ok} ended profitable "
                f"(positive money P&L)."
            )
            lines.append(
                f"- Best by Sharpe: **{best_sharpe['algorithm_name']}** "
                f"(Sharpe {best_sharpe['sharpe_ratio']:+.3f}, "
                f"{BenchmarkRunner._money(best_sharpe.get('pnl_rub', float('nan')))})."
            )
            lines.append(
                f"- Worst by Sharpe: **{worst_sharpe['algorithm_name']}** "
                f"(Sharpe {worst_sharpe['sharpe_ratio']:+.3f}, "
                f"{BenchmarkRunner._money(worst_sharpe.get('pnl_rub', float('nan')))})."
            )
            if "pnl_rub" in ranked.columns and not ranked["pnl_rub"].isna().all():
                by_money = ranked.sort_values("pnl_rub", ascending=False)
                best_money = by_money.iloc[0]
                worst_money = by_money.iloc[-1]
                lines.append(
                    f"- Made the most money: **{best_money['algorithm_name']}** "
                    f"({BenchmarkRunner._money(best_money['pnl_rub'])}, Sharpe {best_money['sharpe_ratio']:+.3f})."
                )
                lines.append(
                    f"- Lost the most money: **{worst_money['algorithm_name']}** "
                    f"({BenchmarkRunner._money(worst_money['pnl_rub'])}, Sharpe {worst_money['sharpe_ratio']:+.3f})."
                )
        else:
            lines.append(f"- {n_ok}/{n_total} algorithms completed without errors ({n_errors} failed).")
        lines.append("")

        if not ranked.empty:
            lines.append("## Top 10 by Sharpe ratio (out-of-sample)")
            lines.append("")
            top = ranked[["algorithm_name", "sharpe_ratio", "total_return", "max_drawdown", "pnl_rub"]].head(10).copy()
            top["total_return"] = top["total_return"].map(lambda x: f"{x:.2%}")
            top["max_drawdown"] = top["max_drawdown"].map(lambda x: f"{x:.2%}")
            top["sharpe_ratio"] = top["sharpe_ratio"].map(lambda x: f"{x:.3f}")
            top["pnl_rub"] = top["pnl_rub"].map(BenchmarkRunner._money)
            lines.append(top.to_markdown(index=False))
            lines.append("")

        if "pnl_rub" in ranked.columns and not ranked.empty:
            lines.append("## Top 10 by money made (P&L, RUB)")
            lines.append("")
            by_money = ranked.sort_values("pnl_rub", ascending=False)
            top_money = by_money[["algorithm_name", "pnl_rub", "sharpe_ratio", "total_return"]].head(10).copy()
            top_money["pnl_rub"] = top_money["pnl_rub"].map(BenchmarkRunner._money)
            top_money["sharpe_ratio"] = top_money["sharpe_ratio"].map(lambda x: f"{x:.3f}")
            top_money["total_return"] = top_money["total_return"].map(lambda x: f"{x:.2%}")
            lines.append(top_money.to_markdown(index=False))
            lines.append("")

            lines.append("## Biggest losses (P&L, RUB)")
            lines.append("")
            bottom_money = by_money[["algorithm_name", "pnl_rub", "sharpe_ratio", "total_return"]].tail(10).iloc[::-1].copy()
            bottom_money["pnl_rub"] = bottom_money["pnl_rub"].map(BenchmarkRunner._money)
            bottom_money["sharpe_ratio"] = bottom_money["sharpe_ratio"].map(lambda x: f"{x:.3f}")
            bottom_money["total_return"] = bottom_money["total_return"].map(lambda x: f"{x:.2%}")
            lines.append(bottom_money.to_markdown(index=False))
            lines.append("")

        failed = results_df[results_df["error"].notna()]
        if not failed.empty:
            lines.append("## Algorithms that finished with an error")
            lines.append("")
            lines.append(failed[["algorithm_name", "error"]].to_markdown(index=False))
            lines.append("")

        return "\n".join(lines)
