"""
Single entry point to run the benchmark for ALL algorithms living in
src/algorithms/ - without manual registration in an import list (unlike
run_benchmarks.py and run_benchmarks_wide.py, where the algorithm list is
spelled out explicitly). The script scans the folder itself, imports each
module and finds classes in it that subclass BaseTradingAlgorithm and can be
instantiated without required constructor arguments.

Classes that need another algorithm to be constructed (wrappers from
src/algorithms/composite.py: AnomalyRiskOverlay, VolTargetSizer,
ThompsonWithStrongArms - all take a *_factory as their first required
argument) are automatically skipped with a note in the output: they cannot
be meaningfully run "by default", they need a specific combination of
algorithms - see scripts/run_composite_benchmarks.py.

Tickers and costs are the same set as in the baseline experiment
(scripts/run_benchmarks.py): 10 liquid MOEX stocks, SBER as the primary
ticker for single-asset algorithms, the whole set for multi-asset ones. This
is a quick tool to check "are all the algorithms in the folder still alive
and what do they show", not a replacement for the specialized runs (wide
cross-section, holdout, walk-forward, composites) - each of which has its
own methodology.

Each algorithm is trained and tested in a SEPARATE subprocess (see --worker
mode below), not in one shared process. This is not just extra caution:
different algorithms pull in lightgbm/xgboost/catboost/torch/hmmlearn/hdbscan,
each with its own copy of native runtime bits (libomp etc.), and under
auto-discovery, modules are imported and trained in alphabetical order
rather than the hand-picked "safe" order used in run_benchmarks.py. In
practice this reliably triggers a native-code segfault around the 15th-20th
algorithm, and without subprocess isolation it takes down the entire run.
A subprocess per algorithm fixes this at the cost of a small overhead for
re-importing libraries.

The process prints a progress bar: N/total, algorithm name, current
backtest stage (fit / generate signals / metrics - see Backtester.STAGE_*,
streamed from the subprocess line by line in real time), and after each
algorithm finishes - its result (Sharpe/return/MDD, or an error message,
including a segfault/timeout of the individual subprocess).

Run: source .venv/bin/activate && python scripts/run_all_benchmarks.py
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Standard fix for segfaults caused by repeated initialization of
# libomp.dylib when mixing lightgbm/xgboost/catboost/torch in one process.
# Must be set BEFORE the first import of any of these libraries. Not
# strictly needed in the main mode (each algorithm is already isolated in
# its own subprocess), but doesn't hurt; in --worker mode the same
# precaution is applied just in case.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "ROSN", "NVTK", "MTSS", "TATN", "MOEX"]
PRIMARY_TICKER = "SBER"
START_ISO = "2019-01-01"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

STAGE_PREFIX = "STAGE\t"
RESULT_PREFIX = "RESULT\t"


# ---------------------------------------------------------------------------
# Algorithm discovery (safe to do in the main process - segfaults are
# triggered by model training, not by merely importing a module). Shared
# with scripts/run_market_simulation.py via core/algorithm_discovery.py.
# ---------------------------------------------------------------------------

from core.algorithm_discovery import discover_algorithms  # noqa: E402

STARTING_CAPITAL_RUB = 1_000_000.0

"""
--worker mode: trains and tests ONE algorithm in a separate process, prints
stages and the final result to stdout line by line so the parent can show
progress in real time.
"""
def _run_worker(modname: str, clsname: str) -> None:
    from core.backtester import Backtester
    from core.base import InputMode
    from core.providers.tinvest import TInvestDataLoader

    module = importlib.import_module(f"algorithms.{modname}")
    cls = getattr(module, clsname)
    algo = cls()

    loader = TInvestDataLoader()
    start = datetime.fromisoformat(START_ISO).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    if algo.input_mode == InputMode.SINGLE_ASSET:
        run_data = loader.load_candles(PRIMARY_TICKER, start, end, interval="day", use_cache=True)
        algo.ticker = PRIMARY_TICKER  # type: ignore[attr-defined]
    else:
        run_data = loader.load_many(TICKERS, start, end, interval="day")

    def on_stage(stage: str) -> None:
        print(f"{STAGE_PREFIX}{stage}", flush=True)

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7, starting_capital=STARTING_CAPITAL_RUB)
    result = backtester.run(algo, run_data, on_stage=on_stage)

    row = {"spec_name": modname, "class_name": clsname, "algo_display_name": algo.name}
    row.update(result.to_flat_dict())
    print(f"{RESULT_PREFIX}{json.dumps(row, default=str)}", flush=True)


class ProgressBar:
    def __init__(self, total: int, width: int = 24):
        self.total = total
        self.width = width
        self.done = 0

    def stage(self, name: str, stage: str) -> None:
        filled = int(self.width * self.done / self.total) if self.total else self.width
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r[{bar}] {self.done}/{self.total}  {name[:38]:<38}  {stage}"
        sys.stdout.write(line.ljust(110))
        sys.stdout.flush()

    def finish(self, result_line: str) -> None:
        self.done += 1
        sys.stdout.write("\r" + " " * 110 + "\r")
        print(result_line)
        sys.stdout.flush()


def format_result(algo_name: str, row: dict) -> str:
    if row.get("error"):
        return f"  {algo_name:<40} ERROR: {row['error']}"
    return (
        f"  {algo_name:<40} SR={row['sharpe_ratio']:+.3f}  return={row['total_return']:+.2%}  "
        f"MDD={row['max_drawdown']:.2%}  trades={row['n_trades']}"
    )


def _run_one_algorithm(modname: str, clsname: str, bar: ProgressBar) -> dict:
    """Runs the worker in a subprocess, streams its STAGE lines into the
    progress bar in real time, returns the final result row (or a
    synthesized error if the subprocess died without a RESULT)."""
    bar.stage(clsname, "starting subprocess")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", modname, clsname],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    result_row: dict | None = None
    display_name = clsname
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip("\n")
        if line.startswith(STAGE_PREFIX):
            bar.stage(display_name, line[len(STAGE_PREFIX):])
        elif line.startswith(RESULT_PREFIX):
            result_row = json.loads(line[len(RESULT_PREFIX):])
            display_name = result_row.get("algo_display_name", clsname)
    proc.wait()
    stderr_tail = (proc.stderr.read() or "").strip().splitlines()[-1:] if proc.stderr else []

    if result_row is None:
        reason = (
            f"subprocess terminated by signal {-proc.returncode} (likely a native library segfault)"
            if proc.returncode is not None and proc.returncode < 0
            else f"subprocess exited with code {proc.returncode}"
        )
        if stderr_tail:
            reason += f" - {stderr_tail[0]}"
        result_row = {
            "spec_name": modname, "class_name": clsname, "algo_display_name": clsname,
            "algorithm_name": clsname, "category": "", "tickers": "", "train_seconds": 0.0,
            "inference_seconds": 0.0, "n_train_rows": 0, "n_test_rows": 0, "error": reason,
            "total_return": None, "annualized_return": None, "annualized_volatility": None,
            "sharpe_ratio": None, "max_drawdown": None, "win_rate": None, "hit_rate": None, "n_trades": None,
        }
    return result_row


def main() -> None:
    found, skipped = discover_algorithms()
    print(f"Algorithms found in src/algorithms/: {len(found)}")
    if skipped:
        print(f"Skipped (require required constructor arguments, see run_composite_benchmarks.py): {len(skipped)}")
        for modname, clsname in skipped:
            print(f"  - {modname}.{clsname}")
    print()

    bar = ProgressBar(total=len(found))
    rows: list[dict] = []
    for modname, clsname in found:
        row = _run_one_algorithm(modname, clsname, bar)
        rows.append(row)
        bar.finish(format_result(row.get("algo_display_name", clsname), row))
        # Autosave after every algorithm - a run killed partway through
        # (segfault storm, Ctrl-C, machine sleep) still leaves everything
        # completed so far in results/all_benchmark_results.{json,md}.
        _save_results(rows, found, skipped)

    print("\nSaved results/all_benchmark_results.json and .md")

    n_errors = sum(1 for r in rows if r.get("error"))
    if n_errors:
        print(f"Finished with errors: {n_errors}/{len(rows)}")


def _save_results(rows: list[dict], found: list[tuple[str, str]], skipped: list[tuple[str, str]]) -> None:
    results_df = pd.DataFrame(rows).drop(columns=["algo_display_name", "class_name"], errors="ignore")
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_algorithms_found": len(found),
        "n_algorithms_skipped": len(skipped),
        "n_algorithms_completed": len(rows),
        "skipped": [f"{modname}.{clsname}" for modname, clsname in skipped],
        "results": results_df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "all_benchmark_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    from core.benchmark_runner import BenchmarkRunner
    (RESULTS_DIR / "all_benchmark_results.md").write_text(
        BenchmarkRunner._to_markdown(results_df), encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2], sys.argv[3])
    else:
        main()
