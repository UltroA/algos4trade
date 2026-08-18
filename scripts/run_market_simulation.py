"""
Live market simulator entry point: runs every auto-discovered algorithm
(same discovery as scripts/run_all_benchmarks.py) against real MOEX/T-Invest
data, paced either in real time (default) or in an accelerated demo replay.

Live mode (default): each new candle is fetched from T-Invest the moment it
happens - the session's wall-clock pacing IS the real network latency (no
delay is simulated). Runs for the configured trading-session window (see
configs/market_simulator.json: session_start/session_end MSK, trading_days,
max_duration_seconds), autosaving progress continuously, and writes
results/<basename>.{json,md} - a detailed per-algorithm stats table, a
"who performed better/worse" ranking, money P&L, and T-Invest latency stats
- when the session ends (or is interrupted with Ctrl-C).

Demo mode (--demo): replays already-fetched recent bars with a synthetic
per-tick delay calibrated from real measured T-Invest latency by default, so
the whole pipeline can be exercised in minutes without waiting for real
market hours - or pass --demo-speed-ms to go faster still (0 = no delay at
all, replay as fast as the machine can fit/tick each algorithm). Both demo
and live mode fit each algorithm in its own subprocess (see
core/market_simulator._fit_isolated), the same isolation
scripts/run_all_benchmarks.py uses to avoid native-library segfaults when
mixing many ML libraries in one process.

Usage:
    python scripts/run_market_simulation.py                        # live, all algorithms, configs/market_simulator.json
    python scripts/run_market_simulation.py --config path.json
    python scripts/run_market_simulation.py --duration 3600          # cap the live session at 1h
    python scripts/run_market_simulation.py --demo --duration 60 --algorithms buy_and_hold_baseline,sma_crossover_baseline
    python scripts/run_market_simulation.py --demo --demo-speed-ms 0 --duration 30   # fastest possible replay
"""

import argparse
import os
import sys
from pathlib import Path

# Both demo and live mode fit each algorithm in its own --fit-worker
# subprocess (see core/market_simulator._fit_isolated), but the unpickled,
# fitted instances are all imported back into THIS long-lived process so it
# can generate signals for the whole session - so it ends up with every
# algorithm's native runtime (lightgbm, xgboost, catboost, torch, hmmlearn,
# hdbscan, ...) loaded simultaneously. KMP_DUPLICATE_LIB_OK=TRUE keeps that
# from hard-aborting on libomp's duplicate-init check. (An earlier attempt
# at also pinning OMP_NUM_THREADS/MKL_NUM_THREADS/etc. to 1 to avoid
# thread-pool races made things worse - it triggered a separate, more
# immediate crash: "OMP: Error #179: Function pthread_mutex_init failed",
# a known Intel OpenMP runtime bug on newer macOS - so don't reintroduce
# that without confirming it's actually safe on the target OS/library
# versions first.)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# lightgbm specifically (confirmed by bisection - see the SIGSEGV writeup in
# git history/PR notes for this fix) crashes when its Booster gets unpickled
# back into this process AFTER other algorithms' native libraries are
# already resident, e.g. via core.algorithm_discovery.discover_algorithms()
# importing every module in src/algorithms/ alphabetically (catboost before
# lightgbm) to build the run list - even though the actual fit happens in an
# isolated --fit-worker subprocess, unpickling the fitted Booster back here
# still exercises lightgbm's native code for the first time in this process,
# and whichever OpenMP runtime got resident first "wins" in a way lightgbm's
# does not tolerate. Importing lightgbm before anything else sidesteps this
# entirely (reproduced: crashes when imported after discovery, fine when
# imported first) - cheaper and more reliable than hunting down every other
# import-order-sensitive pair.
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from core.algorithm_discovery import discover_algorithms  # noqa: E402
from core.market_simulator import MarketSimulator, SessionConfig  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "market_simulator.json"


def _load_config(args: argparse.Namespace) -> SessionConfig:
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config = SessionConfig.from_file(config_path) if config_path.exists() else SessionConfig()
    if args.duration is not None:
        config.max_duration_seconds = args.duration
    if args.warmup_days is not None:
        config.warmup_days = args.warmup_days
    if args.basename:
        config.basename = args.basename
    if args.demo_speed_ms is not None:
        config.demo_speed_ms = args.demo_speed_ms
    return config


def _select_algorithms(names_filter: str | None) -> list[tuple[str, str]]:
    found, skipped = discover_algorithms()
    if skipped:
        print(f"Skipped (need another algorithm to construct, see run_composite_benchmarks.py): {len(skipped)}")
        for modname, clsname in skipped:
            print(f"  - {modname}.{clsname}")
    if names_filter:
        wanted = {n.strip() for n in names_filter.split(",") if n.strip()}
        found = [(m, c) for m, c in found if m in wanted]
        missing = wanted - {m for m, _ in found}
        if missing:
            print(f"Warning: not found among discovered algorithms, ignored: {sorted(missing)}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help=f"Path to a SessionConfig JSON file (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--demo", action="store_true", help="Accelerated replay of recent bars instead of real-time")
    parser.add_argument("--duration", type=float, default=None, help="Override max_duration_seconds")
    parser.add_argument("--lookback-days", type=float, default=5.0, help="--demo only: how many recent days to replay")
    parser.add_argument(
        "--demo-speed-ms", type=float, default=None,
        help="--demo only: override demo_speed_ms - synthetic delay per replayed tick. "
             "0 = no delay (fastest possible, limited only by fit/tick compute time). "
             "Omit to use the config value, or real measured T-Invest latency if that's also unset.",
    )
    parser.add_argument("--warmup-days", type=float, default=None, help="Override warmup_days (history pulled to fit() before going live)")
    parser.add_argument("--algorithms", default=None, help="Comma-separated module names to run (default: all discovered)")
    parser.add_argument("--basename", default=None, help="Override results/<basename>.{json,md}")
    args = parser.parse_args()

    config = _load_config(args)
    found = _select_algorithms(args.algorithms)
    print(f"Running {len(found)} algorithms in {'DEMO' if args.demo else 'LIVE'} mode "
          f"on tickers {list(config.tickers)} ({config.interval} candles)\n")

    simulator = MarketSimulator(config)
    specs = {modname: (modname, clsname) for modname, clsname in found}

    if args.demo:
        simulator.run_replay_demo(specs=specs, lookback_days=args.lookback_days)
    else:
        simulator.run_live(specs=specs)


if __name__ == "__main__":
    main()
