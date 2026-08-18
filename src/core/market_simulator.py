"""
Live market simulator: replays real MOEX/T-Invest data through the whole
algorithm suite the way a live paper-trading bot would experience it, not
the way the fast vectorized Backtester does.

Two differences from core.backtester.Backtester that matter here:

  1. Timing is real. In `run_live()`, every "new candle" is a genuine
     T-Invest API response fetched at the moment it happens (via
     TInvestDataLoader.load_recent(use_cache=False)) - there is no injected/
     fake delay, the wall-clock pacing IS the real network latency, recorded
     by TInvestClient and reported in the final report (LatencyTracker).
     `run_replay_demo()` exists only so the whole pipeline can be exercised
     in minutes instead of waiting for real market hours: it replays already
     -fetched recent bars and sleeps a synthetic delay per tick, calibrated
     from that same real measured latency (or a config override) rather than
     a guessed constant.
  2. Money is tracked with an actual ledger (SimulatedBroker: cash + shares
     per ticker), not derived algebraically from a returns formula - so the
     "how much money did this algorithm make/lose" number is something that
     was actually computed step by step, the way a broker statement would
     be, not just total_return * starting_capital. Final metrics (Sharpe,
     drawdown, hit rate, ...) are still computed via core.metrics so the
     definitions match every other benchmark in this repo exactly.

Session hours (SessionConfig.session_start/end, trading_days) and
max_duration_seconds are "the configurator that determines runtime" - they
say when the simulator is actually working (polling/trading) versus asleep,
and how long a run is allowed to go before it stops and finalizes.

Warm-up fit isolation: both `run_live()` and `run_replay_demo()` fit every
algorithm in its own subprocess (`specs={name: (module_name, class_name)}`,
see `_warmup`/`_fit_isolated`) - the same issue scripts/run_all_benchmarks.py
already isolates this way: mixing lightgbm/xgboost/catboost/torch/hmmlearn/
hdbscan's native runtimes in one interpreter reliably segfaults (or aborts
on a duplicate-OpenMP-init error) once enough distinct libraries have
trained a model, typically a dozen or so algorithms in - fitting ALL of them
in one process (as demo mode used to, being small/fast enough to seem safe)
hits this just as reliably as live mode's full 30+-algorithm run. Each
worker receives the already-loaded (and, for demo, already-truncated-to-
hold-back-the-replay-tail) history via a temp pickle instead of re-fetching
it itself, and hands the fitted, pickled algorithm back to this process via
another temp file, so the tick loop can hold and repeatedly call all of them
without re-isolating (`generate_signals` alone does not exhibit this
crash - only training does).
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, fields
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning

from .base import AlgorithmCategory, BaseTradingAlgorithm, InputMode
from .benchmark_runner import BenchmarkRunner
from .data_loader import TInvestDataLoader
from .metrics import BacktestMetrics, compute_hit_rate, compute_metrics
from .tinvest_client import TInvestClient

# A live/demo session fits 30+ algorithms back to back; per-fit convergence
# notices (sklearn's GaussianProcessRegressor, hmmlearn's GaussianHMM, ...)
# are benign - the fit just stopped at its iteration/tolerance limit - but
# drown out the actually useful [simulator] progress log and this session's
# results/<basename>_progress.json log tail shown by the web dashboard.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

MSK = timezone(timedelta(hours=3))
_NEWS_LOG_DIR = Path("data/news_signals")


def _rss_mb(pid: int | None = None) -> float:
    """Resident set size of the given (or current) process, in MB."""
    try:
        return psutil.Process(pid or os.getpid()).memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return float("nan")


def _json_safe(obj):
    """Recursively replaces float NaN/+-Infinity with None. Some metrics are
    legitimately undefined (e.g. compute_hit_rate on a strategy that never
    opened a position - see core.metrics) and come through as NaN; Python's
    json module happily emits those as bare `NaN`/`Infinity` tokens, which
    is NOT valid JSON (ECMA-404) - a browser's strict `fetch(...).json()`
    (used by scripts/web/dashboard.html to poll /api/status) throws on them,
    which silently breaks every single poll and looks like "the dashboard
    shows nothing at all", not like a metrics-formatting issue."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Configuration - the literal "configurator" (see configs/market_simulator.json)
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    tickers: tuple[str, ...] = (
        "SBER", "GAZP", "LKOH", "GMKN", "VTBR", "ROSN", "NVTK", "MTSS", "TATN", "MOEX",
    )
    interval: str = "5min"
    starting_capital_rub: float = 1_000_000.0
    transaction_cost_bps: float = 5.0
    # History pulled to fit() each algorithm before going live. Deliberately
    # much smaller than the daily-bar backtests' multi-year windows: at
    # intraday intervals, T-Invest's GetCandles is chunked at 1 day/request
    # (core.data_loader._INTERVAL_MAP), so warmup_days is also "how many
    # sequential API calls per ticker warm-up costs" - 20d x 10 tickers is
    # already ~2000-3500 rows of 5-min bars per ticker, plenty to fit these
    # models (comparable to the ~300-row minimum used elsewhere in this repo
    # for daily data), without turning warm-up into a multi-hour fetch.
    warmup_days: float = 20.0
    session_start: str = "10:00"        # MSK, MOEX main session
    session_end: str = "18:39"          # MSK, MOEX main session close
    trading_days: tuple[int, ...] = (0, 1, 2, 3, 4)   # Mon-Fri
    poll_interval_seconds: float = 60.0
    max_duration_seconds: float | None = None   # None = run until session_end, else hard cap
    autosave_every_ticks: int = 1
    run_news_monitor: bool = False      # spin up scripts/run_news_monitor.py if a news-sentiment algo is present
    demo_speed_ms: float | None = None  # --demo only: None = sample from measured latency, 0 = no delay
    results_dir: str = "results"
    basename: str = "market_simulation"

    @classmethod
    def from_file(cls, path: str | Path) -> "SessionConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "tickers" in kwargs:
            kwargs["tickers"] = tuple(kwargs["tickers"])
        if "trading_days" in kwargs:
            kwargs["trading_days"] = tuple(kwargs["trading_days"])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Latency: real T-Invest round-trip time, not a guessed constant
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Reads TInvestClient.recent_latencies_ms() - the ground truth for how
    long real T-Invest API calls actually take this session."""

    _FALLBACK_MS = 180.0  # documented fallback only used before any real call has been recorded

    def __init__(self, client: TInvestClient):
        self._client = client

    def stats(self) -> dict:
        samples = self._client.recent_latencies_ms()
        if not samples:
            return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
        arr = np.array(samples, dtype=float)
        return {
            "n": len(arr),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }

    def sample_ms(self) -> float:
        """Bootstrap draw from real recorded latencies (demo mode only) -
        falls back to a documented constant if nothing's been measured yet."""
        samples = self._client.recent_latencies_ms()
        if not samples:
            return self._FALLBACK_MS
        return float(np.random.choice(samples))


# ---------------------------------------------------------------------------
# Simulated broker: a real cash/shares ledger, not a returns formula
# ---------------------------------------------------------------------------

class SimulatedBroker:
    """
    Per-algorithm paper-trading account. Each tick, every actively traded
    ticker gets an equal slice of current equity (mirrors the equal-weight
    averaging Backtester._run_multi already uses for multi-asset algorithms,
    just implemented as an actual ledger instead of an averaged-returns
    formula) and is rebalanced toward `target_position * slice_equity /
    price`, paying `transaction_cost_bps` on the traded notional.

    Single-asset callers simply pass single-key dicts - the same step()
    reduces exactly to "rebalance the whole account toward one position".
    """

    def __init__(self, starting_capital: float, transaction_cost_bps: float):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.shares: dict[str, float] = {}
        self.transaction_cost_bps = transaction_cost_bps
        self.trade_log: list[dict] = []
        self.timestamps: list = []
        self.equity_curve: list[float] = []
        self.positions_curve: list[dict[str, float]] = []
        self.price_curve: list[dict[str, float]] = []
        self._last_prices: dict[str, float] = {}

    def step(self, timestamp, prices: dict[str, float], target_positions: dict[str, float]) -> None:
        for ticker, price in prices.items():
            if price is not None and pd.notna(price) and price > 0:
                self._last_prices[ticker] = float(price)

        active = list(target_positions.keys())
        n = max(len(active), 1)
        equity_before = self.cash + sum(
            self.shares.get(t, 0.0) * self._last_prices.get(t, 0.0) for t in self.shares
        )
        slice_equity = equity_before / n

        for ticker in active:
            price = self._last_prices.get(ticker)
            if not price:
                continue
            target_position = max(-1.0, min(1.0, float(target_positions[ticker])))
            target_shares = (target_position * slice_equity) / price
            current_shares = self.shares.get(ticker, 0.0)
            delta_shares = target_shares - current_shares
            if abs(delta_shares) * price > 1e-6:
                trade_value = abs(delta_shares) * price
                cost = trade_value * (self.transaction_cost_bps / 10_000.0)
                self.cash -= delta_shares * price + cost
                self.shares[ticker] = target_shares
                self.trade_log.append({
                    "time": str(timestamp), "ticker": ticker, "price": price,
                    "delta_shares": delta_shares, "cost": cost, "target_position": target_position,
                })

        equity_after = self.cash + sum(
            self.shares.get(t, 0.0) * self._last_prices.get(t, 0.0) for t in self.shares
        )
        self.timestamps.append(timestamp)
        self.equity_curve.append(equity_after)
        self.positions_curve.append({t: max(-1.0, min(1.0, float(v))) for t, v in target_positions.items()})
        self.price_curve.append(dict(self._last_prices))

    @property
    def equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.starting_capital

    def metrics(self, starting_capital: float | None = None) -> BacktestMetrics:
        sc = starting_capital if starting_capital is not None else self.starting_capital
        if len(self.equity_curve) < 2:
            return compute_metrics(
                pd.Series(dtype=float), pd.Series(dtype=float), hit_rate=float("nan"), starting_capital=sc,
            )
        idx = pd.DatetimeIndex(self.timestamps)
        equity = pd.Series(self.equity_curve, index=idx)
        strategy_returns = equity.pct_change().fillna(0.0)

        # Aggregate exposure across tickers (abs-mean, same convention as
        # Backtester._run_multi - a mean would give a false zero for hedged
        # strategies like pairs trading whose legs have opposite signs).
        signals = pd.Series(
            [np.mean([abs(v) for v in p.values()]) if p else 0.0 for p in self.positions_curve], index=idx,
        )

        all_tickers = sorted({t for p in self.positions_curve for t in p})
        hr_positions, hr_returns = [], []
        for ticker in all_tickers:
            pos = pd.Series([p.get(ticker, 0.0) for p in self.positions_curve], index=idx).shift(1).fillna(0.0)
            price = pd.Series([pr.get(ticker, np.nan) for pr in self.price_curve], index=idx)
            ret = price.pct_change().fillna(0.0)
            hr_positions.append(pos.reset_index(drop=True))
            hr_returns.append(ret.reset_index(drop=True))
        hit_rate = (
            compute_hit_rate(pd.concat(hr_positions, ignore_index=True), pd.concat(hr_returns, ignore_index=True))
            if hr_positions else float("nan")
        )
        return compute_metrics(strategy_returns, signals, hit_rate=hit_rate, starting_capital=sc)


# ---------------------------------------------------------------------------
# Live news sentiment feed - reuses scripts/run_news_monitor.py's log format,
# no RSS/LLM logic duplicated here.
# ---------------------------------------------------------------------------

def _latest_sentiment(
    ticker: str, as_of: datetime, log_dir: Path = _NEWS_LOG_DIR, half_life_hours: float = 6.0
) -> float | None:
    # Looks at today's and yesterday's log file so a story scored just before
    # UTC midnight keeps decaying smoothly instead of vanishing the instant
    # the calendar day rolls over. Records older than ~10 half-lives contribute
    # a negligible weight (<0.1%), so the two-file window never needs to grow
    # further back regardless of half_life_hours.
    #
    # Direction and staleness are tracked as two separate confidence-weighted
    # averages rather than folded into one weight: normalizing a decayed
    # weight by its own sum cancels the decay out entirely for a single
    # record (a weighted average of one value is just that value), so instead
    # "what do today's items say" (position, confidence-weighted across
    # items) and "how stale is that read as of now" (the decay factor,
    # itself confidence-weighted across the same items) are computed
    # independently, then multiplied.
    position_weighted, decay_weighted, confidence_total, any_record = 0.0, 0.0, 0.0, False
    for day in (as_of.date(), (as_of - timedelta(days=1)).date()):
        path = log_dir / f"{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ticker") != ticker:
                continue
            try:
                processed_at = datetime.fromisoformat(rec["processed_at"])
            except (KeyError, ValueError):
                continue
            if processed_at > as_of:
                continue
            any_record = True
            elapsed_hours = (as_of - processed_at).total_seconds() / 3600.0
            confidence = float(rec.get("confidence", 1.0))
            decay = math.exp(-math.log(2) * elapsed_hours / half_life_hours)
            position_weighted += confidence * float(rec.get("position", 0.0))
            decay_weighted += confidence * decay
            confidence_total += confidence
    if not any_record:
        return None
    if confidence_total <= 0.0:
        return 0.0
    return (position_weighted / confidence_total) * (decay_weighted / confidence_total)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------

class MarketSimulator:
    def __init__(
        self,
        config: SessionConfig,
        loader: TInvestDataLoader | None = None,
        client: TInvestClient | None = None,
    ):
        self.config = config
        self.client = client or TInvestClient()
        self.loader = loader or TInvestDataLoader(client=self.client)
        self.latency = LatencyTracker(self.client)
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._news_proc: subprocess.Popen | None = None

        # Real per-algorithm compute-load tracking (CPU time / RSS), keyed by
        # algorithm name - "current load and consumption of each algorithm"
        # (see _process_tick / _fit_isolated / _build_payload). Not a
        # simulated/estimated number: fit_seconds and tick_cpu_seconds_total
        # are measured with time.process_time()/perf_counter() around the
        # actual fit()/generate_signals() calls, and *_rss_mb comes from
        # psutil reading real process memory.
        self._resource_stats: dict[str, dict] = {}
        self._self_proc = psutil.Process(os.getpid())
        self._self_proc.cpu_percent(interval=None)  # prime the counter (first call is always 0.0)

    def _primary_ticker(self) -> str:
        return self.config.tickers[0]

    # -- session clock ------------------------------------------------------

    def _is_trading_time(self, now_msk: datetime) -> bool:
        if now_msk.weekday() not in self.config.trading_days:
            return False
        start_t = dt_time.fromisoformat(self.config.session_start)
        end_t = dt_time.fromisoformat(self.config.session_end)
        return start_t <= now_msk.time() <= end_t

    def _next_session_open(self, now_msk: datetime) -> datetime:
        start_t = dt_time.fromisoformat(self.config.session_start)
        candidate = now_msk.replace(hour=start_t.hour, minute=start_t.minute, second=0, microsecond=0)
        if candidate <= now_msk:
            candidate += timedelta(days=1)
        while candidate.weekday() not in self.config.trading_days:
            candidate += timedelta(days=1)
        return candidate

    # -- warm-up --------------------------------------------------------------

    def _warmup(
        self,
        specs: dict[str, tuple[str, str]],
        extra_days: float = 0.0,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, BaseTradingAlgorithm], dict[str, SimulatedBroker], dict[str, str]]:
        """
        Loads `warmup_days` (+ `extra_days`, for demo mode's replay tail - see
        run_replay_demo) of history, then fits every algorithm in `specs`
        (name -> (module_name, class_name)), each isolated in its own
        subprocess (see _fit_isolated / the module docstring) to avoid
        native-library segfaults from mixing many ML libraries in one
        process. The already-loaded (and, for demo, already-truncated-to-
        hold-back-the-replay-tail) history is hand off to each worker via a
        temp pickle rather than having it re-fetch from T-Invest itself.
        """
        print(
            f"[simulator] warm-up: loading {self.config.warmup_days:.0f}d of "
            f"{self.config.interval} history for {len(self.config.tickers)} tickers ..."
        )
        history: dict[str, pd.DataFrame] = {}
        for ticker in self.config.tickers:
            df = self.loader.load_recent(
                ticker, self.config.interval, lookback_days=self.config.warmup_days + extra_days, use_cache=True,
            )
            df = df.copy()
            df["sentiment_score"] = np.nan  # no historical RSS archive (see README) - live-only going forward
            history[ticker] = df

        primary = self._primary_ticker()
        instances: dict[str, BaseTradingAlgorithm] = {}
        brokers: dict[str, SimulatedBroker] = {}
        skipped: dict[str, str] = {}
        warmup_cutoff = {t: len(df) - self._n_periods(extra_days, df) for t, df in history.items()} if extra_days else None

        print(f"[simulator] fitting {len(specs)} algorithms, each isolated in its own subprocess "
              f"(avoids native-library segfaults when mixing many ML libraries in one process) ...")
        for i, (name, spec) in enumerate(specs.items(), 1):
            print(f"[simulator]  ({i}/{len(specs)}) fitting {name} ...")
            fit_history = history if warmup_cutoff is None else {
                t: df.iloc[: warmup_cutoff[t]] for t, df in history.items()
            }
            try:
                instances[name] = self._fit_isolated(name, spec, fit_history, primary)
                brokers[name] = SimulatedBroker(self.config.starting_capital_rub, self.config.transaction_cost_bps)
            except Exception as exc:
                skipped[name] = f"{type(exc).__name__}: {exc}"
                print(f"[simulator] {name}: warm-up fit failed - {skipped[name]}")

        print(f"[simulator] {len(instances)} algorithms fitted, {len(skipped)} skipped (fit failed)")
        return history, instances, brokers, skipped

    def _fit_isolated(
        self, name: str, spec: tuple[str, str], fit_history: dict[str, pd.DataFrame], primary: str,
    ) -> BaseTradingAlgorithm:
        """Fits ONE algorithm in a fresh subprocess (python -m core.market_simulator
        --fit-worker ...), handing it `fit_history` via a temp pickle, and returns
        the fitted, unpickled instance. Also records that worker's own measured fit
        time and peak RSS into self._resource_stats (the worker measures itself -
        see _fit_worker_main - since a subprocess's memory isn't visible from the
        parent after it exits)."""
        modname, clsname = spec
        fd_data, data_path = tempfile.mkstemp(prefix="simfit_data_", suffix=".pkl")
        os.close(fd_data)
        fd_out, out_path = tempfile.mkstemp(prefix="simfit_", suffix=".pkl")
        os.close(fd_out)
        try:
            with open(data_path, "wb") as f:
                pickle.dump(fit_history, f)
            src_dir = str(Path(__file__).resolve().parents[1])
            env = os.environ.copy()
            env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            cmd = [
                sys.executable, "-m", "core.market_simulator", "--fit-worker",
                modname, clsname, primary, data_path, out_path,
            ]
            wall_t0 = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, timeout=900, env=env, cwd=src_dir)
            wall_seconds = time.perf_counter() - wall_t0
            if result.returncode != 0 or not Path(out_path).exists():
                stderr_tail = (result.stderr or b"").decode(errors="replace").strip().splitlines()
                reason = stderr_tail[-1] if stderr_tail else f"worker exited with code {result.returncode}"
                raise RuntimeError(reason)
            with open(out_path, "rb") as f:
                payload = pickle.load(f)
            self._resource_stats[name] = {
                "fit_seconds": payload.get("fit_seconds", wall_seconds),
                "fit_wall_seconds": wall_seconds,
                "fit_peak_rss_mb": payload.get("fit_peak_rss_mb"),
                "fit_isolated": True,
            }
            return payload["algo"]
        finally:
            Path(out_path).unlink(missing_ok=True)
            Path(data_path).unlink(missing_ok=True)

    @staticmethod
    def _n_periods(lookback_days: float, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
        return len(df) - int(df.index.searchsorted(cutoff))

    # -- news monitor subprocess ---------------------------------------------

    def _maybe_start_news_monitor(self, instances: dict[str, BaseTradingAlgorithm]) -> None:
        if not self.config.run_news_monitor:
            return
        has_news_algo = any(a.category == AlgorithmCategory.NEWS_SENTIMENT for a in instances.values())
        if not has_news_algo:
            return
        script = Path(__file__).resolve().parents[2] / "scripts" / "run_news_monitor.py"
        if not script.exists():
            print(f"[simulator] run_news_monitor=true but {script} not found, skipping live news feed")
            return
        print(f"[simulator] starting live news monitor subprocess ({script.name}) ...")
        try:
            self._news_proc = subprocess.Popen([sys.executable, str(script), "--interval", "120"])
        except OSError as exc:
            print(f"[simulator] could not start news monitor subprocess: {exc}")

    def _stop_news_monitor(self) -> None:
        if self._news_proc and self._news_proc.poll() is None:
            self._news_proc.terminate()

    # -- one simulated tick ---------------------------------------------------

    def _process_tick(
        self,
        now: datetime,
        history: dict[str, pd.DataFrame],
        instances: dict[str, BaseTradingAlgorithm],
        brokers: dict[str, SimulatedBroker],
    ) -> None:
        primary = self._primary_ticker()
        for name, algo in instances.items():
            cpu_t0 = time.process_time()
            wall_t0 = time.perf_counter()
            try:
                if algo.input_mode == InputMode.SINGLE_ASSET:
                    signals = algo.generate_signals(history[primary])
                    position = float(signals.iloc[-1]) if len(signals) else 0.0
                    price = float(history[primary]["close"].iloc[-1])
                    brokers[name].step(now, {primary: price}, {primary: position})
                else:
                    signals_by_ticker = algo.generate_signals(history)
                    positions = {t: float(s.iloc[-1]) if len(s) else 0.0 for t, s in signals_by_ticker.items()}
                    prices = {t: float(history[t]["close"].iloc[-1]) for t in positions if t in history}
                    brokers[name].step(now, prices, positions)
            except Exception as exc:
                print(f"[simulator] {name}: tick failed - {type(exc).__name__}: {exc}")
            finally:
                self._record_tick_load(name, cpu_seconds=time.process_time() - cpu_t0, wall_seconds=time.perf_counter() - wall_t0)

    def _record_tick_load(self, name: str, cpu_seconds: float, wall_seconds: float) -> None:
        """Accumulates real measured per-tick CPU/wall time for `name` -
        time.process_time() is this process's actual CPU time consumed
        during the call, not an estimate. Used to report each algorithm's
        share of total compute load (see _build_payload)."""
        stats = self._resource_stats.setdefault(name, {})
        stats["tick_cpu_seconds_total"] = stats.get("tick_cpu_seconds_total", 0.0) + cpu_seconds
        stats["tick_wall_seconds_total"] = stats.get("tick_wall_seconds_total", 0.0) + wall_seconds
        stats["n_ticks_measured"] = stats.get("n_ticks_measured", 0) + 1
        stats["last_tick_ms"] = wall_seconds * 1000.0
        stats["avg_tick_ms"] = stats["tick_wall_seconds_total"] / stats["n_ticks_measured"] * 1000.0
        stats["process_rss_mb"] = _rss_mb()

    # -- live mode --------------------------------------------------------------

    def run_live(self, specs: dict[str, tuple[str, str]]) -> dict:
        """`specs` (module/class name pairs) - each algorithm is fitted in
        its own isolated subprocess, see module docstring."""
        history, instances, brokers, skipped = self._warmup(specs)
        self._maybe_start_news_monitor(instances)
        last_seen = {t: (df.index[-1] if len(df) else pd.Timestamp.min.tz_localize("UTC")) for t, df in history.items()}

        started_at = datetime.now(timezone.utc)
        n_ticks = 0
        try:
            while True:
                now_utc = datetime.now(timezone.utc)
                if (
                    self.config.max_duration_seconds is not None
                    and (now_utc - started_at).total_seconds() >= self.config.max_duration_seconds
                ):
                    print("[simulator] max_duration_seconds reached, stopping.")
                    break

                now_msk = now_utc.astimezone(MSK)
                if not self._is_trading_time(now_msk):
                    next_open = self._next_session_open(now_msk)
                    sleep_s = max(1.0, min((next_open - now_msk).total_seconds(), self.config.poll_interval_seconds * 10))
                    print(f"[simulator] outside session hours (MSK {now_msk:%a %H:%M}), sleeping {sleep_s:.0f}s ...")
                    time.sleep(sleep_s)
                    continue

                if self._poll_new_bars(history, last_seen):
                    self._process_tick(now_utc, history, instances, brokers)
                    n_ticks += 1
                    if n_ticks % self.config.autosave_every_ticks == 0:
                        self._autosave(instances, brokers, skipped, started_at, n_ticks, mode="live")
                time.sleep(self.config.poll_interval_seconds)
        except KeyboardInterrupt:
            print("\n[simulator] interrupted by user, finalizing ...")
        finally:
            self._stop_news_monitor()

        return self._finalize(instances, brokers, skipped, started_at, n_ticks, mode="live")

    def _poll_new_bars(self, history: dict[str, pd.DataFrame], last_seen: dict[str, pd.Timestamp]) -> bool:
        any_new = False
        for ticker in self.config.tickers:
            try:
                fresh = self.loader.load_recent(ticker, self.config.interval, lookback_days=2, use_cache=False)
            except Exception as exc:
                print(f"[simulator] {ticker}: poll failed - {type(exc).__name__}: {exc}")
                continue
            if fresh.empty:
                continue
            new_rows = fresh[fresh.index > last_seen[ticker]]
            if new_rows.empty:
                continue
            as_of = new_rows.index[-1].to_pydatetime()
            new_rows = new_rows.copy()
            sentiment = _latest_sentiment(ticker, as_of)
            new_rows["sentiment_score"] = np.nan if sentiment is None else sentiment
            history[ticker] = pd.concat([history[ticker], new_rows])
            last_seen[ticker] = new_rows.index[-1]
            any_new = True
        return any_new

    # -- accelerated demo mode ---------------------------------------------------

    def run_replay_demo(self, specs: dict[str, tuple[str, str]], lookback_days: float = 5.0) -> dict:
        print(f"[simulator] DEMO mode: replaying the last {lookback_days:.0f}d of {self.config.interval} bars "
              f"with synthetic per-tick delay calibrated from real T-Invest latency.")
        history, instances, brokers, skipped = self._warmup(specs, extra_days=lookback_days)

        primary = self._primary_ticker()
        replay_index = history[primary].iloc[-self._n_periods(lookback_days, history[primary]):].index
        # Split off the replay tail so it can be fed in one row at a time,
        # exactly like new bars would arrive live.
        live_history = {t: df.iloc[: len(df) - self._n_periods(lookback_days, df)].copy() for t, df in history.items()}
        full_history = history

        started_at = datetime.now(timezone.utc)
        n_ticks = 0
        for ts in replay_index:
            for ticker in self.config.tickers:
                df = full_history[ticker]
                if ts in df.index:
                    live_history[ticker] = pd.concat([live_history[ticker], df.loc[[ts]]])
            self._process_tick(pd.Timestamp(ts).to_pydatetime(), live_history, instances, brokers)
            n_ticks += 1

            delay_ms = self.config.demo_speed_ms if self.config.demo_speed_ms is not None else self.latency.sample_ms()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            if n_ticks % self.config.autosave_every_ticks == 0:
                self._autosave(instances, brokers, skipped, started_at, n_ticks, mode="demo")

        return self._finalize(instances, brokers, skipped, started_at, n_ticks, mode="demo")

    # -- reporting ---------------------------------------------------------------

    def _build_payload(
        self,
        instances: dict[str, BaseTradingAlgorithm],
        brokers: dict[str, SimulatedBroker],
        skipped: dict[str, str],
        started_at: datetime,
        n_ticks: int,
        mode: str,
    ) -> dict:
        total_tick_cpu = sum(s.get("tick_cpu_seconds_total", 0.0) for s in self._resource_stats.values()) or None

        rows = []
        for name, algo in instances.items():
            broker = brokers[name]
            m = broker.metrics(self.config.starting_capital_rub)
            row = {
                "spec_name": name,
                "algorithm_name": getattr(algo, "name", name),
                "category": str(algo.category.value if hasattr(algo.category, "value") else algo.category),
                "tickers": self._primary_ticker() if algo.input_mode == InputMode.SINGLE_ASSET else ",".join(self.config.tickers),
                "n_trades_executed": len(broker.trade_log),
                "n_ticks": len(broker.equity_curve),
                "error": None,
            }
            row.update(m.to_dict())
            res = dict(self._resource_stats.get(name, {}))
            res["cpu_load_share_pct"] = (
                100.0 * res.get("tick_cpu_seconds_total", 0.0) / total_tick_cpu if total_tick_cpu else None
            )
            row["resource"] = res
            rows.append(row)

        empty_metrics = compute_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float), hit_rate=float("nan"),
            starting_capital=self.config.starting_capital_rub,
        ).to_dict()
        for name, err in skipped.items():
            rows.append({
                "spec_name": name, "algorithm_name": name, "category": "", "tickers": "",
                "n_trades_executed": 0, "n_ticks": 0, "error": err,
                "resource": dict(self._resource_stats.get(name, {})), **empty_metrics,
            })

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "session_started_at": started_at.isoformat(),
            "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            "n_ticks": n_ticks,
            "config": self.config.to_dict(),
            "latency_stats_ms": self.latency.stats(),
            "process_stats": {
                "pid": os.getpid(),
                "cpu_percent": self._self_proc.cpu_percent(interval=None),
                "rss_mb": _rss_mb(),
                "num_threads": self._self_proc.num_threads(),
            },
            "n_algorithms_run": len(instances),
            "n_algorithms_skipped": len(skipped),
            "results": rows,
        }

    def _autosave(self, instances, brokers, skipped, started_at, n_ticks, mode: str) -> None:
        payload = self._build_payload(instances, brokers, skipped, started_at, n_ticks, mode)
        path = self.results_dir / f"{self.config.basename}_progress.json"
        path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _finalize(self, instances, brokers, skipped, started_at, n_ticks, mode: str) -> dict:
        payload = self._build_payload(instances, brokers, skipped, started_at, n_ticks, mode)
        json_path = self.results_dir / f"{self.config.basename}.json"
        json_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        md_path = self.results_dir / f"{self.config.basename}.md"
        md_path.write_text(self._to_markdown(payload), encoding="utf-8")
        print(f"[simulator] saved {json_path} and {md_path}")
        return payload

    @staticmethod
    def _to_markdown(payload: dict) -> str:
        results_df = pd.DataFrame(payload["results"])
        lat = payload["latency_stats_ms"]
        lat_line = (
            f"- T-Invest API calls measured: {lat['n']}, mean {lat['mean_ms']:.0f} ms, "
            f"p50 {lat['p50_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms, max {lat['max_ms']:.0f} ms"
            if lat["n"] else "- no T-Invest API calls recorded this session yet"
        )
        ps = payload.get("process_stats", {})
        header = [
            "# Live market simulation results", "",
            f"Mode: **{payload['mode']}**  |  Generated: {payload['generated_at']}",
            f"Session started: {payload['session_started_at']}  |  "
            f"Elapsed: {payload['elapsed_seconds']:.0f}s  |  Ticks: {payload['n_ticks']}",
            f"Algorithms run: {payload['n_algorithms_run']}  |  Skipped (warm-up fit failed): {payload['n_algorithms_skipped']}",
            "",
            "## Process load",
            "",
            f"- simulator PID {ps.get('pid')}: CPU {ps.get('cpu_percent', 0):.0f}%, "
            f"RSS {ps.get('rss_mb', 0):.0f} MB, {ps.get('num_threads', '?')} threads",
            "",
            "## T-Invest API latency observed this session",
            "",
            lat_line,
            "",
            "In live mode this is real measured latency (no delay is simulated - the wall-clock "
            "pacing of the session IS the real network round trip). In demo mode, per-tick sleeps "
            "are sampled from these same measured values instead of a guessed constant.",
            "",
        ]
        body = BenchmarkRunner._body_markdown(results_df) if not results_df.empty else "(no results)\n"
        body += MarketSimulator._resource_markdown(payload["results"])
        return "\n".join(header) + body

    @staticmethod
    def _resource_markdown(rows: list[dict]) -> str:
        """Per-algorithm compute load table: real fit time/RSS and per-tick
        CPU time measured this session (see _resource_stats / _record_tick_load)."""
        recs = []
        for r in rows:
            res = r.get("resource") or {}
            if not res:
                continue
            recs.append({
                "algorithm_name": r.get("algorithm_name"),
                "fit_seconds": res.get("fit_seconds"),
                "fit_peak_rss_mb": res.get("fit_peak_rss_mb"),
                "fit_rss_delta_mb": res.get("fit_rss_delta_mb"),
                "avg_tick_ms": res.get("avg_tick_ms"),
                "tick_cpu_seconds_total": res.get("tick_cpu_seconds_total"),
                "cpu_load_share_pct": res.get("cpu_load_share_pct"),
            })
        if not recs:
            return ""
        df = pd.DataFrame(recs).sort_values("cpu_load_share_pct", ascending=False, na_position="last")
        for col in ("fit_seconds", "avg_tick_ms", "tick_cpu_seconds_total"):
            df[col] = df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        for col in ("fit_peak_rss_mb", "fit_rss_delta_mb"):
            df[col] = df[col].map(lambda x: f"{x:+.1f}" if pd.notnull(x) else "")
        df["cpu_load_share_pct"] = df["cpu_load_share_pct"].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
        return "\n## Compute load per algorithm\n\n" + df.to_markdown(index=False) + "\n"


# ---------------------------------------------------------------------------
# --fit-worker: fits ONE algorithm in isolation, invoked by
# MarketSimulator._fit_isolated via `python -m core.market_simulator
# --fit-worker <modname> <clsname> <primary_ticker> <data_pickle_path>
# <out_pickle_path>`. The parent process (MarketSimulator._warmup) has
# already loaded (and, for demo mode, truncated to hold back the replay
# tail) the history and handed it over via data_pickle_path, so this worker
# never touches T-Invest itself - it just needs a fresh interpreter so this
# algorithm's native library gets its own process. Kept in this file
# (rather than a separate script) so the fit logic - how to call fit() for
# single- vs multi-asset algorithms - lives in one place.
# ---------------------------------------------------------------------------

def _fit_worker_main(argv: list[str]) -> None:
    import importlib

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    modname, clsname, primary_ticker, data_path, out_path = argv

    module = importlib.import_module(f"algorithms.{modname}")
    cls = getattr(module, clsname)
    algo = cls()

    with open(data_path, "rb") as f:
        history: dict[str, pd.DataFrame] = pickle.load(f)

    t0 = time.perf_counter()
    if algo.input_mode == InputMode.SINGLE_ASSET:
        algo.ticker = primary_ticker  # type: ignore[attr-defined]
        algo.fit(history[primary_ticker])
    else:
        algo.fit(history)
    fit_seconds = time.perf_counter() - t0
    fit_peak_rss_mb = _rss_mb()  # this worker's own RSS - the fitted model is its dominant consumer

    with open(out_path, "wb") as f:
        pickle.dump({"algo": algo, "fit_seconds": fit_seconds, "fit_peak_rss_mb": fit_peak_rss_mb}, f)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--fit-worker":
        _fit_worker_main(sys.argv[2:])
    else:
        # Tiny smoke test on synthetic data, no network access needed -
        # mirrors the if __name__ == "__main__" convention used by every
        # algorithm file.
        rng = np.random.default_rng(0)
        n = 200
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        broker = SimulatedBroker(starting_capital=1_000_000.0, transaction_cost_bps=5.0)
        for i in range(n):
            # constant long position - equity should roughly track the price path net of costs
            broker.step(idx[i], {"TEST": float(price[i])}, {"TEST": 1.0})
        m = broker.metrics()
        print(f"constant-long smoke test: total_return={m.total_return:+.2%}, "
              f"pnl_rub={m.pnl_rub:+,.0f}, n_trades={len(broker.trade_log)}")
