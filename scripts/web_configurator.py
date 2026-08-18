"""
Web configurator for scripts/run_market_simulation.py: a local dashboard to
edit configs/market_simulator.json, pick which algorithms to run, start/stop
a simulation session, and watch its live progress - including each
algorithm's real measured compute load (CPU time / RSS), not a simulated
number, as reported by core.market_simulator's per-algorithm resource
tracking (MarketSimulator._resource_stats).

The simulation itself always runs as a child process of this server (the
same `python scripts/run_market_simulation.py ...` you'd run by hand) so a
crash or Ctrl-C-equivalent stop here can't take the dashboard down with it,
and so results/<basename>_progress.json (autosaved every tick) is the same
file both a CLI run and this dashboard read.

Usage:
    python scripts/web_configurator.py                  # http://127.0.0.1:8765
    python scripts/web_configurator.py --port 9000
    python scripts/web_configurator.py --host 0.0.0.0    # expose beyond localhost (opt-in)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import time as dt_time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from core.algorithm_discovery import discover_algorithms  # noqa: E402
from core.market_simulator import SessionConfig, _json_safe  # noqa: E402
from core.news_feed import feed_status, set_feed_enabled  # noqa: E402

RUN_SCRIPT = REPO_ROOT / "scripts" / "run_market_simulation.py"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "market_simulator.json"
RESULTS_DIR = REPO_ROOT / "results"
LOG_PATH = RESULTS_DIR / "web_configurator_run.log"
DASHBOARD_HTML = Path(__file__).resolve().parent / "web" / "dashboard.html"

KNOWN_CONFIG_FIELDS = {f.name for f in dataclasses.fields(SessionConfig)}

_lock = threading.Lock()
_state: dict = {
    "proc": None,
    "watch_proc": None,
    "started_at": None,
    "args": None,
    "basename": None,
    "demo": None,
    "log_file": None,
}
_algorithm_catalog_cache: dict | None = None


# ---------------------------------------------------------------------------
# config read/write
# ---------------------------------------------------------------------------

def _read_config() -> dict:
    if DEFAULT_CONFIG_PATH.exists():
        return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    return SessionConfig().to_dict()


_VALID_INTERVALS = {"1min", "5min", "15min", "hour", "day"}


def _validate_config(kwargs: dict) -> None:
    if kwargs.get("interval") not in _VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(_VALID_INTERVALS)}, got {kwargs.get('interval')!r}")
    for field in ("session_start", "session_end"):
        try:
            dt_time.fromisoformat(kwargs[field])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{field} must be HH:MM, got {kwargs.get(field)!r}") from exc
    for d in kwargs.get("trading_days", ()):
        if not isinstance(d, int) or not 0 <= d <= 6:
            raise ValueError(f"trading_days must be integers 0 (Mon) to 6 (Sun), got {d!r}")
    if not kwargs.get("tickers"):
        raise ValueError("tickers must be a non-empty list")


def _write_config(body: dict) -> dict:
    current = _read_config()
    for key, value in body.items():
        if key in KNOWN_CONFIG_FIELDS:
            current[key] = value
    # Round-trip through the dataclass so bad values (wrong type, unknown
    # trading_days encoding, ...) fail loudly here instead of silently
    # producing a config run_market_simulation.py can't parse.
    kwargs = {k: v for k, v in current.items() if k in KNOWN_CONFIG_FIELDS}
    if "tickers" in kwargs:
        kwargs["tickers"] = tuple(kwargs["tickers"])
    if "trading_days" in kwargs:
        kwargs["trading_days"] = tuple(kwargs["trading_days"])
    _validate_config(kwargs)
    cfg = SessionConfig(**kwargs)
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg.to_dict()


# ---------------------------------------------------------------------------
# algorithm catalog (name/category, via class attributes - no instantiation)
# ---------------------------------------------------------------------------

def _algorithm_catalog() -> dict:
    global _algorithm_catalog_cache
    if _algorithm_catalog_cache is not None:
        return _algorithm_catalog_cache
    found, skipped = discover_algorithms()
    catalog = []
    for modname, clsname in found:
        module = importlib.import_module(f"algorithms.{modname}")
        cls = getattr(module, clsname)
        category = getattr(cls, "category", "")
        catalog.append({
            "module": modname,
            "class": clsname,
            "name": getattr(cls, "name", clsname),
            "category": category.value if hasattr(category, "value") else str(category),
        })
    catalog.sort(key=lambda r: r["module"])
    _algorithm_catalog_cache = {
        "found": catalog,
        "skipped": [{"module": m, "class": c} for m, c in skipped],
    }
    return _algorithm_catalog_cache


# ---------------------------------------------------------------------------
# run lifecycle
# ---------------------------------------------------------------------------

def _start_run(body: dict) -> dict:
    with _lock:
        proc = _state["proc"]
        if proc is not None and proc.poll() is None:
            raise RuntimeError("a simulation is already running - stop it first")

        cfg = _read_config()
        basename = (body.get("basename") or cfg.get("basename") or "market_simulation").strip()
        demo = bool(body.get("demo"))

        argv = [sys.executable, str(RUN_SCRIPT), "--basename", basename]
        if demo:
            argv.append("--demo")
            if body.get("lookback_days") is not None:
                argv += ["--lookback-days", str(body["lookback_days"])]
            if body.get("demo_speed_ms") is not None:
                argv += ["--demo-speed-ms", str(body["demo_speed_ms"])]
        if body.get("duration") is not None:
            argv += ["--duration", str(body["duration"])]
        if body.get("warmup_days") is not None:
            argv += ["--warmup-days", str(body["warmup_days"])]
        algorithms = body.get("algorithms")
        if algorithms:
            argv += ["--algorithms", ",".join(algorithms)]

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Fresh progress file for this run so the dashboard doesn't show a
        # stale snapshot from a previous session under the same basename.
        progress_path = RESULTS_DIR / f"{basename}_progress.json"
        progress_path.unlink(missing_ok=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_file = open(LOG_PATH, "w", encoding="utf-8")
        proc = subprocess.Popen(
            argv, cwd=str(REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        log_file.close()  # child holds its own duped fd; this process doesn't need one open

        watch_proc = psutil.Process(proc.pid)
        watch_proc.cpu_percent(interval=None)  # prime, first call is always 0.0

        _state.update(
            proc=proc, watch_proc=watch_proc, started_at=time.time(),
            args=argv, basename=basename, demo=demo,
        )
        return {"pid": proc.pid, "args": argv, "basename": basename, "demo": demo}


def _stop_run() -> dict:
    with _lock:
        proc = _state["proc"]
        if proc is None or proc.poll() is not None:
            return {"running": False, "message": "no running simulation"}
        proc.send_signal(signal.SIGINT)  # same as Ctrl-C: run_live() catches this and finalizes results
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    return {"running": proc.poll() is None, "returncode": proc.returncode}


def _status() -> dict:
    proc = _state["proc"]
    running = bool(proc is not None and proc.poll() is None)
    info = {
        "running": running,
        "pid": proc.pid if proc else None,
        "started_at": _state["started_at"],
        "elapsed_seconds": (time.time() - _state["started_at"]) if _state["started_at"] else None,
        "args": _state["args"],
        "basename": _state["basename"],
        "demo": _state["demo"],
        "returncode": proc.poll() if proc else None,
    }
    watch_proc: psutil.Process | None = _state["watch_proc"]
    if running and watch_proc is not None:
        try:
            children = watch_proc.children(recursive=True)
            info["system_load"] = {
                "cpu_percent": watch_proc.cpu_percent(interval=None),
                "rss_mb": watch_proc.memory_info().rss / (1024 * 1024),
                "num_threads": watch_proc.num_threads(),
                "n_child_processes": len(children),
                "children_rss_mb": sum(c.memory_info().rss for c in children if c.is_running()) / (1024 * 1024),
            }
        except psutil.NoSuchProcess:
            pass

    basename = _state["basename"] or "market_simulation"
    progress_path = RESULTS_DIR / f"{basename}_progress.json"
    final_path = RESULTS_DIR / f"{basename}.json"
    for path in (progress_path, final_path):
        if path.exists():
            try:
                info["progress"] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
            break

    if LOG_PATH.exists():
        try:
            info["log_tail"] = LOG_PATH.read_text(encoding="utf-8", errors="replace")[-8000:]
        except OSError:
            info["log_tail"] = ""
    return info


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "MarketSimConfigurator/1.0"

    def log_message(self, fmt, *args):  # quieter default access log
        pass

    def _send_json(self, payload, status=200):
        # NaN/Infinity are legitimate values for some metrics (see
        # core.market_simulator._json_safe) but not valid JSON tokens per
        # ECMA-404 - the browser's fetch(...).json() throws on them, which
        # silently breaks every dashboard poll. Progress/result files on
        # disk go through the same sanitizer before being written, but this
        # covers stale files written before that fix, or any other source.
        body = json.dumps(_json_safe(payload), ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/dashboard.html"):
                html = DASHBOARD_HTML.read_text(encoding="utf-8")
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/config":
                self._send_json({"config": _read_config(), "config_path": str(DEFAULT_CONFIG_PATH)})
            elif path == "/api/algorithms":
                self._send_json(_algorithm_catalog())
            elif path == "/api/news_feeds":
                self._send_json({"feeds": feed_status()})
            elif path == "/api/status":
                self._send_json(_status())
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json_body()
            if path == "/api/config":
                self._send_json({"config": _write_config(body)})
            elif path == "/api/run/start":
                self._send_json(_start_run(body))
            elif path == "/api/run/stop":
                self._send_json(_stop_run())
            elif path == "/api/news_feeds":
                source = body.get("source")
                if not source:
                    raise ValueError("body must include 'source'")
                self._send_json({"feeds": set_feed_enabled(source, bool(body.get("enabled")))})
            else:
                self._send_json({"error": "not found"}, status=404)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=409)
        except (ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1, localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[web_configurator] serving on http://{args.host}:{args.port} (config: {DEFAULT_CONFIG_PATH})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proc = _state["proc"]
        if proc is not None and proc.poll() is None:
            print("[web_configurator] shutting down running simulation ...")
            proc.send_signal(signal.SIGINT)
        server.server_close()


if __name__ == "__main__":
    main()
