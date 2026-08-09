"""
Walk-forward validation: closes limitation #1 from the "Study Limitations"
section of the main document (docs/research/*.typ) - so far all metrics
have been computed on a SINGLE chronological 70:30 split, which does not
allow estimating the variance of the result across different periods of
history.

Method: the SBER ticker (the same one as in the baseline run - this closes
the "single split" limitation, not the "single instrument" one, which is a
separate limitation already closed by the wide/holdout validation) is split
into 4 NON-OVERLAPPING consecutive chronological blocks. Within each block,
a regular Backtester is used (its own 70:30 split and 5bp costs on that
block). Runs: the top-8 single-asset algorithms by Sharpe from the baseline
SBER experiment (VAE, Symbolic Regression, N-BEATS, PPO, Isolation Forest,
DDPG, One-Class SVM, Gaussian Process - multi-asset algorithms are
deliberately excluded: they need a wide ticker pool, not a time window over
a single ticker) + 3 composites from composite.py.

The result is not the average metric across the 4 windows, but the SPREAD
(mean/std/min/max) between windows: a robust algorithm should look like
itself across different periods of history, not just on average.

Run: source .venv/bin/activate && python scripts/run_walkforward_benchmarks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from core.backtester import Backtester
from core.data_loader import TInvestDataLoader

from algorithms.vae import VAEFactorModel
from algorithms.genetic_programming import SymbolicRegressionAlpha
from algorithms.nbeats import NBEATSForecaster
from algorithms.ppo_agent import PPOAgent
from algorithms.isolation_forest import IsolationForestRiskSwitch
from algorithms.ddpg_agent import DDPGAgent
from algorithms.one_class_svm import OneClassSVMRiskSwitch
from algorithms.gaussian_process import GaussianProcessTrader
from algorithms.elastic_net import ElasticNetFactorModel
from algorithms.composite import AnomalyRiskOverlay, VolTargetSizer

PRIMARY_TICKER = "SBER"
START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)
N_WINDOWS = 4
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

STRATEGIES = {
    "vae": lambda: VAEFactorModel(),
    "genetic_programming": lambda: SymbolicRegressionAlpha(),
    "nbeats": lambda: NBEATSForecaster(),
    "ppo_agent": lambda: PPOAgent(),
    "isolation_forest": lambda: IsolationForestRiskSwitch(),
    "ddpg_agent": lambda: DDPGAgent(),
    "one_class_svm": lambda: OneClassSVMRiskSwitch(),
    "gaussian_process": lambda: GaussianProcessTrader(),
    "vae_isoforest": lambda: AnomalyRiskOverlay(lambda: VAEFactorModel(), detector="isolation_forest"),
    "nbeats_ocsvm": lambda: AnomalyRiskOverlay(lambda: NBEATSForecaster(), detector="one_class_svm"),
    "elasticnet_isoforest_sized": lambda: VolTargetSizer(
        lambda: AnomalyRiskOverlay(lambda: ElasticNetFactorModel(), detector="isolation_forest")
    ),
}


def make_windows(df: pd.DataFrame, n_windows: int) -> list[pd.DataFrame]:
    """N non-overlapping consecutive blocks in a row (no shared rows)."""
    edges = np.linspace(0, len(df), n_windows + 1, dtype=int)
    return [df.iloc[edges[i]:edges[i + 1]] for i in range(n_windows)]


def run_walkforward(backtester: Backtester, windows: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec_name, factory in STRATEGIES.items():
        for wi, window_df in enumerate(windows, 1):
            result = backtester.run(factory(), window_df)
            row = {"spec_name": spec_name, "window": wi, "window_start": str(window_df.index[0].date()),
                   "window_end": str(window_df.index[-1].date())}
            row.update(result.to_flat_dict())
            rows.append(row)
        status = ", ".join(
            f"w{r['window']}:{r['sharpe_ratio']:.2f}" for r in rows if r["spec_name"] == spec_name
        )
        print(f"[walkforward] {spec_name}: {status}")
    return pd.DataFrame(rows)


def aggregate(raw_df: pd.DataFrame) -> pd.DataFrame:
    ok = raw_df[raw_df["error"].isna()]
    agg_rows = []
    for spec_name, g in ok.groupby("spec_name"):
        agg_rows.append({
            "spec_name": spec_name,
            "algorithm_name": g["algorithm_name"].iloc[0],
            "n_windows_ok": len(g),
            "mean_sharpe": g["sharpe_ratio"].mean(),
            "std_sharpe": g["sharpe_ratio"].std(),
            "min_sharpe": g["sharpe_ratio"].min(),
            "max_sharpe": g["sharpe_ratio"].max(),
            "mean_hit_rate": g["hit_rate"].mean(),
            "mean_max_drawdown": g["max_drawdown"].mean(),
            "mean_total_return": g["total_return"].mean(),
            "pct_positive_sharpe": (g["sharpe_ratio"] > 0).mean(),
        })
    return pd.DataFrame(agg_rows).sort_values("mean_sharpe", ascending=False)


def main() -> None:
    loader = TInvestDataLoader()
    df = loader.load_candles(PRIMARY_TICKER, START, END, interval="day", use_cache=True)
    print(f"{PRIMARY_TICKER}: {len(df)} rows total")

    windows = make_windows(df, N_WINDOWS)
    for i, w in enumerate(windows, 1):
        print(f"  window {i}: {w.index[0].date()} .. {w.index[-1].date()} ({len(w)} rows)")

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7)

    raw_df = run_walkforward(backtester, windows)
    agg_df = aggregate(raw_df)

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": PRIMARY_TICKER,
        "n_windows": N_WINDOWS,
        "window_bounds": [
            {"window": i + 1, "start": str(w.index[0].date()), "end": str(w.index[-1].date()), "n_rows": len(w)}
            for i, w in enumerate(windows)
        ],
        "raw": raw_df.to_dict(orient="records"),
        "aggregated": agg_df.to_dict(orient="records"),
    }
    out_path = RESULTS_DIR / "walkforward_benchmark_results.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved {out_path}")

    print("\n--- Aggregated (mean/std/min/max Sharpe across 4 windows) ---")
    print(agg_df[["algorithm_name", "mean_sharpe", "std_sharpe", "min_sharpe", "max_sharpe", "pct_positive_sharpe"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
