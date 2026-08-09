"""
Validation on an independent holdout of 10 new MOEX instruments that never
appeared in the baseline run (10 stocks), the wide validation (97 MOEXBMI
stocks), or the development of the composite pipelines
(src/algorithms/composite.py). The goal is to check how well results
obtained on MOEXBMI-100 transfer to instruments outside that index (smaller/
less liquid stocks absent from the broad market index).

Tickers were selected via the MOEX ISS API: common/preferred stocks
(SECTYPE in {1,2}, INSTRID=EQIN) in TQBR trading mode, NOT included in
scripts/moex_universe.TICKERS_100, with quote history since 2019 (except for
recent IPOs, where all available history >= 300 days was taken):

    LEAS  - PJSC "Europlan Leasing Company"
    MVID  - "M.Video" PJSC
    ABIO  - PJSC "Artgen"
    DELI  - Carsharing Russia (Delimobil)
    IVAT  - PJSC IVA
    KMAZ  - KAMAZ PJSC
    MRKZ  - PJSC Rosseti North-West
    TGKN  - PJSC "TGC-14"
    GCHE  - Cherkizovo Group PJSC
    BANE  - Bashneft ANK (common)

Runs: 5 "pure" components (VAE, N-BEATS, Elastic Net, Isolation Forest,
One-Class SVM), 3 composites from composite.py, and Thompson Sampling with
a "strong arm" versus the original Thompson with raw momentum - on this
same 10-ticker set.

Run: source .venv/bin/activate && python scripts/run_holdout10_benchmarks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from core.backtester import Backtester
from core.data_loader import TInvestDataLoader

from algorithms.vae import VAEFactorModel
from algorithms.nbeats import NBEATSForecaster
from algorithms.elastic_net import ElasticNetFactorModel
from algorithms.isolation_forest import IsolationForestRiskSwitch
from algorithms.one_class_svm import OneClassSVMRiskSwitch
from algorithms.thompson_bandits import ThompsonSamplingAllocator
from algorithms.composite import AnomalyRiskOverlay, VolTargetSizer, ThompsonWithStrongArms
from algorithms.sma_crossover import SMACrossoverBaseline
from algorithms.buy_and_hold import BuyAndHoldBaseline
from algorithms.random_baseline import RandomPositionBaseline

HOLDOUT_TICKERS = ["LEAS", "MVID", "ABIO", "DELI", "IVAT", "KMAZ", "MRKZ", "TGKN", "GCHE", "BANE"]
START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)
MIN_ROWS = 300
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SINGLE_SPECS = {
    "vae": lambda: VAEFactorModel(),
    "nbeats": lambda: NBEATSForecaster(),
    "elastic_net": lambda: ElasticNetFactorModel(),
    "isolation_forest": lambda: IsolationForestRiskSwitch(),
    "one_class_svm": lambda: OneClassSVMRiskSwitch(),
    "vae_isoforest": lambda: AnomalyRiskOverlay(lambda: VAEFactorModel(), detector="isolation_forest"),
    "nbeats_ocsvm": lambda: AnomalyRiskOverlay(lambda: NBEATSForecaster(), detector="one_class_svm"),
    "elasticnet_isoforest_sized": lambda: VolTargetSizer(
        lambda: AnomalyRiskOverlay(lambda: ElasticNetFactorModel(), detector="isolation_forest")
    ),
    "sma_crossover_baseline": lambda: SMACrossoverBaseline(),
    "buy_and_hold_baseline": lambda: BuyAndHoldBaseline(),
    "random_baseline": lambda: RandomPositionBaseline(),
}


def load_holdout(loader: TInvestDataLoader) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in HOLDOUT_TICKERS:
        df = loader.load_candles(ticker, START, END, interval="day", use_cache=True)
        if len(df) >= MIN_ROWS:
            data[ticker] = df
        else:
            print(f"  [holdout] {ticker}: too few rows ({len(df)}), skipped")
    return data


def run_single(backtester: Backtester, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec_name, factory in SINGLE_SPECS.items():
        for ticker, df in data.items():
            result = backtester.run(factory(), df)
            row = {"spec_name": spec_name, "ticker": ticker}
            row.update(result.to_flat_dict())
            rows.append(row)
        print(f"[holdout10] {spec_name}: done on {len(data)} tickers")
    return pd.DataFrame(rows)


def run_thompson(backtester: Backtester, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    specs = [
        ("thompson_momentum", lambda: ThompsonSamplingAllocator()),
        (
            "thompson_strong_arms",
            lambda: ThompsonWithStrongArms(
                arm_algo_factory=lambda: AnomalyRiskOverlay(lambda: ElasticNetFactorModel(), detector="isolation_forest"),
                arm_label="Elastic Net + Isolation Forest",
            ),
        ),
    ]
    for spec_name, factory in specs:
        result = backtester.run(factory(), data)
        row = {"spec_name": spec_name}
        row.update(result.to_flat_dict())
        rows.append(row)
        status = "ERROR: " + result.error if result.error else (f"sharpe={result.metrics.sharpe_ratio:.3f}, "
                                                                f"n_trades={result.metrics.n_trades}")
        print(f"[holdout10] {spec_name} -> {status}")
    return pd.DataFrame(rows)


def aggregate(raw_df: pd.DataFrame) -> pd.DataFrame:
    ok = raw_df[raw_df["error"].isna()]
    agg_rows = []
    for spec_name, g in ok.groupby("spec_name"):
        agg_rows.append({
            "spec_name": spec_name,
            "algorithm_name": g["algorithm_name"].iloc[0],
            "n_tickers_ok": len(g),
            "mean_sharpe": g["sharpe_ratio"].mean(),
            "median_sharpe": g["sharpe_ratio"].median(),
            "pct_positive_sharpe": (g["sharpe_ratio"] > 0).mean(),
            "mean_hit_rate": g["hit_rate"].mean(),
            "mean_max_drawdown": g["max_drawdown"].mean(),
            "mean_total_return": g["total_return"].mean(),
        })
    return pd.DataFrame(agg_rows).sort_values("mean_sharpe", ascending=False)


def main() -> None:
    loader = TInvestDataLoader()
    print("Loading 10-ticker holdout set ...")
    data = load_holdout(loader)
    print(f"Holdout ready: {len(data)}/{len(HOLDOUT_TICKERS)} tickers usable\n")

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7)

    print("=== Single-asset components + composites ===")
    single_df = run_single(backtester, data)
    agg_df = aggregate(single_df)

    print("\n=== Thompson: momentum vs strong arms ===")
    thompson_df = run_thompson(backtester, data)

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_tickers": sorted(data.keys()),
        "single_raw": single_df.to_dict(orient="records"),
        "single_aggregated": agg_df.to_dict(orient="records"),
        "thompson": thompson_df.to_dict(orient="records"),
    }
    out_path = RESULTS_DIR / "holdout10_benchmark_results.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved {out_path}")

    print("\n--- Aggregated (mean across 10 holdout tickers) ---")
    print(agg_df[["algorithm_name", "n_tickers_ok", "mean_sharpe", "median_sharpe", "pct_positive_sharpe"]])
    print("\n--- Thompson ---")
    print(thompson_df[["spec_name", "sharpe_ratio", "total_return", "max_drawdown", "n_trades"]])
    print("\n--- Per-ticker detail ---")
    print(single_df.pivot(index="ticker", columns="spec_name", values="sharpe_ratio").round(3))


if __name__ == "__main__":
    main()
