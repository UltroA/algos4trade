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
STARTING_CAPITAL_RUB = 1_000_000.0


def _money(x: float) -> str:
    return f"{x:,.0f} ₽".replace(",", " ") if pd.notnull(x) else ""

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
        # checkpoint after each algorithm - in case a long run gets interrupted
        pd.DataFrame(rows).to_json(
            RESULTS_DIR / "_holdout10_checkpoint.json", orient="records", indent=2, force_ascii=False
        )
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
            "mean_pnl_rub": g["pnl_rub"].mean(),
            "total_pnl_rub": g["pnl_rub"].sum(),
        })
    return pd.DataFrame(agg_rows).sort_values("mean_sharpe", ascending=False)


def save_markdown(agg_df: pd.DataFrame, thompson_df: pd.DataFrame, n_tickers: int, path: Path) -> None:
    lines = [
        "# Holdout-10 benchmark validation (instruments outside MOEXBMI-100)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Holdout tickers usable: {n_tickers}/{len(HOLDOUT_TICKERS)}",
        "",
        "## Single-asset components + composites: aggregated across the 10 holdout tickers",
        "",
    ]
    disp = agg_df.copy()
    for col in ("mean_sharpe", "median_sharpe"):
        disp[col] = disp[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    disp["pct_positive_sharpe"] = disp["pct_positive_sharpe"].map(lambda x: f"{x:.1%}")
    disp["mean_hit_rate"] = disp["mean_hit_rate"].map(lambda x: f"{x:.2%}")
    disp["mean_max_drawdown"] = disp["mean_max_drawdown"].map(lambda x: f"{x:.2%}")
    disp["mean_total_return"] = disp["mean_total_return"].map(lambda x: f"{x:.2%}")
    disp["mean_pnl_rub"] = disp["mean_pnl_rub"].map(_money)
    disp["total_pnl_rub"] = disp["total_pnl_rub"].map(_money)
    cols = ["algorithm_name", "n_tickers_ok", "mean_sharpe", "median_sharpe", "pct_positive_sharpe",
            "mean_hit_rate", "mean_max_drawdown", "mean_total_return", "mean_pnl_rub", "total_pnl_rub"]
    lines.append(disp[cols].to_markdown(index=False))
    lines.append("")

    if not agg_df.empty:
        by_money = agg_df.sort_values("total_pnl_rub", ascending=False)
        lines.append(
            f"Made the most money: **{by_money.iloc[0]['algorithm_name']}** "
            f"({_money(by_money.iloc[0]['total_pnl_rub'])}). "
            f"Lost the most: **{by_money.iloc[-1]['algorithm_name']}** "
            f"({_money(by_money.iloc[-1]['total_pnl_rub'])})."
        )
        lines.append("")

    lines.append("## Thompson Sampling: raw momentum arm vs. strong-arm composite")
    lines.append("")
    tdisp = thompson_df.copy()
    for col in ("total_return", "max_drawdown", "hit_rate", "win_rate"):
        if col in tdisp.columns:
            tdisp[col] = tdisp[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    if "sharpe_ratio" in tdisp.columns:
        tdisp["sharpe_ratio"] = tdisp["sharpe_ratio"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    for col in ("starting_capital", "final_capital", "pnl_rub"):
        if col in tdisp.columns:
            tdisp[col] = tdisp[col].map(_money)
    show_cols = [c for c in ["spec_name", "algorithm_name", "sharpe_ratio", "total_return", "max_drawdown",
                              "n_trades", "pnl_rub", "error"] if c in tdisp.columns]
    lines.append(tdisp[show_cols].to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    loader = TInvestDataLoader()
    print("Loading 10-ticker holdout set ...")
    data = load_holdout(loader)
    print(f"Holdout ready: {len(data)}/{len(HOLDOUT_TICKERS)} tickers usable\n")

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7, starting_capital=STARTING_CAPITAL_RUB)

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
    md_path = RESULTS_DIR / "holdout10_benchmark_results.md"
    save_markdown(agg_df, thompson_df, len(data), md_path)
    checkpoint = RESULTS_DIR / "_holdout10_checkpoint.json"
    if checkpoint.exists():
        checkpoint.unlink()
    print(f"\nSaved {out_path} and {md_path}")

    print("\n--- Aggregated (mean across 10 holdout tickers) ---")
    print(agg_df[["algorithm_name", "n_tickers_ok", "mean_sharpe", "median_sharpe", "pct_positive_sharpe"]])
    print("\n--- Thompson ---")
    print(thompson_df[["spec_name", "sharpe_ratio", "total_return", "max_drawdown", "n_trades"]])
    print("\n--- Per-ticker detail ---")
    print(single_df.pivot(index="ticker", columns="spec_name", values="sharpe_ratio").round(3))


if __name__ == "__main__":
    main()
