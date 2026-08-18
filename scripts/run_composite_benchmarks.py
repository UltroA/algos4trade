"""
Benchmark of composite pipelines (src/algorithms/composite.py): combine
already-tested algorithms so that one's documented weakness is covered by
another's documented strength. See composite.py's docstring for the
rationale behind each combination.

Composites tested:
  - VAE + Isolation Forest overlay            (single-asset)
  - N-BEATS + One-Class SVM overlay            (single-asset)
  - Elastic Net + Isolation Forest overlay + vol-targeted sizing  (single-asset, 3-stage)
  - Thompson Sampling with "strong arms" = Elastic Net + Isolation Forest overlay (multi-asset)

Each single-asset composite is run:
  (a) on SBER - for direct comparison with the baseline (10-ticker) run;
  (b) cross-sectionally on 97 MOEXBMI tickers - for comparison with the wide
      validation (same aggregates: mean/median Sharpe, share positive).

Thompson-with-strong-arms is run on baseline-10, on a 70-ticker calendar-
aligned set, and on the full 97-ticker set - for direct comparison with
results/benchmark_results.json and results/benchmark_results_wide.json.

Run: source .venv/bin/activate && python scripts/run_composite_benchmarks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from core.backtester import Backtester
from core.providers.tinvest import TInvestDataLoader

from algorithms.vae import VAEFactorModel
from algorithms.nbeats import NBEATSForecaster
from algorithms.elastic_net import ElasticNetFactorModel
from algorithms.composite import AnomalyRiskOverlay, VolTargetSizer, ThompsonWithStrongArms

from moex_universe import TICKERS_100

START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)
MIN_ROWS = 300
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
STARTING_CAPITAL_RUB = 1_000_000.0


def _money(x: float) -> str:
    return f"{x:,.0f} ₽".replace(",", " ") if pd.notnull(x) else ""

PRIMARY_TICKER = "SBER"
BASELINE_10 = ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "ROSN", "NVTK", "MTSS", "TATN", "MOEX"]
LONGHIST_70 = set("""AFKS AFLT AKRN ALRS APTK AQUA BANEP BELU BSPB CBOM CHMF ELFV ENPG ETLN FEES FESH
GAZP GMKN HYDR IRAO LKOH LSNGP LSRG MAGN MGNT MOEX MRKC MRKP MRKU MRKV MSNG MSRS MTLR MTLRP MTSS
NKHP NKNCP NLMK NMTP NVTK OGKB PHOR PIKK PLZL RAGR RASP RNFT ROSN RTKM RTKMP RUAL SBER SBERP SELG
SFIN SNGS SNGSP SVAV T TATN TATNP TGKA TRMK TRNFP UPRO UWGN VSMO VTBR X5 YDEX""".split())

SINGLE_COMPOSITES = {
    "vae_isoforest": lambda: AnomalyRiskOverlay(lambda: VAEFactorModel(), detector="isolation_forest"),
    "nbeats_ocsvm": lambda: AnomalyRiskOverlay(lambda: NBEATSForecaster(), detector="one_class_svm"),
    "elasticnet_isoforest_sized": lambda: VolTargetSizer(
        lambda: AnomalyRiskOverlay(lambda: ElasticNetFactorModel(), detector="isolation_forest")
    ),
}


def load_universe(loader: TInvestDataLoader) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in TICKERS_100:
        try:
            df = loader.load_candles(ticker, START, END, interval="day", use_cache=True)
        except Exception as exc:
            print(f"  [universe] {ticker}: load failed - {type(exc).__name__}: {exc}")
            continue
        if len(df) >= MIN_ROWS:
            data[ticker] = df
    return data


def run_single_sber(backtester: Backtester, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec_name, factory in SINGLE_COMPOSITES.items():
        result = backtester.run(factory(), data[PRIMARY_TICKER])
        row = {"spec_name": spec_name, "ticker": PRIMARY_TICKER}
        row.update(result.to_flat_dict())
        rows.append(row)
        status = "ERROR: " + result.error if result.error else f"sharpe={result.metrics.sharpe_ratio:.3f}"
        print(f"[composite/SBER] {spec_name} -> {status}")
    return pd.DataFrame(rows)


def run_single_wide(backtester: Backtester, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    n_tickers = len(data)
    for spec_name, factory in SINGLE_COMPOSITES.items():
        for ticker, df in data.items():
            result = backtester.run(factory(), df)
            row = {"spec_name": spec_name, "ticker": ticker}
            row.update(result.to_flat_dict())
            rows.append(row)
        print(f"[composite/wide] {spec_name}: {n_tickers} tickers done")
        # checkpoint after each composite - in case a long run gets interrupted
        pd.DataFrame(rows).to_json(
            RESULTS_DIR / "_composite_wide_checkpoint.json", orient="records", indent=2, force_ascii=False
        )
    raw_df = pd.DataFrame(rows)

    agg_rows = []
    ok = raw_df[raw_df["error"].isna()]
    for spec_name, g in ok.groupby("spec_name"):
        n = len(g)
        agg_rows.append({
            "spec_name": spec_name,
            "algorithm_name": g["algorithm_name"].iloc[0],
            "n_tickers_ok": n,
            "n_tickers_total": len(raw_df[raw_df["spec_name"] == spec_name]),
            "mean_sharpe": g["sharpe_ratio"].mean(),
            "median_sharpe": g["sharpe_ratio"].median(),
            "std_sharpe": g["sharpe_ratio"].std(),
            "pct_positive_sharpe": (g["sharpe_ratio"] > 0).mean(),
            "mean_hit_rate": g["hit_rate"].mean(),
            "mean_max_drawdown": g["max_drawdown"].mean(),
            "mean_total_return": g["total_return"].mean(),
            "median_total_return": g["total_return"].median(),
            "mean_pnl_rub": g["pnl_rub"].mean(),
            "total_pnl_rub": g["pnl_rub"].sum(),
        })
    agg_df = pd.DataFrame(agg_rows).sort_values("mean_sharpe", ascending=False)
    return raw_df, agg_df


def run_thompson_strong(backtester: Backtester, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    universes = {
        "baseline-10": {t: data[t] for t in BASELINE_10 if t in data},
        "longhist-70": {t: data[t] for t in data if t in LONGHIST_70},
        "wide-97": data,
    }
    for label, subset in universes.items():
        algo = ThompsonWithStrongArms(
            arm_algo_factory=lambda: AnomalyRiskOverlay(lambda: ElasticNetFactorModel(), detector="isolation_forest"),
            arm_label="Elastic Net + Isolation Forest",
        )
        print(f"[composite/thompson] running on {label} ({len(subset)} tickers) ...")
        result = backtester.run(algo, subset)
        row = {"spec_name": f"thompson_strong_{label}", "universe": label, "n_tickers": len(subset)}
        row.update(result.to_flat_dict())
        rows.append(row)
        status = "ERROR: " + result.error if result.error else (f"sharpe={result.metrics.sharpe_ratio:.3f}, "
                                                                f"n_trades={result.metrics.n_trades}")
        print(f"[composite/thompson] {label} -> {status}")
    return pd.DataFrame(rows)


def save_markdown(sber_df: pd.DataFrame, wide_agg_df: pd.DataFrame, thompson_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Composite pipeline benchmark results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Single-asset composites on SBER",
        "",
    ]
    sdisp = sber_df.copy()
    for col in ("total_return", "max_drawdown", "hit_rate", "win_rate"):
        if col in sdisp.columns:
            sdisp[col] = sdisp[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    if "sharpe_ratio" in sdisp.columns:
        sdisp["sharpe_ratio"] = sdisp["sharpe_ratio"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    for col in ("starting_capital", "final_capital", "pnl_rub"):
        if col in sdisp.columns:
            sdisp[col] = sdisp[col].map(_money)
    show_cols = [c for c in ["spec_name", "algorithm_name", "sharpe_ratio", "total_return", "max_drawdown",
                              "n_trades", "pnl_rub", "error"] if c in sdisp.columns]
    lines.append(sdisp[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Single-asset composites, cross-sectional (97-ticker MOEXBMI universe)")
    lines.append("")
    wdisp = wide_agg_df.copy()
    for col in ("mean_sharpe", "median_sharpe", "std_sharpe"):
        if col in wdisp.columns:
            wdisp[col] = wdisp[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    for col in ("pct_positive_sharpe",):
        if col in wdisp.columns:
            wdisp[col] = wdisp[col].map(lambda x: f"{x:.1%}")
    for col in ("mean_hit_rate", "mean_max_drawdown", "mean_total_return", "median_total_return"):
        if col in wdisp.columns:
            wdisp[col] = wdisp[col].map(lambda x: f"{x:.2%}")
    for col in ("mean_pnl_rub", "total_pnl_rub"):
        if col in wdisp.columns:
            wdisp[col] = wdisp[col].map(_money)
    lines.append(wdisp.to_markdown(index=False))
    lines.append("")
    if not wide_agg_df.empty:
        by_money = wide_agg_df.sort_values("total_pnl_rub", ascending=False)
        lines.append(
            f"Made the most money: **{by_money.iloc[0]['algorithm_name']}** "
            f"({_money(by_money.iloc[0]['total_pnl_rub'])}). "
            f"Lost the most: **{by_money.iloc[-1]['algorithm_name']}** "
            f"({_money(by_money.iloc[-1]['total_pnl_rub'])})."
        )
        lines.append("")

    lines.append("## Thompson Sampling with strong arms, across ticker universes")
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
    show_cols = [c for c in ["universe", "n_tickers", "sharpe_ratio", "total_return", "max_drawdown",
                              "n_trades", "pnl_rub", "error"] if c in tdisp.columns]
    lines.append(tdisp[show_cols].to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    loader = TInvestDataLoader()
    print("Loading universe from cache ...")
    data = load_universe(loader)
    print(f"Universe ready: {len(data)}/{len(TICKERS_100)} tickers usable\n")

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7, starting_capital=STARTING_CAPITAL_RUB)

    print("=== Single-asset composites on SBER ===")
    sber_df = run_single_sber(backtester, data)

    print("\n=== Single-asset composites, cross-sectional (wide) ===")
    wide_raw_df, wide_agg_df = run_single_wide(backtester, data)

    print("\n=== Thompson Sampling with strong arms ===")
    thompson_df = run_thompson_strong(backtester, data)

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sber_single": sber_df.to_dict(orient="records"),
        "wide_single_raw": wide_raw_df.to_dict(orient="records"),
        "wide_single_aggregated": wide_agg_df.to_dict(orient="records"),
        "thompson_strong_arms": thompson_df.to_dict(orient="records"),
    }
    out_path = RESULTS_DIR / "composite_benchmark_results.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path = RESULTS_DIR / "composite_benchmark_results.md"
    save_markdown(sber_df, wide_agg_df, thompson_df, md_path)
    checkpoint = RESULTS_DIR / "_composite_wide_checkpoint.json"
    if checkpoint.exists():
        checkpoint.unlink()
    print(f"\nSaved {out_path} and {md_path}")

    print("\n--- SBER results ---")
    print(sber_df[["spec_name", "sharpe_ratio", "total_return", "max_drawdown", "n_trades"]])
    print("\n--- Wide aggregated ---")
    print(wide_agg_df[["algorithm_name", "n_tickers_ok", "mean_sharpe", "median_sharpe", "pct_positive_sharpe"]])
    print("\n--- Thompson strong arms ---")
    print(thompson_df[["universe", "n_tickers", "sharpe_ratio", "total_return", "max_drawdown", "n_trades"]])


if __name__ == "__main__":
    main()
