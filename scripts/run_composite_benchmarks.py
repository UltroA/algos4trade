"""
Бенчмарк композитных пайплайнов (src/algorithms/composite.py): объединяют
уже протестированные алгоритмы так, чтобы задокументированный недостаток
одного покрывался задокументированной сильной стороной другого. См.
docstring composite.py для обоснования каждой комбинации.

Тестируемые композиты:
  - VAE + Isolation Forest overlay          (single-asset)
  - N-BEATS + One-Class SVM overlay          (single-asset)
  - Elastic Net + Isolation Forest overlay + vol-targeted sizing  (single-asset, 3 звена)
  - Thompson Sampling с "сильными руками" = Elastic Net + Isolation Forest overlay (multi-asset)

Каждый single-asset композит прогоняется:
  (a) на SBER - для прямого сопоставления с базовым (10-тикерным) прогоном;
  (b) кросс-секционно на 97 тикерах MOEXBMI - для сопоставления с расширенной
      валидацией (те же агрегаты: mean/median Sharpe, доля положительных).

Thompson-с-сильными-руками прогоняется на baseline-10, на 70-тикерном
календарно согласованном наборе и на полном наборе 97 тикеров - для прямого
сопоставления с results/benchmark_results.json и results/benchmark_results_wide.json.

Запуск: source .venv/bin/activate && python scripts/run_composite_benchmarks.py
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
from core.data_loader import TInvestDataLoader

from algorithms.vae import VAEFactorModel
from algorithms.nbeats import NBEATSForecaster
from algorithms.elastic_net import ElasticNetFactorModel
from algorithms.composite import AnomalyRiskOverlay, VolTargetSizer, ThompsonWithStrongArms

from moex_universe import TICKERS_100

START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)
MIN_ROWS = 300
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

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


def main() -> None:
    loader = TInvestDataLoader()
    print("Loading universe from cache ...")
    data = load_universe(loader)
    print(f"Universe ready: {len(data)}/{len(TICKERS_100)} tickers usable\n")

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7)

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
    print(f"\nSaved {out_path}")

    print("\n--- SBER results ---")
    print(sber_df[["spec_name", "sharpe_ratio", "total_return", "max_drawdown", "n_trades"]])
    print("\n--- Wide aggregated ---")
    print(wide_agg_df[["algorithm_name", "n_tickers_ok", "mean_sharpe", "median_sharpe", "pct_positive_sharpe"]])
    print("\n--- Thompson strong arms ---")
    print(thompson_df[["universe", "n_tickers", "sharpe_ratio", "total_return", "max_drawdown", "n_trades"]])


if __name__ == "__main__":
    main()
