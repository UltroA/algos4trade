"""
Кросс-секционная валидация бенчмарка на расширенном универсуме из ~100
инструментов MOEX (scripts/moex_universe.TICKERS_100), а не на одной акции
SBER, как в scripts/run_benchmarks.py.

Зачем: исходный бенчмарк (results/benchmark_results.{json,md}) и раздел
docs/research/*.typ явно фиксируют методологическое ограничение - почти все
single-asset алгоритмы протестированы на единственном тикере, а
multi-asset алгоритмы (в первую очередь HDBSCAN-кластеризация) недополучили
данных на пуле всего из 10 акций, из-за чего HDBSCAN не нашёл ни одного
кластера. Этот скрипт:

  1. single-asset алгоритмы (26 шт.) - обучает и тестирует НА КАЖДОМ доступном
     тикере универсума отдельно (тот же 70/30 хронологический сплит на
     каждом), затем агрегирует метрики (среднее/медиана Sharpe, доля
     тикеров с положительным Sharpe и т.д.) - это прямая проверка
     устойчивости результата к выбору инструмента.
  2. multi-asset алгоритмы, которым нужен широкий пул инструментов
     (HDBSCAN clustering, correlation clustering, Thompson bandits) -
     прогоняются на ПОЛНОМ доступном универсуме (а не на 10 акциях).
  3. Kalman Filter Pairs Trading оставлен как в базовом прогоне (SBER/VTBR) -
     это по конструкции тест на одной паре, расширение числа инструментов
     ему не даёт дополнительной информации без отдельной процедуры отбора
     пар (не входит в объём этой валидации).

Результаты сохраняются в results/benchmark_results_wide.json (полные,
по каждому тикеру) и results/benchmark_results_wide.md (агрегированная
сводная таблица + топ/антитоп).

Запуск: source .venv/bin/activate && python scripts/run_benchmarks_wide.py
Ожидаемое время выполнения: ~1-2 часа (26 алгоритмов x ~90 тикеров).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from core.backtester import Backtester
from core.data_loader import TInvestDataLoader

from algorithms.lightgbm_ranker import LightGBMRanker
from algorithms.xgboost_ranker import XGBoostRanker
from algorithms.catboost_ranker import CatBoostRanker
from algorithms.random_forest import RandomForestBaseline
from algorithms.elastic_net import ElasticNetFactorModel
from algorithms.lasso import LassoFactorModel
from algorithms.logistic_regression import LogisticRegressionDirection
from algorithms.meta_labeling import MetaLabelingModel
from algorithms.hmm_regime import HMMRegimeDetector
from algorithms.kalman_filter_pairs import KalmanFilterPairsTrading
from algorithms.hdbscan_clustering import HDBSCANPairsClustering
from algorithms.correlation_clustering import CorrelationClustering
from algorithms.lstm_predictor import LSTMPredictor
from algorithms.gru_predictor import GRUPredictor
from algorithms.tcn_predictor import TCNPredictor
from algorithms.transformer_patchtst import PatchTSTPredictor
from algorithms.informer import InformerPredictor
from algorithms.nbeats import NBEATSForecaster
from algorithms.nhits import NHiTSForecaster
from algorithms.cnn_candlestick import CandlestickCNN
from algorithms.autoencoder import AutoencoderFactorModel
from algorithms.vae import VAEFactorModel
from algorithms.isolation_forest import IsolationForestRiskSwitch
from algorithms.one_class_svm import OneClassSVMRiskSwitch
from algorithms.svm_rbf import SVMDirectionClassifier
from algorithms.ppo_agent import PPOAgent
from algorithms.sac_agent import SACAgent
from algorithms.ddpg_agent import DDPGAgent
from algorithms.thompson_bandits import ThompsonSamplingAllocator
from algorithms.gaussian_process import GaussianProcessTrader
from algorithms.genetic_programming import SymbolicRegressionAlpha
from algorithms.sma_crossover import SMACrossoverBaseline
from algorithms.buy_and_hold import BuyAndHoldBaseline
from algorithms.random_baseline import RandomPositionBaseline

from moex_universe import TICKERS_100

START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)
MIN_ROWS = 300
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SINGLE_ASSET_ALGOS = {
    "lightgbm_ranker": LightGBMRanker,
    "xgboost_ranker": XGBoostRanker,
    "catboost_ranker": CatBoostRanker,
    "random_forest": RandomForestBaseline,
    "elastic_net": ElasticNetFactorModel,
    "lasso": LassoFactorModel,
    "logistic_regression": LogisticRegressionDirection,
    "meta_labeling": MetaLabelingModel,
    "hmm_regime": HMMRegimeDetector,
    "lstm_predictor": LSTMPredictor,
    "gru_predictor": GRUPredictor,
    "tcn_predictor": TCNPredictor,
    "transformer_patchtst": PatchTSTPredictor,
    "informer": InformerPredictor,
    "nbeats": NBEATSForecaster,
    "nhits": NHiTSForecaster,
    "cnn_candlestick": CandlestickCNN,
    "autoencoder": AutoencoderFactorModel,
    "vae": VAEFactorModel,
    "isolation_forest": IsolationForestRiskSwitch,
    "one_class_svm": OneClassSVMRiskSwitch,
    "svm_rbf": SVMDirectionClassifier,
    "ppo_agent": PPOAgent,
    "sac_agent": SACAgent,
    "ddpg_agent": DDPGAgent,
    "gaussian_process": GaussianProcessTrader,
    "genetic_programming": SymbolicRegressionAlpha,
    "sma_crossover_baseline": SMACrossoverBaseline,
    "buy_and_hold_baseline": BuyAndHoldBaseline,
    "random_baseline": RandomPositionBaseline,
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
        else:
            print(f"  [universe] {ticker}: too few rows ({len(df)}), skipped")
    return data


def run_single_asset_cross_sectional(backtester: Backtester, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    n_algos, n_tickers = len(SINGLE_ASSET_ALGOS), len(data)
    t_start = time.perf_counter()
    for ai, (spec_name, cls) in enumerate(SINGLE_ASSET_ALGOS.items(), 1):
        for ti, (ticker, df) in enumerate(data.items(), 1):
            result = backtester.run(cls(), df)
            row = {"spec_name": spec_name, "ticker": ticker}
            row.update(result.to_flat_dict())
            rows.append(row)
        elapsed = time.perf_counter() - t_start
        done = ai * n_tickers
        total = n_algos * n_tickers
        print(f"[wide] {spec_name}: {n_tickers} tickers done ({done}/{total}, {elapsed:.0f}s elapsed)")
        # чекпоинт после каждого алгоритма - на случай прерывания долгого прогона
        pd.DataFrame(rows).to_json(RESULTS_DIR / "_wide_single_asset_checkpoint.json", orient="records", indent=2, force_ascii=False)
    return pd.DataFrame(rows)


def aggregate_single_asset(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["error"].isna()]
    agg_rows = []
    for spec_name, g in ok.groupby("spec_name"):
        n = len(g)
        agg_rows.append({
            "spec_name": spec_name,
            "algorithm_name": g["algorithm_name"].iloc[0],
            "category": g["category"].iloc[0],
            "n_tickers_ok": n,
            "n_tickers_total": len(df[df["spec_name"] == spec_name]),
            "mean_sharpe": g["sharpe_ratio"].mean(),
            "median_sharpe": g["sharpe_ratio"].median(),
            "std_sharpe": g["sharpe_ratio"].std(),
            "pct_positive_sharpe": (g["sharpe_ratio"] > 0).mean(),
            "mean_hit_rate": g["hit_rate"].mean(),
            "mean_max_drawdown": g["max_drawdown"].mean(),
            "mean_total_return": g["total_return"].mean(),
            "median_total_return": g["total_return"].median(),
            "mean_train_seconds": g["train_seconds"].mean(),
        })
    return pd.DataFrame(agg_rows).sort_values("mean_sharpe", ascending=False)


def run_multi_asset(backtester: Backtester, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    specs = [
        ("kalman_filter_pairs", lambda: KalmanFilterPairsTrading(), ["SBER", "VTBR"]),
        ("hdbscan_clustering", lambda: HDBSCANPairsClustering(), list(data.keys())),
        ("correlation_clustering", lambda: CorrelationClustering(), list(data.keys())),
        ("thompson_bandits", lambda: ThompsonSamplingAllocator(), list(data.keys())),
    ]
    for spec_name, factory, tickers in specs:
        tickers = [t for t in tickers if t in data]
        subset = {t: data[t] for t in tickers}
        print(f"[wide] running multi-asset {spec_name} on {len(subset)} tickers ...")
        result = backtester.run(factory(), subset)
        row = {"spec_name": spec_name, "ticker": f"<{len(subset)} tickers>"}
        row.update(result.to_flat_dict())
        rows.append(row)
        status = "ERROR: " + result.error if result.error else f"sharpe={result.metrics.sharpe_ratio:.3f}, n_trades={result.metrics.n_trades}"
        print(f"[wide] {spec_name} done -> {status}")
    return pd.DataFrame(rows)


def save_markdown(agg_df: pd.DataFrame, multi_df: pd.DataFrame, n_tickers: int, path: Path) -> None:
    lines = [
        "# Кросс-секционная валидация бенчмарка на 100-инструментном универсуме MOEX",
        "",
        f"Сгенерировано: {datetime.now(timezone.utc).isoformat()}",
        f"Доступно тикеров с достаточной историей: {n_tickers} из {len(TICKERS_100)} (MOEXBMI)",
        "",
        "## Single-asset алгоритмы: агрегированные метрики по всем доступным тикерам",
        "",
        "Каждый алгоритм обучен и протестирован независимо на каждом тикере "
        "(тот же 70/30 хронологический сплит, что и в базовом прогоне). "
        "`mean_sharpe`/`median_sharpe` - среднее/медиана Sharpe по тикерам, "
        "`pct_positive_sharpe` - доля тикеров, на которых Sharpe > 0.",
        "",
    ]
    disp = agg_df.copy()
    for col in ("mean_sharpe", "median_sharpe", "std_sharpe"):
        disp[col] = disp[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    disp["pct_positive_sharpe"] = disp["pct_positive_sharpe"].map(lambda x: f"{x:.1%}")
    disp["mean_hit_rate"] = disp["mean_hit_rate"].map(lambda x: f"{x:.2%}")
    disp["mean_max_drawdown"] = disp["mean_max_drawdown"].map(lambda x: f"{x:.2%}")
    disp["mean_total_return"] = disp["mean_total_return"].map(lambda x: f"{x:.2%}")
    disp["median_total_return"] = disp["median_total_return"].map(lambda x: f"{x:.2%}")
    disp["mean_train_seconds"] = disp["mean_train_seconds"].map(lambda x: f"{x:.3f}")
    cols = ["algorithm_name", "category", "n_tickers_ok", "n_tickers_total", "mean_sharpe",
            "median_sharpe", "std_sharpe", "pct_positive_sharpe", "mean_hit_rate",
            "mean_max_drawdown", "mean_total_return", "median_total_return", "mean_train_seconds"]
    lines.append(disp[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Multi-asset алгоритмы на полном доступном универсуме")
    lines.append("")
    mdisp = multi_df.copy()
    for col in ("total_return", "annualized_return", "max_drawdown", "win_rate", "hit_rate"):
        if col in mdisp.columns:
            mdisp[col] = mdisp[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    if "sharpe_ratio" in mdisp.columns:
        mdisp["sharpe_ratio"] = mdisp["sharpe_ratio"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    show_cols = [c for c in ["spec_name", "algorithm_name", "tickers", "sharpe_ratio", "total_return",
                              "max_drawdown", "win_rate", "hit_rate", "n_trades", "error"] if c in mdisp.columns]
    lines.append(mdisp[show_cols].to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    loader = TInvestDataLoader()
    print("Loading 100-ticker universe from cache (network only for missing tickers)...")
    data = load_universe(loader)
    print(f"Universe ready: {len(data)}/{len(TICKERS_100)} tickers usable\n")

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7)

    print("=== Single-asset cross-sectional run ===")
    single_df = run_single_asset_cross_sectional(backtester, data)
    agg_df = aggregate_single_asset(single_df)

    print("\n=== Multi-asset run on full universe ===")
    multi_df = run_multi_asset(backtester, data)

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers_universe": len(TICKERS_100),
        "n_tickers_usable": len(data),
        "tickers_usable": sorted(data.keys()),
        "single_asset_raw": single_df.to_dict(orient="records"),
        "single_asset_aggregated": agg_df.to_dict(orient="records"),
        "multi_asset": multi_df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "benchmark_results_wide.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    save_markdown(agg_df, multi_df, len(data), RESULTS_DIR / "benchmark_results_wide.md")

    checkpoint = RESULTS_DIR / "_wide_single_asset_checkpoint.json"
    if checkpoint.exists():
        checkpoint.unlink()

    print("\nSaved results/benchmark_results_wide.json and .md")
    print("\nTop-10 by mean Sharpe across tickers:")
    print(agg_df[["algorithm_name", "n_tickers_ok", "mean_sharpe", "median_sharpe", "pct_positive_sharpe"]].head(10))


if __name__ == "__main__":
    main()
