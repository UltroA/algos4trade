"""
Runs all algorithms from src/algorithms/ through BenchmarkRunner on real
historical MOEX data (T-Invest API) and saves summary results to
results/benchmark_results.json and results/benchmark_results.md.

Run: source .venv/bin/activate && python scripts/run_benchmarks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from core.benchmark_runner import BenchmarkRunner
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

TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "ROSN", "NVTK", "MTSS", "TATN", "MOEX"]
PRIMARY_TICKER = "SBER"
START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)


def main() -> None:
    loader = TInvestDataLoader()
    data = loader.load_many(TICKERS, START, END, interval="day")
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} rows")

    runner = BenchmarkRunner(single_asset_data=data, results_dir="results")

    # single-asset algorithms (trade PRIMARY_TICKER)
    single_asset_algos = {
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
    for spec_name, cls in single_asset_algos.items():
        runner.register(spec_name, lambda cls=cls: cls(), ticker=PRIMARY_TICKER)

    # multi-asset algorithms
    runner.register("kalman_filter_pairs", lambda: KalmanFilterPairsTrading(), tickers=["SBER", "VTBR"])
    runner.register("hdbscan_clustering", lambda: HDBSCANPairsClustering(), tickers=TICKERS)
    runner.register("correlation_clustering", lambda: CorrelationClustering(), tickers=TICKERS)
    runner.register("thompson_bandits", lambda: ThompsonSamplingAllocator(), tickers=TICKERS)

    results = runner.run_all(verbose=True)
    runner.save(results, basename="benchmark_results")
    print("\nSaved results/benchmark_results.json and results/benchmark_results.md")
    print(results[["spec_name", "sharpe_ratio", "total_return", "max_drawdown", "error"]])


if __name__ == "__main__":
    main()
