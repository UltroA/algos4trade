**English** | [Русский](./README_RU.md)

The project implements trading algorithms as OOP classes with a unified interface, on top of real historical MOEX data from the T-Invest API.

## 1. Installation and setup

```bash
cd alogs4trade
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `.env` file (in the root of `alogs4trade/`) must contain a token:

```
T_INVEST_TOKEN=t.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

A **read-only** token is enough - trading permissions are not required for downloading historical candles.

> **SSL note:** the `*.tinkoff.ru` domain currently serves a certificate issued by the root CA of the Russian Ministry of Digital Development ("Russian Trusted Root CA"), which is absent by default from trusted certificate stores outside Russia. `core/tinvest_client.py` already handles this on its own - it assembles a CA bundle from `certifi` + `core/certs/russian_trusted_ca.pem`. No additional steps are required.

Any script that uses the algorithms must add `src/` to `sys.path` (or be run with `PYTHONPATH=src`) and call `load_dotenv()` before creating a `TInvestDataLoader`:

```python
import sys
sys.path.insert(0, "/path/to/alogs4trade/src")

from dotenv import load_dotenv
load_dotenv("/path/to/alogs4trade/.env")

from core import TInvestDataLoader, Backtester
from algorithms.lightgbm_ranker import LightGBMRanker
```

## 2. Loading data

```python
from datetime import datetime, timezone
from core import TInvestDataLoader

loader = TInvestDataLoader() # the token is taken from T_INVEST_TOKEN
df = loader.load_candles(
    "SBER",
    start=datetime(2019, 1, 1, tzinfo=timezone.utc),
    end=datetime.now(timezone.utc),
    interval="day", # "1min" | "5min" | "15min" | "hour" | "day"
)
```

> df: a DataFrame with open/high/low/close/volume columns, indexed by time (UTC)

The result is cached in `data/cache/*.parquet` - repeating a call with the same parameters does not hit the network. `loader.load_many(["SBER", "GAZP"], start, end)` returns `dict[ticker, DataFrame]` - needed for multi-asset algorithms.

## 3. The unified algorithm interface

All algorithms inherit from `core.base.BaseTradingAlgorithm` via one of two convenience base classes:

- **`SingleAssetAlgorithm`** - works with a single instrument:
    
    ```python
    def fit(self, train_data: pd.DataFrame) -> "Self": ...def generate_signals(self, data: pd.DataFrame) -> pd.Series: ...
    ```
    
- **`MultiAssetAlgorithm`** - works with several instruments (pairs, clustering, capital allocation):
    
    ```python
    def fit(self, train_data: dict[str, pd.DataFrame]) -> "Self": ...def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]: ...
    ```
    

`generate_signals` always returns a position in the range **[-1, 1]** (short..long) for every point in time, using only the data known at that moment (no look-ahead). `fit()` is trained only on the data passed to it - during a backtest that is strictly the train part.

### Example: a single algorithm on your own data

```python
from datetime import datetime, timezone
from core import TInvestDataLoader
from core.features import chronological_split
from algorithms.lightgbm_ranker import LightGBMRanker

loader = TInvestDataLoader()
df = loader.load_candles("SBER", datetime(2019, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc))

train_df, test_df = chronological_split(df, train_frac=0.7)

algo = LightGBMRanker(n_estimators=200, max_depth=4)
algo.fit(train_df)
signals = algo.generate_signals(test_df)   # pd.Series in [-1, 1], index = test_df.index
```

### Example: a full backtest with metrics

```python
from core import Backtester

backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7)
result = backtester.run(LightGBMRanker(), df)   # df - the whole series, the split happens inside

print(result.metrics.sharpe_ratio, result.metrics.total_return, result.metrics.max_drawdown)
```

### Example: a multi-asset algorithm (pairs trading)

```python
from algorithms.kalman_filter_pairs import KalmanFilterPairsTrading

data = loader.load_many(["SBER", "VTBR"], datetime(2019, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc))
result = Backtester().run(KalmanFilterPairsTrading(), data)
```

### Example: bulk run and comparison of algorithms

```python
from core import BenchmarkRunner

data = loader.load_many(["SBER", "GAZP", "LKOH"], start, end)
runner = BenchmarkRunner(single_asset_data=data, results_dir="results")
runner.register("lightgbm", lambda: LightGBMRanker(), ticker="SBER")
runner.register("kalman_pairs", lambda: KalmanFilterPairsTrading(), tickers=["SBER", "GAZP"])

results_df = runner.run_all()
runner.save(results_df, basename="my_run") # -> results/my_run.json, results/my_run.md
```

The ready-made script `scripts/run_benchmarks.py` already runs all 31 algorithms on 10 liquid MOEX stocks and saves `results/benchmark_results.{json,md}`

- that file is the one containing the summary benchmark results.

### Running absolutely every algorithm without manual registration

`scripts/run_all_benchmarks.py` - the same baseline run (10 MOEX tickers, SBER as the primary single-asset ticker, 5 bps costs, a 70:30 split), but without a list of imports: the script scans `src/algorithms/*.py` itself, finds every subclass of `BaseTradingAlgorithm` there that can be instantiated without mandatory constructor arguments, and runs them all. Classes that require another algorithm as input (the `AnomalyRiskOverlay`, `VolTargetSizer` and `ThompsonWithStrongArms` wrappers from `src/algorithms/composite.py`) are skipped automatically, along with the reasons - they need a specific combination of algorithms, see `scripts/run_composite_benchmarks.py`.

Each algorithm is trained and tested in its own subprocess - this is not excessive caution: lightgbm/xgboost/catboost/torch/hmmlearn/hdbscan carry different copies of native runtimes (libomp and the like), and with the automatic (alphabetical) training order, unlike the manually tuned order in `run_benchmarks.py`, they do in practice segfault the Python interpreter at the 10th-20th algorithm. A subprocess per algorithm fully isolates such crashes - the failed algorithm ends up in the results as `error`, and the rest run as if nothing had happened.

Meanwhile the terminal shows a progress bar with the algorithm name, the current stage (`обучение` / `генерация сигналов` / `метрики`, i.e. training / signal generation / metrics - see `Backtester.STAGE_*`) and the result of every finished algorithm. The results are saved to `results/all_benchmark_results.{json,md}` in the same format as those of `run_benchmarks.py`.

```bash
python scripts/run_all_benchmarks.py
```

This is a quick sanity-check tool for "are all the algorithms in the folder alive and what do they show" - not a replacement for the specialized runs (`run_benchmarks_wide.py`, `run_holdout10_benchmarks.py`, `run_walkforward_benchmarks.py`, `run_composite_benchmarks.py`), each of which has its own methodology (ticker universe, number of splits, selection of combinations).

## 4. Algorithm catalogue

All files live in `src/algorithms/`. The category matches the row in `docs/Алгоритмы.md`. `single` = `SingleAssetAlgorithm`, `multi` = `MultiAssetAlgorithm`.

|File|Class|Category|Mode|
|---|---|---|---|
|`lightgbm_ranker.py`|`LightGBMRanker`|supervised_ranking|single|
|`xgboost_ranker.py`|`XGBoostRanker`|supervised_ranking|single|
|`catboost_ranker.py`|`CatBoostRanker`|supervised_ranking|single|
|`random_forest.py`|`RandomForestBaseline`|supervised_ranking|single|
|`elastic_net.py`|`ElasticNetFactorModel`|linear_factor|single|
|`lasso.py`|`LassoFactorModel`|linear_factor|single|
|`logistic_regression.py`|`LogisticRegressionDirection`|supervised_ranking|single|
|`meta_labeling.py`|`MetaLabelingModel`|meta_labeling|single|
|`hmm_regime.py`|`HMMRegimeDetector`|regime_detection|single|
|`kalman_filter_pairs.py`|`KalmanFilterPairsTrading`|pairs_stat_arb|multi|
|`hdbscan_clustering.py`|`HDBSCANPairsClustering`|clustering|multi|
|`correlation_clustering.py`|`CorrelationClustering`|clustering|multi|
|`lstm_predictor.py`|`LSTMPredictor`|sequence_model|single|
|`gru_predictor.py`|`GRUPredictor`|sequence_model|single|
|`tcn_predictor.py`|`TCNPredictor`|sequence_model|single|
|`transformer_patchtst.py`|`PatchTSTPredictor`|sequence_model|single|
|`informer.py`|`InformerPredictor`|sequence_model|single|
|`nbeats.py`|`NBEATSForecaster`|time_series_forecast|single|
|`nhits.py`|`NHiTSForecaster`|time_series_forecast|single|
|`cnn_candlestick.py`|`CandlestickCNN`|pattern_recognition|single|
|`autoencoder.py`|`AutoencoderFactorModel`|representation_learning|single|
|`vae.py`|`VAEFactorModel`|representation_learning|single|
|`isolation_forest.py`|`IsolationForestRiskSwitch`|anomaly_detection|single|
|`one_class_svm.py`|`OneClassSVMRiskSwitch`|anomaly_detection|single|
|`svm_rbf.py`|`SVMDirectionClassifier`|supervised_ranking|single|
|`ppo_agent.py`|`PPOAgent`|reinforcement_learning|single|
|`sac_agent.py`|`SACAgent`|reinforcement_learning|single|
|`ddpg_agent.py`|`DDPGAgent`|reinforcement_learning|single|
|`thompson_bandits.py`|`ThompsonSamplingAllocator`|capital_allocation|multi|
|`gaussian_process.py`|`GaussianProcessTrader`|bayesian_optimization|single|
|`genetic_programming.py`|`SymbolicRegressionAlpha`|symbolic_regression|single|

Every file is a self-contained `if __name__ == "__main__":` smoke test on synthetic data: `PYTHONPATH=src python src/algorithms/<file>.py`.

## 5. Project structure

```
alogs4trade/
  .env
  requirements.txt
  src/
    core/
      base.py        # BaseTradingAlgorithm, SingleAssetAlgorithm, MultiAssetAlgorithm
      tinvest_client.py        # REST client for the T-Invest API
      data_loader.py        # TInvestDataLoader (cache in data/cache/*.parquet)
      features.py        # shared feature engineering (make_features etc.)
      trading_env.py        # environment for RL algorithms (PPO/SAC/DDPG)
      metrics.py        # sharpe/max_drawdown/win_rate/hit_rate
      backtester.py        # Backtester.run(algo, data) -> BacktestResult
      benchmark_runner.py        # BenchmarkRunner - bulk run + saving
    algorithms/        # one algorithm per file
  scripts/
    prepare_data.py        # warming up the cache for a set of MOEX tickers
    run_benchmarks.py        # full run of all algorithms -> results/
  results/
    benchmark_results.json
    benchmark_results.md
```

## 6. How to add your own algorithm

1. Create `src/algorithms/my_algo.py`.
2. Inherit from `SingleAssetAlgorithm` or `MultiAssetAlgorithm`.
3. Set `name`, `category` (see `core.base.AlgorithmCategory`), `description`.
4. Hyperparameters are `__init__` arguments passed to `super().__init__(**kwargs)`.
5. Implement `fit()` and `generate_signals()` without look-ahead.
6. Register it in `scripts/run_benchmarks.py` via `runner.register(...)` (required for it to be included in the main run and in `results/benchmark_results.*`). Registering it separately for `run_all_benchmarks.py` is not necessary - that script finds the new class automatically on the next launch, as long as the constructor does not require mandatory arguments (see "Running absolutely every algorithm" above).

## 7. Limitations (important to understand before using with real money)

This is a research project, not a production-ready trading system: no slippage beyond fixed bps, no accounting for liquidity or order limits, no portfolio risk management, no live execution through T-Invest (only read-only access to market data is used). The metrics in `results/benchmark_results.md` are out-of-sample on a single chronological split of a single set of MOEX stocks, and are no substitute for full walk-forward validation.

## 8. Why this project was created

This project is part of my work on researching various algorithms for trading on the stock market.

After fully implementing the methods, tests and core components, I thought my implementation might be useful to someone.

There may be errors in the code; please report them in `pull requests`.