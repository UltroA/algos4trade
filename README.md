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

### Data providers

`TInvestDataLoader` is a thin, provider-specific convenience wrapper. The actual caching/chunking logic lives in `core.data_loader.MarketDataLoader`, which is written entirely against the `MarketDataProvider` interface (`core/providers/base.py`) and has no knowledge of T-Invest or any other specific broker/exchange API:

```python
from core.providers.base import MarketDataProvider, Instrument, Candle, CandleInterval

class MyBrokerProvider(MarketDataProvider):
    def resolve_instrument(self, ticker: str) -> Instrument: ...
    def fetch_candles(self, instrument_id: str, start, end, interval: CandleInterval) -> list[Candle]: ...
    # optional: max_chunk_size(interval), recent_latencies_ms()
```

```python
from core.data_loader import MarketDataLoader

loader = MarketDataLoader(MyBrokerProvider())
df = loader.load_candles("SBER", start, end, interval="day")  # same API as TInvestDataLoader
```

`core/providers/tinvest.py` is the T-Invest implementation, built on top of that interface: `TInvestClient` (raw REST client), `TInvestProvider` (adapts it to `MarketDataProvider`), and `TInvestDataLoader` (a `MarketDataLoader` subclass that wires up `TInvestProvider` automatically - what the example above uses). To plug in a different data source, write one new `MarketDataProvider` subclass; `MarketDataLoader` and every algorithm/backtester/simulator downstream stay unchanged.

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

**Money P&L and autosave**: every `results/*.md` produced by `run_benchmarks.py`, `run_all_benchmarks.py`, and the four specialized scripts above now also reports each algorithm's result in money terms, not just percentages - `starting_capital`, `final_capital`, and `pnl_rub` (default starting capital: 1,000,000 ₽, configurable via `Backtester(starting_capital=...)`/`BenchmarkRunner(starting_capital=...)`), plus a "Summary: who performed better, who worse" section and "Top by money made"/"Biggest losses" rankings alongside the existing Sharpe ranking. All of these scripts also autosave: `results/*.{json,md}` (or a `_checkpoint.json` file next to it) is rewritten after every algorithm/ticker finishes, not just once at the end, so a run killed partway through still leaves everything completed so far on disk.

## 4. Live news-sentiment monitor (LM Studio)

`scripts/run_news_monitor.py` polls Russian business-media RSS feeds, matches mentioned companies to MOEX tickers, and asks a local LLM (served by [LM Studio](https://lmstudio.ai/)) how holders/traders are likely to react - turning that into `position = direction * confidence`, logged to `data/news_signals/YYYY-MM-DD.jsonl`.

This is a **live-only, forward-looking** tool, not a backtest: RSS feeds don't expose historical archives, so there's nothing to replay for past dates. Signal quality is judged by letting `data/news_signals/*.jsonl` accumulate over time and comparing it to what prices actually did afterwards. Once enough history has built up, `src/algorithms/news_sentiment.py` (`NewsSentimentSignal`) reads a `sentiment_score` column derived from that log and plugs into `Backtester`/`BenchmarkRunner` exactly like any other algorithm.

Setup:
1. In LM Studio, download a Russian-capable model (e.g. a T-Pro-32B or T-Lite-8B GGUF quant) and start the local server (Developer tab).
2. Add to `.env`:
   ```
   LMSTUDIO_BASE_URL=http://localhost:1234/v1
   LMSTUDIO_MODEL=<model identifier as shown in LM Studio>
   ```
3. Run:
   ```bash
   PYTHONPATH=src python scripts/run_news_monitor.py            # polls every 120s
   PYTHONPATH=src python scripts/run_news_monitor.py --once      # single poll, for testing
   ```

This is a recommendation/signal tool, not automated execution: `T_INVEST_TOKEN` is read-only, and no order placement happens anywhere in this repo.

## 5. Live market simulator

`scripts/run_market_simulation.py` (`core/market_simulator.py`) runs every auto-discovered algorithm (same discovery as `run_all_benchmarks.py`) as a live paper-trading session against real MOEX/T-Invest data, rather than the fast one-shot vectorized backtest `Backtester` does. Two things make it a *simulator* and not just another backtest script:

- **A real broker ledger, not a returns formula.** Each algorithm gets its own `SimulatedBroker`: an actual cash + shares-per-ticker account that gets rebalanced tick by tick and pays `transaction_cost_bps` on every trade, the way a real broker statement would. That ledger is what "money won/lost" (`pnl_rub`, `final_capital`) is computed from - not `total_return * starting_capital`.
- **Real timing, real delay.** In the default (live) mode, every "new candle" is fetched from T-Invest at the moment it actually happens (`TInvestDataLoader.load_recent(use_cache=False)`) - there is no injected/fake latency, the session's wall-clock pacing *is* the real network round trip. `TInvestClient` now records the round-trip time of every call it makes; the final report includes those latency stats (mean/p50/p95/max). A `--demo` mode also exists purely so the whole pipeline can be exercised in minutes without waiting for real market hours: it replays already-fetched recent bars and sleeps a synthetic per-tick delay *sampled from that same real measured latency* instead of a guessed constant.

**The configurator** - `configs/market_simulator.json` - is what decides when the simulator is actually working (`session_start`/`session_end` in MSK, `trading_days`, matching the MOEX main session) and how long a run is allowed to go (`max_duration_seconds`, `poll_interval_seconds`), alongside `tickers`, `interval`, `starting_capital_rub`, `transaction_cost_bps`, `warmup_days` (history pulled to `fit()` each algorithm before going live), `autosave_every_ticks`, and `run_news_monitor`.

**Autosave**: `results/<basename>_progress.json` is rewritten every `autosave_every_ticks` ticks for as long as the session runs, so a multi-hour live session can be interrupted (Ctrl-C is caught and finalizes cleanly) or crash without losing what it already saw.

**Live news integration**: if `run_news_monitor: true` and `NewsSentimentSignal` is among the algorithms being run, the simulator spins up `scripts/run_news_monitor.py` as a background subprocess for the duration of the session, so that algorithm actually gets fresh RSS+LLM sentiment as it's published (reusing the exact log format from section 4 - `data/news_signals/YYYY-MM-DD.jsonl` - not a duplicated RSS/LLM pipeline).

**Warm-up fit isolation**: live mode fits each algorithm in its own subprocess before the tick loop starts (pass `specs={name: (module, class)}` to `MarketSimulator.run_live`, which is what the script does by default) - the same reason `run_all_benchmarks.py` isolates each algorithm: mixing lightgbm/xgboost/catboost/torch/hmmlearn/hdbscan's native runtimes in one interpreter reliably segfaults once enough distinct libraries have trained a model. `--demo` mode fits in-process instead (fine when combined with `--algorithms` to run a handful).

Output: `results/market_simulation.{json,md}` - the same kind of detailed per-algorithm stats table as the other benchmarks (money P&L included), a "who performed better/worse" ranking, and a latency-stats section - written when the session ends (session close, `max_duration_seconds` reached, or Ctrl-C).

```bash
python scripts/run_market_simulation.py                                          # live, all algorithms, configs/market_simulator.json
python scripts/run_market_simulation.py --config my_config.json --duration 3600  # live, capped at 1h
python scripts/run_market_simulation.py --demo --duration 60 \
    --algorithms buy_and_hold,sma_crossover,random_baseline                      # fast local smoke test, no waiting for market hours
```

## 6. Algorithm catalogue

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
|`news_sentiment.py`|`NewsSentimentSignal`|news_sentiment|single|

Every file is a self-contained `if __name__ == "__main__":` smoke test on synthetic data: `PYTHONPATH=src python src/algorithms/<file>.py`.

## 7. Project structure

```
alogs4trade/
  .env
  requirements.txt
  configs/
    market_simulator.json        # the live market simulator's configurator (session hours, duration, capital, ...)
  src/
    core/
      base.py        # BaseTradingAlgorithm, SingleAssetAlgorithm, MultiAssetAlgorithm
      data_loader.py        # MarketDataLoader - provider-agnostic (cache in data/cache/*.parquet; load_recent() for live/dynamic pulls)
      providers/
        base.py        # MarketDataProvider interface + Instrument/Candle/CandleInterval
        tinvest.py        # TInvestClient (REST, tracks real call latency), TInvestProvider, TInvestDataLoader
      features.py        # shared feature engineering (make_features etc.)
      trading_env.py        # environment for RL algorithms (PPO/SAC/DDPG)
      metrics.py        # sharpe/max_drawdown/win_rate/hit_rate/pnl_rub
      backtester.py        # Backtester.run(algo, data) -> BacktestResult
      benchmark_runner.py        # BenchmarkRunner - bulk run + saving + autosave
      market_simulator.py        # MarketSimulator/SessionConfig/SimulatedBroker - the live simulator
      algorithm_discovery.py        # shared "scan src/algorithms/*.py" logic
      news_feed.py        # RssNewsPoller - polls RSS, dedups (data/news_cache/)
      ticker_linker.py        # TickerLinker - company name/brand -> MOEX ticker
      llm_sentiment.py        # LMStudioSentimentClient - LLM news scoring via LM Studio
    algorithms/        # one algorithm per file
  scripts/
    prepare_data.py        # warming up the cache for a set of MOEX tickers
    run_benchmarks.py        # full run of all algorithms -> results/
    run_news_monitor.py        # live RSS -> LLM sentiment monitor -> data/news_signals/
    run_market_simulation.py        # live/demo market simulator entry point -> results/market_simulation.*
  results/
    benchmark_results.json
    benchmark_results.md
    market_simulation.json
    market_simulation.md
```

## 8. How to add your own data provider

Every algorithm/backtester/simulator in this repo consumes `core.data_loader.MarketDataLoader`, never a broker API directly - so plugging in a new market-data source (a different broker's REST API, an exchange's own API, a local database, ...) requires writing one new class and nothing else.

1. Create `src/core/providers/my_broker.py` (or anywhere importable - there is nothing special about the `providers/` package itself, it just groups the ones already provided).
2. Inherit from `MarketDataProvider` (`core/providers/base.py`) and implement its two required methods:
   - `resolve_instrument(ticker: str) -> Instrument` - look up a ticker symbol and return its identity in your API (the `id` field is whatever you'll pass back into `fetch_candles`).
   - `fetch_candles(instrument_id: str, start: datetime, end: datetime, interval: CandleInterval) -> list[Candle]` - a single page/window of OHLCV candles for that instrument. Map your API's own candle format to the generic `Candle` dataclass here (this is the only place that translation needs to happen).
3. Optionally override:
   - `max_chunk_size(interval) -> timedelta` if your API caps how long a `[start, end)` window can be per request (`MarketDataLoader` splits any longer request into chunks of this size automatically) - the default is a permissive decade, i.e. effectively unchunked.
   - `recent_latencies_ms() -> list[float]` if you want `core.market_simulator.LatencyTracker` to calibrate its simulated demo-mode delay from your provider's own real measured round-trip time instead of a documented fallback constant.
4. Use it exactly like `TInvestDataLoader`:
   ```python
   from core.data_loader import MarketDataLoader
   from core.providers.my_broker import MyBrokerProvider

   loader = MarketDataLoader(MyBrokerProvider())
   df = loader.load_candles("SBER", start, end, interval="day")
   ```
   Nothing downstream (algorithms, `Backtester`, `BenchmarkRunner`, `MarketSimulator`) needs to know or care which provider produced `df` - they all consume the same `DataFrame`/`dict[ticker, DataFrame]` shape regardless.
5. If you want a convenience wrapper matching `TInvestDataLoader`'s pattern (a `MarketDataLoader` subclass that wires up your provider automatically so callers don't need to import and construct it themselves), add a small subclass the same way `core/providers/tinvest.py` does for `TInvestDataLoader` - this is optional, `MarketDataLoader(MyBrokerProvider())` alone is a complete, working data source.

## 9. How to add your own algorithm

1. Create `src/algorithms/my_algo.py`.
2. Inherit from `SingleAssetAlgorithm` or `MultiAssetAlgorithm`.
3. Set `name`, `category` (see `core.base.AlgorithmCategory`), `description`.
4. Hyperparameters are `__init__` arguments passed to `super().__init__(**kwargs)`.
5. Implement `fit()` and `generate_signals()` without look-ahead.
6. Register it in `scripts/run_benchmarks.py` via `runner.register(...)` (required for it to be included in the main run and in `results/benchmark_results.*`). Registering it separately for `run_all_benchmarks.py`/`run_market_simulation.py` is not necessary - both find the new class automatically on the next launch, as long as the constructor does not require mandatory arguments (see "Running absolutely every algorithm" above).

## 10. Limitations (important to understand before using with real money)

This is a research project, not a production-ready trading system: no slippage beyond fixed bps, no accounting for liquidity or order limits, no portfolio risk management, no live *execution* through T-Invest (only read-only access to market data is used - `T_INVEST_TOKEN` never places an order anywhere in this repo, including in the live market simulator, which is paper trading against a `SimulatedBroker`, not a real account). The metrics in `results/benchmark_results.md` are out-of-sample on a single chronological split of a single set of MOEX stocks, and are no substitute for full walk-forward validation; money P&L figures (`pnl_rub`) are notional, computed from a configurable default starting capital of 1,000,000 ₽, not a claim about what any real account would have made. The news-sentiment monitor (section 4) additionally has no historical RSS archive to validate against, and its LLM-generated `reasoning`/`direction` is a model opinion, not a verified fact - cross-check against official disclosure (e.g. e-disclosure.ru) before acting on it.

## 11. Why this project was created

This project is part of my work on researching various algorithms for trading on the stock market.

After fully implementing the methods, tests and core components, I thought my implementation might be useful to someone.

There may be errors in the code; please report them in `pull requests`.

## 12. AI Usage

AI is used in this project to verify the correctness of certain algorithm implementations and to produce initial translations of in-code comments into English. The model used is Claude Sonnet 5.
#### Requirements for AI-Assisted Code
If you intend to contribute code written with the help of AI, the following rules are strongly recommended:
- cover such code with tests;
- verify that it contains no critical defects or model hallucinations;
- explicitly mark generated fragments in the source code;
- indicate the use of AI in the commit and specify in its description exactly which changes were AI-generated;
- Add information about AI model you used;