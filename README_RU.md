[English](./README.md) | **Русский**

Проект реализует торговые алгоритмы как ООП-классы с единым интерфейсом, поверх реальных исторических данных MOEX из T-Invest API.

## 1. Установка и запуск

```bash
cd alogs4trade
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

В `.env` (в корне `alogs4trade/`) должен быть токен:

```
T_INVEST_TOKEN=t.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Токен можно взять только с правами **на чтение** (read-only) - для загрузки исторических свечей торговые права не нужны.

> **SSL-нюанс:** домен `*.tinkoff.ru` сейчас отдаёт сертификат, выпущенный корневым УЦ Минцифры России ("Russian Trusted Root CA"), который по умолчанию отсутствует в доверенных хранилищах сертификатов за пределами РФ. `core/tinvest_client.py` уже решает это сам - собирает CA-бандл из `certifi` + `core/certs/russian_trusted_ca.pem`. Никаких дополнительных действий не требуется.

Любой скрипт, использующий алгоритмы, должен добавить `src/` в `sys.path` (или запускаться с `PYTHONPATH=src`) и вызвать `load_dotenv()` до создания `TInvestDataLoader`:

```python
import sys
sys.path.insert(0, "/путь/до/alogs4trade/src")

from dotenv import load_dotenv
load_dotenv("/путь/до/alogs4trade/.env")

from core import TInvestDataLoader, Backtester
from algorithms.lightgbm_ranker import LightGBMRanker
```

## 2. Загрузка данных

```python
from datetime import datetime, timezone
from core import TInvestDataLoader

loader = TInvestDataLoader() # токен берётся из T_INVEST_TOKEN
df = loader.load_candles(
    "SBER",
    start=datetime(2019, 1, 1, tzinfo=timezone.utc),
    end=datetime.now(timezone.utc),
    interval="day", # "1min" | "5min" | "15min" | "hour" | "day"
)
```

> df: DataFrame с колонками open/high/low/close/volume, индекс - время (UTC)

Результат кэшируется в `data/cache/*.parquet` - повторный вызов с теми же параметрами не бьёт по сети. `loader.load_many(["SBER", "GAZP"], start, end)` возвращает `dict[ticker, DataFrame]` - нужно для мульти-активных алгоритмов.

## 3. Единый интерфейс алгоритма

Все алгоритмы наследуют `core.base.BaseTradingAlgorithm` через один из двух удобных базовых классов:

- **`SingleAssetAlgorithm`** - работает с одним инструментом:
    
    ```python
    def fit(self, train_data: pd.DataFrame) -> "Self": ...def generate_signals(self, data: pd.DataFrame) -> pd.Series: ...
    ```
    
- **`MultiAssetAlgorithm`** - работает с несколькими инструментами (пары, кластеризация, аллокация капитала):
    
    ```python
    def fit(self, train_data: dict[str, pd.DataFrame]) -> "Self": ...def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]: ...
    ```
    

`generate_signals` всегда возвращает позицию в диапазоне **[-1, 1]** (шорт..лонг) на каждый момент времени, используя только данные, известные к этому моменту (без заглядывания в будущее). `fit()` обучается только на переданных ему данных - при бэктесте это строго train-часть.

### Пример: один алгоритм на своих данных

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
signals = algo.generate_signals(test_df)   # pd.Series в [-1, 1], индекс = test_df.index
```

### Пример: полный бэктест с метриками

```python
from core import Backtester

backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7)
result = backtester.run(LightGBMRanker(), df)   # df - весь ряд, сплит внутри

print(result.metrics.sharpe_ratio, result.metrics.total_return, result.metrics.max_drawdown)
```

### Пример: мульти-активный алгоритм (парный трейдинг)

```python
from algorithms.kalman_filter_pairs import KalmanFilterPairsTrading

data = loader.load_many(["SBER", "VTBR"], datetime(2019, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc))
result = Backtester().run(KalmanFilterPairsTrading(), data)
```

### Пример: массовый прогон и сравнение алгоритмов

```python
from core import BenchmarkRunner

data = loader.load_many(["SBER", "GAZP", "LKOH"], start, end)
runner = BenchmarkRunner(single_asset_data=data, results_dir="results")
runner.register("lightgbm", lambda: LightGBMRanker(), ticker="SBER")
runner.register("kalman_pairs", lambda: KalmanFilterPairsTrading(), tickers=["SBER", "GAZP"])

results_df = runner.run_all()
runner.save(results_df, basename="my_run") # -> results/my_run.json, results/my_run.md
```

Готовый скрипт `scripts/run_benchmarks.py` уже прогоняет все 31 алгоритм на 10 ликвидных акциях MOEX и сохраняет `results/benchmark_results.{json,md}`

- именно этот файл содержит сводные результаты бенчмарков.

### Прогон вообще всех алгоритмов без ручной регистрации

`scripts/run_all_benchmarks.py` - тот же базовый прогон (10 тикеров MOEX, SBER как основной для single-asset, издержки 5 бп, сплит 70:30), но без списка импортов: скрипт сам сканирует `src/algorithms/*.py`, находит там все классы-наследники `BaseTradingAlgorithm`, которые можно создать без обязательных аргументов конструктора, и прогоняет их все. Классы, которым нужен другой алгоритм на входе (обёртки `AnomalyRiskOverlay`, `VolTargetSizer`, `ThompsonWithStrongArms` из `src/algorithms/composite.py`), автоматически пропускаются со списком причин - для них нужна конкретная комбинация алгоритмов, см. `scripts/run_composite_benchmarks.py`.

Каждый алгоритм обучается и тестируется в своём подпроцессе - это не избыточная осторожность: lightgbm/xgboost/catboost/torch/hmmlearn/hdbscan несут разные копии нативных рантаймов (libomp и т.п.), и при автоматическом (алфавитном) порядке обучения, в отличие от вручную подобранного порядка в `run_benchmarks.py`, они на практике сегфолтят интерпретатор Python на 10-20-м алгоритме. Подпроцесс на каждый алгоритм полностью изолирует такие падения - упавший алгоритм попадает в результаты как `error`, остальные прогоняются как ни в чём не бывало.

В терминале при этом виден прогресс-бар с именем алгоритма, текущей стадией (`обучение` / `генерация сигналов` / `метрики` - см. `Backtester.STAGE_*`) и результатом каждого завершённого алгоритма. Результаты сохраняются в `results/all_benchmark_results.{json,md}` в том же формате, что и у `run_benchmarks.py`.

```bash
python scripts/run_all_benchmarks.py
```

Это инструмент быстрой проверки "все ли алгоритмы в папке живы и что они показывают" - не замена специализированных прогонов (`run_benchmarks_wide.py`, `run_holdout10_benchmarks.py`, `run_walkforward_benchmarks.py`, `run_composite_benchmarks.py`), у каждого из которых своя методика (универсум тикеров, число сплитов, отбор комбинаций).

## 4. Каталог алгоритмов

Все файлы - в `src/algorithms/`. Категория соответствует строке в `docs/Алгоритмы.md`. `single` = `SingleAssetAlgorithm`, `multi` = `MultiAssetAlgorithm`.

|Файл|Класс|Категория|Режим|
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

Каждый файл - самодостаточный `if __name__ == "__main__":` смок-тест на синтетических данных: `PYTHONPATH=src python src/algorithms/<файл>.py`.

## 5. Структура проекта

```
alogs4trade/
  .env
  requirements.txt
  src/
    core/
      base.py        # BaseTradingAlgorithm, SingleAssetAlgorithm, MultiAssetAlgorithm
      tinvest_client.py        # REST-клиент T-Invest API
      data_loader.py        # TInvestDataLoader (кэш в data/cache/*.parquet)
      features.py        # общая инженерия признаков (make_features и т.д.)
      trading_env.py        # окружение для RL-алгоритмов (PPO/SAC/DDPG)
      metrics.py        # sharpe/max_drawdown/win_rate/hit_rate
      backtester.py        # Backtester.run(algo, data) -> BacktestResult
      benchmark_runner.py        # BenchmarkRunner - массовый прогон + сохранение
    algorithms/        # по одному алгоритму на файл
  scripts/
    prepare_data.py        # прогрев кэша по набору тикеров MOEX
    run_benchmarks.py        # полный прогон всех алгоритмов -> results/
  results/
    benchmark_results.json
    benchmark_results.md
```

## 6. Как добавить свой алгоритм

1. Создать `src/algorithms/my_algo.py`.
2. Отнаследоваться от `SingleAssetAlgorithm` или `MultiAssetAlgorithm`.
3. Задать `name`, `category` (см. `core.base.AlgorithmCategory`), `description`.
4. Гиперпараметры - аргументы `__init__`, переданные в `super().__init__(**kwargs)`.
5. Реализовать `fit()` и `generate_signals()` без заглядывания в будущее.
6. Зарегистрировать в `scripts/run_benchmarks.py` через `runner.register(...)` (нужно для попадания в основной прогон и в `results/benchmark_results.*`). Регистрировать отдельно для `run_all_benchmarks.py` не нужно - он находит новый класс автоматически при следующем запуске, если конструктор не требует обязательных аргументов (см. "Прогон вообще всех алгоритмов" выше).

## 7. Ограничения (важно понимать перед использованием на реальных деньгах)

Это исследовательский проект, а не production-ready торговая система: без учёта проскальзывания сверх фиксированных бп, без учёта ликвидности/лимитов заявок, без риск-менеджмента портфеля, без live-исполнения через T-Invest (используется только read-only доступ к рыночным данным). Метрики в `results/benchmark_results.md` - out-of-sample на одном хронологическом сплите одного набора акций MOEX, не заменяют полноценную walk-forward валидацию.

## 8. Зачем этот проект создан

Данный проект является частью моей работы по исследованию различных алгоритмов для торгов на фондовом рынке.

После полной реализации методов, тестов и базовых компонентов я подумал, что кому-то может быть полезна моя реализация.

В коде могут быть ошибки, просьба указывать их в `pull requests`.