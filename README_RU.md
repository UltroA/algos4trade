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

**Деньги (P&L) и автосохранение**: каждый `results/*.md`, который создают `run_benchmarks.py`, `run_all_benchmarks.py` и четыре специализированных скрипта выше, теперь показывает результат каждого алгоритма ещё и в деньгах, а не только в процентах - `starting_capital`, `final_capital` и `pnl_rub` (стартовый капитал по умолчанию: 1 000 000 ₽, настраивается через `Backtester(starting_capital=...)`/`BenchmarkRunner(starting_capital=...)`), плюс раздел "Summary: who performed better, who worse" и рейтинги "Top by money made"/"Biggest losses" рядом с уже существующим рейтингом по Sharpe. Все эти скрипты также делают автосохранение: `results/*.{json,md}` (либо файл `_checkpoint.json` рядом) перезаписывается после каждого завершённого алгоритма/тикера, а не только один раз в конце - так что прерванный на середине прогон всё равно оставляет на диске всё, что успело досчитаться.

## 4. Живой мониторинг новостей (LM Studio)

`scripts/run_news_monitor.py` опрашивает RSS-ленты российских деловых СМИ, сопоставляет упомянутые компании с тикерами MOEX и спрашивает локальную LLM (через [LM Studio](https://lmstudio.ai/)), как держатели/трейдеры вероятно отреагируют на новость - превращая это в `position = direction * confidence`, которая логируется в `data/news_signals/YYYY-MM-DD.jsonl`.

Это **только живой, форвардный** инструмент, а не бэктест: RSS-ленты не отдают исторический архив, поэтому переигрывать прошлые даты нечем. Качество сигнала оценивается по накоплению `data/news_signals/*.jsonl` во времени и сравнению с тем, что цена делала после. Когда истории накопится достаточно, `src/algorithms/news_sentiment.py` (`NewsSentimentSignal`) читает колонку `sentiment_score`, построенную из этого лога, и подключается к `Backtester`/`BenchmarkRunner` точно так же, как любой другой алгоритм.

Настройка:
1. В LM Studio скачать русскоязычную модель (например, GGUF-квант T-Pro-32B или T-Lite-8B) и запустить локальный сервер (вкладка Developer).
2. Добавить в `.env`:
   ```
   LMSTUDIO_BASE_URL=http://localhost:1234/v1
   LMSTUDIO_MODEL=<идентификатор модели, как он показан в LM Studio>
   ```
3. Запустить:
   ```bash
   PYTHONPATH=src python scripts/run_news_monitor.py            # опрос каждые 120с
   PYTHONPATH=src python scripts/run_news_monitor.py --once      # один опрос, для теста
   ```

Это инструмент рекомендаций/сигналов, а не автоматическое исполнение: `T_INVEST_TOKEN` доступен только на чтение, и выставления заявок нигде в этом репозитории не происходит.

## 5. Живой симулятор рынка

`scripts/run_market_simulation.py` (`core/market_simulator.py`) прогоняет все автоматически найденные алгоритмы (то же обнаружение, что и в `run_all_benchmarks.py`) как живую бумажную торговую сессию на реальных данных MOEX/T-Invest - в отличие от быстрого одноразового векторного бэктеста `Backtester`. Симулятором это делают две вещи:

- **Настоящий брокерский учёт, а не формула доходности.** У каждого алгоритма свой `SimulatedBroker` - реальный счёт с наличными и позициями по каждому тикеру, который перебалансируется на каждом тике и платит `transaction_cost_bps` за каждую сделку, как настоящая брокерская выписка. Именно из этого учёта считаются "деньги выиграно/проиграно" (`pnl_rub`, `final_capital`), а не по формуле `total_return * starting_capital`.
- **Настоящее время, настоящая задержка.** В режиме по умолчанию (live) каждая "новая свеча" реально запрашивается у T-Invest в момент, когда она появляется (`TInvestDataLoader.load_recent(use_cache=False)`) - никакая задержка не имитируется, темп сессии в реальном времени - это и есть настоящий сетевой round-trip. `TInvestClient` теперь фиксирует время отклика каждого своего вызова; в итоговом отчёте есть статистика этой задержки (mean/p50/p95/max). Отдельно есть режим `--demo` - он существует только для того, чтобы весь конвейер можно было прогнать за минуты, не дожидаясь реальных торговых часов: он переигрывает уже загруженные недавние свечи и делает синтетическую паузу на каждый тик, *взятую из этой же реально измеренной задержки*, а не из выдуманной константы.

**Конфигуратор** - `configs/market_simulator.json` - определяет, когда симулятор вообще работает (`session_start`/`session_end` по МСК, `trading_days`, соответствуют основной сессии MOEX) и как долго может идти прогон (`max_duration_seconds`, `poll_interval_seconds`), а также `tickers`, `interval`, `starting_capital_rub`, `transaction_cost_bps`, `warmup_days` (история, на которой каждый алгоритм обучается перед выходом в live), `autosave_every_ticks` и `run_news_monitor`.

**Автосохранение**: `results/<basename>_progress.json` перезаписывается каждые `autosave_every_ticks` тиков на всём протяжении сессии - многочасовую живую сессию можно прервать (Ctrl-C перехватывается и корректно завершает сессию) или она может упасть, не потеряв то, что уже успела увидеть.

**Живая интеграция с новостями**: если `run_news_monitor: true` и среди алгоритмов есть `NewsSentimentSignal`, симулятор на время сессии поднимает `scripts/run_news_monitor.py` фоновым подпроцессом, чтобы этот алгоритм реально получал свежий RSS+LLM-сентимент по мере публикации новостей (используется тот же формат лога из раздела 4 - `data/news_signals/YYYY-MM-DD.jsonl`, RSS/LLM-пайплайн не дублируется).

**Изоляция обучения при разогреве**: live-режим обучает каждый алгоритм в отдельном подпроцессе перед стартом цикла тиков (в `MarketSimulator.run_live` передаётся `specs={имя: (модуль, класс)}` - именно так делает скрипт по умолчанию) - по той же причине, по которой `run_all_benchmarks.py` изолирует каждый алгоритм: смешивание нативных рантаймов lightgbm/xgboost/catboost/torch/hmmlearn/hdbscan в одном интерпретаторе на практике сегфолтит после обучения на достаточном числе разных библиотек. Режим `--demo` вместо этого обучает алгоритмы в основном процессе (нормально, если вместе с `--algorithms` запускается небольшой набор).

Результат: `results/market_simulation.{json,md}` - такая же подробная таблица статистики по каждому алгоритму, как и у остальных бенчмарков (с учётом денег P&L), рейтинг "кто показал себя лучше, кто хуже" и раздел со статистикой задержки - записывается, когда сессия завершается (закрытие сессии, достигнут `max_duration_seconds` или Ctrl-C).

```bash
python scripts/run_market_simulation.py                                          # live, все алгоритмы, configs/market_simulator.json
python scripts/run_market_simulation.py --config my_config.json --duration 3600  # live, ограничено 1 часом
python scripts/run_market_simulation.py --demo --duration 60 \
    --algorithms buy_and_hold,sma_crossover,random_baseline                      # быстрый локальный смок-тест, без ожидания торговых часов
```

## 6. Каталог алгоритмов

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
|`news_sentiment.py`|`NewsSentimentSignal`|news_sentiment|single|

Каждый файл - самодостаточный `if __name__ == "__main__":` смок-тест на синтетических данных: `PYTHONPATH=src python src/algorithms/<файл>.py`.

## 7. Структура проекта

```
alogs4trade/
  .env
  requirements.txt
  configs/
    market_simulator.json        # конфигуратор живого симулятора рынка (торговые часы, длительность, капитал, ...)
  src/
    core/
      base.py        # BaseTradingAlgorithm, SingleAssetAlgorithm, MultiAssetAlgorithm
      tinvest_client.py        # REST-клиент T-Invest API (теперь также фиксирует реальную задержку вызовов)
      data_loader.py        # TInvestDataLoader (кэш в data/cache/*.parquet; load_recent() для live/динамических загрузок)
      features.py        # общая инженерия признаков (make_features и т.д.)
      trading_env.py        # окружение для RL-алгоритмов (PPO/SAC/DDPG)
      metrics.py        # sharpe/max_drawdown/win_rate/hit_rate/pnl_rub
      backtester.py        # Backtester.run(algo, data) -> BacktestResult
      benchmark_runner.py        # BenchmarkRunner - массовый прогон + сохранение + автосохранение
      market_simulator.py        # MarketSimulator/SessionConfig/SimulatedBroker - живой симулятор
      algorithm_discovery.py        # общая логика "просканировать src/algorithms/*.py"
      news_feed.py        # RssNewsPoller - опрос RSS, дедуп (data/news_cache/)
      ticker_linker.py        # TickerLinker - название/бренд компании -> тикер MOEX
      llm_sentiment.py        # LMStudioSentimentClient - оценка новостей через LM Studio
    algorithms/        # по одному алгоритму на файл
  scripts/
    prepare_data.py        # прогрев кэша по набору тикеров MOEX
    run_benchmarks.py        # полный прогон всех алгоритмов -> results/
    run_news_monitor.py        # живой мониторинг RSS -> LLM-сентимент -> data/news_signals/
    run_market_simulation.py        # точка входа живого/демо-симулятора рынка -> results/market_simulation.*
  results/
    benchmark_results.json
    benchmark_results.md
    market_simulation.json
    market_simulation.md
```

## 8. Как добавить свой алгоритм

1. Создать `src/algorithms/my_algo.py`.
2. Отнаследоваться от `SingleAssetAlgorithm` или `MultiAssetAlgorithm`.
3. Задать `name`, `category` (см. `core.base.AlgorithmCategory`), `description`.
4. Гиперпараметры - аргументы `__init__`, переданные в `super().__init__(**kwargs)`.
5. Реализовать `fit()` и `generate_signals()` без заглядывания в будущее.
6. Зарегистрировать в `scripts/run_benchmarks.py` через `runner.register(...)` (нужно для попадания в основной прогон и в `results/benchmark_results.*`). Регистрировать отдельно для `run_all_benchmarks.py`/`run_market_simulation.py` не нужно - оба находят новый класс автоматически при следующем запуске, если конструктор не требует обязательных аргументов (см. "Прогон вообще всех алгоритмов" выше).

## 9. Ограничения (важно понимать перед использованием на реальных деньгах)

Это исследовательский проект, а не production-ready торговая система: без учёта проскальзывания сверх фиксированных бп, без учёта ликвидности/лимитов заявок, без риск-менеджмента портфеля, без реального *исполнения* через T-Invest (используется только read-only доступ к рыночным данным - `T_INVEST_TOKEN` нигде в этом репозитории, включая живой симулятор рынка, не выставляет заявки; симулятор ведёт бумажную торговлю через `SimulatedBroker`, а не реальный счёт). Метрики в `results/benchmark_results.md` - out-of-sample на одном хронологическом сплите одного набора акций MOEX, не заменяют полноценную walk-forward валидацию; денежные показатели P&L (`pnl_rub`) условны и считаются от настраиваемого стартового капитала по умолчанию в 1 000 000 ₽, а не являются утверждением о том, что заработал бы реальный счёт. У мониторинга новостей (раздел 4) дополнительно нет исторического RSS-архива для проверки, а его LLM-генерируемые `reasoning`/`direction` - это мнение модели, а не проверенный факт: сверяйтесь с официальным раскрытием (например, e-disclosure.ru), прежде чем действовать на его основе.

## 10. Зачем этот проект создан

Данный проект является частью моей работы по исследованию различных алгоритмов для торгов на фондовом рынке.

После полной реализации методов, тестов и базовых компонентов я подумал, что кому-то может быть полезна моя реализация.

В коде могут быть ошибки, просьба указывать их в `pull requests`.

## 11. Использование ИИ

В проекте ИИ применяется для верификации корректности реализации отдельных алгоритмов, а также для первичного перевода комментариев в коде на английский язык. Используемая модель — Claude Sonnet 5.
#### Требования к коду, созданному с помощью ИИ
Если вы планируете внести в репозиторий код, написанный с использованием ИИ, настоятельно рекомендуется соблюдать следующие правила:
- покрыть такой код тестами;
- убедиться в отсутствии критических ошибок и галлюцинаций модели;
- явно пометить сгенерированные фрагменты в исходном коде;
- отметить факт использования ИИ в коммите и указать в его описании, какие именно изменения были сгенерированы;
- добавить информацию о использованной модели;
