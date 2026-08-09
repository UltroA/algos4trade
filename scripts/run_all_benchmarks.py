"""
Единая точка запуска бенчмарка для ВСЕХ алгоритмов, лежащих в src/algorithms/ -
без ручной регистрации в списке импортов (в отличие от run_benchmarks.py и
run_benchmarks_wide.py, где список алгоритмов прописан явно). Скрипт сам
сканирует папку, импортирует каждый модуль и находит в нём классы-наследники
BaseTradingAlgorithm, которые можно создать без обязательных аргументов
конструктора.

Классы, которым для создания нужен другой алгоритм (обёртки из
src/algorithms/composite.py: AnomalyRiskOverlay, VolTargetSizer,
ThompsonWithStrongArms - все принимают *_factory первым обязательным
аргументом), автоматически пропускаются с пометкой в выводе: их нельзя
осмысленно прогнать "по умолчанию", для них нужна конкретная комбинация
алгоритмов - см. scripts/run_composite_benchmarks.py.

Тикеры и издержки - тот же набор, что в базовом эксперименте
(scripts/run_benchmarks.py): 10 ликвидных акций MOEX, SBER как основной
тикер для single-asset алгоритмов, весь набор - для multi-asset. Это
инструмент быстрой проверки "все ли алгоритмы в папке живы и что они
показывают", а не замена специализированных прогонов (wide-кросс-секция,
голдхолд, walk-forward, композиты) - у каждого из них своя методика.

Каждый алгоритм обучается и тестируется в ОТДЕЛЬНОМ подпроцессе (см. режим
--worker ниже), а не в одном общем процессе. Это не перестраховка: разные
алгоритмы тянут за собой lightgbm/xgboost/catboost/torch/hmmlearn/hdbscan, у
каждого своя копия нативного рантайма (libomp и т.п.), и при автообнаружении
модули импортируются и обучаются в алфавитном порядке, а не в вручную
подобранном "безопасном" порядке, как в run_benchmarks.py. На практике это
провоцирует сегфолт нативного кода на 15-20-м алгоритме и без изоляции по
подпроцессам роняет весь прогон целиком. Подпроцесс на каждый алгоритм чинит
это ценой небольшого оверхеда на повторный импорт библиотек.

В процессе выводится прогресс-бар: N/всего, имя алгоритма, текущая стадия
бэктеста (обучение / генерация сигналов / метрики - см. Backtester.STAGE_*,
транслируется из подпроцесса построчно в реальном времени), и после
завершения каждого алгоритма - его результат (Sharpe/доходность/MDD либо
текст ошибки, включая сегфолт/таймаут отдельного подпроцесса).

Запуск: source .venv/bin/activate && python scripts/run_all_benchmarks.py
"""

import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

# Стандартное исправление для сегфолтов из-за повторной инициализации
# libomp.dylib при смешении lightgbm/xgboost/catboost/torch в одном процессе.
# Ставить нужно ДО первого импорта любой из этих библиотек. В основном режиме
# это не нужно (каждый алгоритм и так изолирован в своём подпроцессе), но не
# мешает; в --worker режиме - то же самое приведение делается на всякий случай.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "ROSN", "NVTK", "MTSS", "TATN", "MOEX"]
PRIMARY_TICKER = "SBER"
START_ISO = "2019-01-01"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

STAGE_PREFIX = "STAGE\t"
RESULT_PREFIX = "RESULT\t"


# ---------------------------------------------------------------------------
# Обнаружение алгоритмов (безопасно делать в основном процессе - сегфолты
# провоцируются обучением моделей, а не одним лишь импортом модуля).
# ---------------------------------------------------------------------------

def _is_instantiable(cls: type) -> bool:
    """True, если cls() можно вызвать без аргументов (все параметры __init__,
    кроме self/*args/**kwargs, имеют значения по умолчанию)."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty:
            return False
    return True


def discover_algorithms() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Сканирует src/algorithms/*.py и возвращает (найденные, пропущенные) -
    списки (имя_модуля, имя_класса), отсортированные по имени модуля."""
    import algorithms as algorithms_pkg
    from core.base import BaseTradingAlgorithm

    found, skipped = [], []
    algo_dir = Path(algorithms_pkg.__file__).parent
    module_names = sorted(m.name for m in pkgutil.iter_modules([str(algo_dir)]))
    for modname in module_names:
        module = importlib.import_module(f"algorithms.{modname}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue  # только классы, определённые прямо в этом файле
            if not issubclass(cls, BaseTradingAlgorithm) or cls is BaseTradingAlgorithm:
                continue
            if inspect.isabstract(cls):
                continue
            (found if _is_instantiable(cls) else skipped).append((modname, cls.__name__))
    return found, skipped

"""
Режим --worker: обучает и тестирует ОДИН алгоритм в отдельном процессе,
печатает стадии и итог в stdout построчно, чтобы родитель мог показывать
прогресс в реальном времени.
"""
def _run_worker(modname: str, clsname: str) -> None:
    from core.backtester import Backtester
    from core.base import InputMode
    from core.data_loader import TInvestDataLoader

    module = importlib.import_module(f"algorithms.{modname}")
    cls = getattr(module, clsname)
    algo = cls()

    loader = TInvestDataLoader()
    start = datetime.fromisoformat(START_ISO).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    if algo.input_mode == InputMode.SINGLE_ASSET:
        run_data = loader.load_candles(PRIMARY_TICKER, start, end, interval="day", use_cache=True)
        algo.ticker = PRIMARY_TICKER  # type: ignore[attr-defined]
    else:
        run_data = loader.load_many(TICKERS, start, end, interval="day")

    def on_stage(stage: str) -> None:
        print(f"{STAGE_PREFIX}{stage}", flush=True)

    backtester = Backtester(transaction_cost_bps=5.0, train_frac=0.7)
    result = backtester.run(algo, run_data, on_stage=on_stage)

    row = {"spec_name": modname, "class_name": clsname, "algo_display_name": algo.name}
    row.update(result.to_flat_dict())
    print(f"{RESULT_PREFIX}{json.dumps(row, default=str)}", flush=True)


class ProgressBar:
    def __init__(self, total: int, width: int = 24):
        self.total = total
        self.width = width
        self.done = 0

    def stage(self, name: str, stage: str) -> None:
        filled = int(self.width * self.done / self.total) if self.total else self.width
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r[{bar}] {self.done}/{self.total}  {name[:38]:<38}  {stage}"
        sys.stdout.write(line.ljust(110))
        sys.stdout.flush()

    def finish(self, result_line: str) -> None:
        self.done += 1
        sys.stdout.write("\r" + " " * 110 + "\r")
        print(result_line)
        sys.stdout.flush()


def format_result(algo_name: str, row: dict) -> str:
    if row.get("error"):
        return f"  {algo_name:<40} ОШИБКА: {row['error']}"
    return (
        f"  {algo_name:<40} SR={row['sharpe_ratio']:+.3f}  доходн.={row['total_return']:+.2%}  "
        f"MDD={row['max_drawdown']:.2%}  сделок={row['n_trades']}"
    )


def _run_one_algorithm(modname: str, clsname: str, bar: ProgressBar) -> dict:
    """Запускает воркер в подпроцессе, транслирует его STAGE-строки в
    прогресс-бар в реальном времени, возвращает финальную строку результата
    (либо синтезированную ошибку, если подпроцесс упал без RESULT)."""
    bar.stage(clsname, "запуск подпроцесса")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", modname, clsname],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    result_row: dict | None = None
    display_name = clsname
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip("\n")
        if line.startswith(STAGE_PREFIX):
            bar.stage(display_name, line[len(STAGE_PREFIX):])
        elif line.startswith(RESULT_PREFIX):
            result_row = json.loads(line[len(RESULT_PREFIX):])
            display_name = result_row.get("algo_display_name", clsname)
    proc.wait()
    stderr_tail = (proc.stderr.read() or "").strip().splitlines()[-1:] if proc.stderr else []

    if result_row is None:
        reason = (
            f"подпроцесс завершился сигналом {-proc.returncode} (вероятно, сегфолт нативной библиотеки)"
            if proc.returncode is not None and proc.returncode < 0
            else f"подпроцесс завершился с кодом {proc.returncode}"
        )
        if stderr_tail:
            reason += f" - {stderr_tail[0]}"
        result_row = {
            "spec_name": modname, "class_name": clsname, "algo_display_name": clsname,
            "algorithm_name": clsname, "category": "", "tickers": "", "train_seconds": 0.0,
            "inference_seconds": 0.0, "n_train_rows": 0, "n_test_rows": 0, "error": reason,
            "total_return": None, "annualized_return": None, "annualized_volatility": None,
            "sharpe_ratio": None, "max_drawdown": None, "win_rate": None, "hit_rate": None, "n_trades": None,
        }
    return result_row


def main() -> None:
    found, skipped = discover_algorithms()
    print(f"Найдено алгоритмов в src/algorithms/: {len(found)}")
    if skipped:
        print(f"Пропущено (требуют обязательных аргументов конструктора, см. run_composite_benchmarks.py): {len(skipped)}")
        for modname, clsname in skipped:
            print(f"  - {modname}.{clsname}")
    print()

    bar = ProgressBar(total=len(found))
    rows: list[dict] = []
    for modname, clsname in found:
        row = _run_one_algorithm(modname, clsname, bar)
        rows.append(row)
        bar.finish(format_result(row.get("algo_display_name", clsname), row))

    results_df = pd.DataFrame(rows).drop(columns=["algo_display_name", "class_name"], errors="ignore")
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_algorithms_found": len(found),
        "n_algorithms_skipped": len(skipped),
        "skipped": [f"{modname}.{clsname}" for modname, clsname in skipped],
        "results": results_df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "all_benchmark_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    from core.benchmark_runner import BenchmarkRunner
    (RESULTS_DIR / "all_benchmark_results.md").write_text(
        BenchmarkRunner._to_markdown(results_df), encoding="utf-8"
    )
    print("\nSaved results/all_benchmark_results.json and .md")

    n_errors = sum(1 for r in rows if r.get("error"))
    if n_errors:
        print(f"Завершилось с ошибкой: {n_errors}/{len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2], sys.argv[3])
    else:
        main()
