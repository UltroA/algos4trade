"""
Прогревает кэш дневных свечей для расширенного универсума из 100 инструментов
MOEX (scripts/moex_universe.TICKERS_100) - нужен для кросс-секционной
валидации бенчмарка (scripts/run_benchmarks_wide.py).

В отличие от prepare_data.py (10 тикеров, жёсткая проверка), здесь ошибки по
отдельным тикерам (не найден в каталоге T-Invest, делистинг, нет истории)
не прерывают загрузку остальных - они просто помечаются как FAILED и
исключаются из дальнейшего анализа.

Запуск: source .venv/bin/activate && python scripts/prepare_data_wide.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from core.data_loader import TInvestDataLoader
from moex_universe import TICKERS_100

START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)
MIN_ROWS = 300  # ~14 месяцев дневных данных - минимум для осмысленного 70/30 сплита


def main() -> None:
    loader = TInvestDataLoader()
    ok: list[str] = []
    failed: dict[str, str] = {}

    for i, ticker in enumerate(TICKERS_100, 1):
        try:
            df = loader.load_candles(ticker, START, END, interval="day")
            if len(df) < MIN_ROWS:
                failed[ticker] = f"too few rows ({len(df)} < {MIN_ROWS})"
                print(f"[{i}/100] {ticker}: SKIP - too few rows ({len(df)})")
                continue
            ok.append(ticker)
            print(f"[{i}/100] {ticker}: {len(df)} rows, {df.index.min().date()} .. {df.index.max().date()}")
        except Exception as exc:
            failed[ticker] = f"{type(exc).__name__}: {exc}"
            print(f"[{i}/100] {ticker}: FAILED - {type(exc).__name__}: {exc}")

    print(f"\nOK: {len(ok)}/100")
    print(f"FAILED/SKIPPED: {len(failed)}/100")
    for t, reason in failed.items():
        print(f"  {t}: {reason}")

    out = Path(__file__).resolve().parents[1] / "results" / "wide_universe_availability.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "OK (" + str(len(ok)) + "):\n" + "\n".join(ok) + "\n\n"
        "FAILED (" + str(len(failed)) + "):\n" + "\n".join(f"{t}: {r}" for t, r in failed.items()) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved availability report to {out}")


if __name__ == "__main__":
    main()
