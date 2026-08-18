"""Loads and caches historical daily candles for a set of liquid MOEX stocks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from core.providers.tinvest import TInvestDataLoader

TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "ROSN", "NVTK", "MTSS", "TATN", "MOEX"]
START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)

if __name__ == "__main__":
    loader = TInvestDataLoader()
    for ticker in TICKERS:
        try:
            df = loader.load_candles(ticker, START, END, interval="day")
            print(f"{ticker}: {len(df)} rows, {df.index.min()} .. {df.index.max()}")
        except Exception as exc:
            print(f"{ticker}: FAILED - {type(exc).__name__}: {exc}")
