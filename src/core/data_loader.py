"""
Загрузчик исторических рыночных данных из T-Invest API с дисковым кэшем.

Используется всеми алгоритмами как единый источник OHLCV-данных, чтобы
бэктесты были воспроизводимы и не требовали повторных сетевых запросов.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .tinvest_client import TInvestClient

_INTERVAL_MAP = {
    "1min": ("CANDLE_INTERVAL_1_MIN", timedelta(days=1)),
    "5min": ("CANDLE_INTERVAL_5_MIN", timedelta(days=1)),
    "15min": ("CANDLE_INTERVAL_15_MIN", timedelta(days=1)),
    "hour": ("CANDLE_INTERVAL_HOUR", timedelta(days=30)),
    "day": ("CANDLE_INTERVAL_DAY", timedelta(days=365)),
}


def _quotation_to_float(q: dict | None) -> float:
    if not q:
        return float("nan")
    return int(q.get("units", 0)) + q.get("nano", 0) / 1e9


@dataclass(frozen=True)
class CandleRequest:
    ticker: str
    start: datetime
    end: datetime
    interval: str = "day"


class TInvestDataLoader:
    """Высокоуровневый загрузчик OHLCV-свечей с кэшированием на диск (parquet)."""

    def __init__(self, client: TInvestClient | None = None, cache_dir: str | Path = "data/cache"):
        self._client = client or TInvestClient()
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._instrument_cache: dict[str, dict] = {}

    def resolve_ticker(self, ticker: str) -> dict:
        if ticker not in self._instrument_cache:
            self._instrument_cache[ticker] = self._client.find_share_by_ticker(ticker)
        return self._instrument_cache[ticker]

    def load_candles(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None = None,
        interval: str = "day",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Возвращает DataFrame с колонками [open, high, low, close, volume],
        индекс - временная метка свечи (UTC).
        """
        if interval not in _INTERVAL_MAP:
            raise ValueError(f"Unsupported interval {interval!r}, expected one of {list(_INTERVAL_MAP)}")
        end = end or datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        cache_path = self._cache_path(ticker, start, end, interval)
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        instrument_id = self.resolve_ticker(ticker)["uid"]
        api_interval, chunk_size = _INTERVAL_MAP[interval]

        raw_candles: list[dict] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_size, end)
            raw_candles.extend(
                self._client.get_candles(
                    instrument_id,
                    chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    api_interval,
                )
            )
            chunk_start = chunk_end

        df = self._to_dataframe(raw_candles)
        if use_cache and not df.empty:
            df.to_parquet(cache_path)
        return df

    def load_many(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime | None = None,
        interval: str = "day",
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        return {t: self.load_candles(t, start, end, interval, use_cache) for t in tickers}

    @staticmethod
    def _to_dataframe(raw_candles: list[dict]) -> pd.DataFrame:
        if not raw_candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rows = []
        seen_times = set()
        for c in raw_candles:
            t = c["time"]
            if t in seen_times:
                continue
            seen_times.add(t)
            rows.append(
                {
                    "time": pd.Timestamp(t),
                    "open": _quotation_to_float(c.get("open")),
                    "high": _quotation_to_float(c.get("high")),
                    "low": _quotation_to_float(c.get("low")),
                    "close": _quotation_to_float(c.get("close")),
                    "volume": float(c.get("volume", 0)),
                }
            )
        df = pd.DataFrame(rows).set_index("time").sort_index()
        return df

    def _cache_path(self, ticker: str, start: datetime, end: datetime, interval: str) -> Path:
        fname = f"{ticker}_{interval}_{start.date()}_{end.date()}.parquet"
        return self._cache_dir / fname
