"""
Provider-agnostic historical/live OHLCV candle loader with disk caching.

Used by all algorithms as the single source of OHLCV data so that
backtests are reproducible and don't require repeated network requests.

`MarketDataLoader` is written entirely against `core.providers.base.
MarketDataProvider` - it holds no knowledge of T-Invest or any other
specific broker/exchange API. Swapping data sources means writing a new
`MarketDataProvider` implementation and constructing `MarketDataLoader`
with it; this file does not change. For T-Invest specifically, see
`core.providers.tinvest.TInvestDataLoader`, a thin subclass that wires up
`TInvestProvider` automatically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .providers.base import Candle, CandleInterval, Instrument, MarketDataProvider


class MarketDataLoader:
    """High-level OHLCV candle loader with disk caching (parquet), backed by any `MarketDataProvider`."""

    def __init__(self, provider: MarketDataProvider, cache_dir: str | Path = "data/cache"):
        self._provider = provider
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._instrument_cache: dict[str, Instrument] = {}

    def resolve_ticker(self, ticker: str) -> Instrument:
        if ticker not in self._instrument_cache:
            self._instrument_cache[ticker] = self._provider.resolve_instrument(ticker)
        return self._instrument_cache[ticker]

    def load_candles(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None = None,
        interval: CandleInterval | str = CandleInterval.DAY,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns [open, high, low, close, volume],
        indexed by candle timestamp (UTC).
        """
        interval = CandleInterval(interval)
        end = end or datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        cache_path = self._cache_path(ticker, start, end, interval)
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        instrument = self.resolve_ticker(ticker)
        chunk_size = self._provider.max_chunk_size(interval)

        raw_candles: list[Candle] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_size, end)
            raw_candles.extend(self._provider.fetch_candles(instrument.id, chunk_start, chunk_end, interval))
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
        interval: CandleInterval | str = CandleInterval.DAY,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        return {t: self.load_candles(t, start, end, interval, use_cache) for t in tickers}

    def load_recent(
        self,
        ticker: str,
        interval: CandleInterval | str = CandleInterval.MIN_5,
        lookback_days: float = 5.0,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch the most recent `lookback_days` of candles up to now. Defaults
        to `use_cache=False` so callers (the live/demo market simulator, in
        particular) always get freshly pulled data from the provider instead
        of a parquet snapshot that may already be stale - this is the
        "dynamic" pull used for live polling, as opposed to `load_candles`'s
        normal cached-by-date-range behaviour used by the fast backtests.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        return self.load_candles(ticker, start, end, interval=interval, use_cache=use_cache)

    @staticmethod
    def _to_dataframe(candles: list[Candle]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rows = []
        seen_times = set()
        for c in candles:
            if c.time in seen_times:
                continue
            seen_times.add(c.time)
            rows.append(
                {
                    "time": pd.Timestamp(c.time),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
            )
        return pd.DataFrame(rows).set_index("time").sort_index()

    def _cache_path(self, ticker: str, start: datetime, end: datetime, interval: CandleInterval) -> Path:
        fname = f"{ticker}_{interval.value}_{start.date()}_{end.date()}.parquet"
        return self._cache_dir / fname
