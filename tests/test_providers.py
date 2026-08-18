"""
Unit tests for core.data_loader.MarketDataLoader against a fake in-memory
MarketDataProvider - no network access, no T_INVEST_TOKEN required. Verifies
the provider-agnostic loader itself (caching, chunking, resolve_ticker
memoization) independently of any specific broker/exchange API - see
README section 8, "How to add your own data provider".
"""

from datetime import datetime, timedelta, timezone

from core.data_loader import MarketDataLoader
from core.providers.base import Candle, CandleInterval, Instrument, MarketDataProvider


class _FakeProvider(MarketDataProvider):
    """One candle per calendar day in [start, end), price = day count. Tracks
    calls so tests can assert on caching/chunking behavior."""

    def __init__(self, chunk_days: int = 3):
        self.resolve_calls: list[str] = []
        self.fetch_calls: list[tuple[str, datetime, datetime, CandleInterval]] = []
        self._chunk_days = chunk_days

    def resolve_instrument(self, ticker: str) -> Instrument:
        self.resolve_calls.append(ticker)
        return Instrument(id=f"id-{ticker}", ticker=ticker)

    def fetch_candles(self, instrument_id, start, end, interval) -> list[Candle]:
        self.fetch_calls.append((instrument_id, start, end, interval))
        candles = []
        day = start
        while day < end:
            price = float(day.toordinal())
            candles.append(Candle(time=day, open=price, high=price, low=price, close=price, volume=1.0))
            day += timedelta(days=1)
        return candles

    def max_chunk_size(self, interval: CandleInterval) -> timedelta:
        return timedelta(days=self._chunk_days)


def test_load_candles_returns_expected_shape(tmp_path):
    provider = _FakeProvider()
    loader = MarketDataLoader(provider, cache_dir=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 10, tzinfo=timezone.utc)

    df = loader.load_candles("SBER", start, end, interval="day", use_cache=False)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 9
    assert df.index.is_monotonic_increasing


def test_load_candles_chunks_requests_at_provider_max_chunk_size(tmp_path):
    provider = _FakeProvider(chunk_days=3)
    loader = MarketDataLoader(provider, cache_dir=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 10, tzinfo=timezone.utc)  # 9 days / 3-day chunks = 3 calls

    loader.load_candles("SBER", start, end, interval="day", use_cache=False)

    assert len(provider.fetch_calls) == 3
    for _, chunk_start, chunk_end, _ in provider.fetch_calls:
        assert (chunk_end - chunk_start) <= timedelta(days=3)


def test_resolve_ticker_is_memoized(tmp_path):
    provider = _FakeProvider()
    loader = MarketDataLoader(provider, cache_dir=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    loader.load_candles("SBER", start, end, use_cache=False)
    loader.load_candles("SBER", start, end, use_cache=False)

    assert provider.resolve_calls == ["SBER"]  # resolved once, cached for the second call


def test_disk_cache_avoids_a_second_fetch(tmp_path):
    provider = _FakeProvider()
    loader = MarketDataLoader(provider, cache_dir=tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)

    first = loader.load_candles("SBER", start, end, interval="day", use_cache=True)
    n_fetches_after_first = len(provider.fetch_calls)
    second = loader.load_candles("SBER", start, end, interval="day", use_cache=True)

    assert len(provider.fetch_calls) == n_fetches_after_first  # no new fetch - served from parquet cache
    assert list(first["close"]) == list(second["close"])
