"""
Provider-agnostic market-data interface.

`core.data_loader.MarketDataLoader` is written entirely against this
module - it holds no knowledge of T-Invest, MOEX, or any other specific
broker/exchange API. Adding support for another data source means writing
one new `MarketDataProvider` implementation (see `providers/tinvest.py`
for the reference implementation against the T-Invest API) and handing
it to `MarketDataLoader`; no change is needed to the loader itself or to
any downstream algorithm, backtester, or live-simulator code, since all
of those consume `MarketDataLoader`, never a provider directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class CandleInterval(str, Enum):
    """Provider-agnostic candle resolution. Providers map each member to their own API's interval identifier."""

    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    HOUR = "hour"
    DAY = "day"


@dataclass(frozen=True)
class Instrument:
    """Provider-resolved identity of a tradeable ticker."""

    id: str
    ticker: str
    raw: dict | None = None


@dataclass(frozen=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(ABC):
    """
    A source of historical/live OHLCV candles for one exchange/broker API.

    Implement this against any market-data or broker API (T-Invest, a
    different broker's REST API, a local database, ...) to plug it into
    `MarketDataLoader` and, transitively, into every algorithm, backtester,
    and the live/demo market simulator in this codebase, without changing
    any of that downstream code.
    """

    @abstractmethod
    def resolve_instrument(self, ticker: str) -> Instrument:
        """Looks up a ticker symbol and returns its provider-specific identity."""

    @abstractmethod
    def fetch_candles(
        self, instrument_id: str, start: datetime, end: datetime, interval: CandleInterval
    ) -> list[Candle]:
        """A single page/window of candles - MarketDataLoader handles chunking across max_chunk_size."""

    def max_chunk_size(self, interval: CandleInterval) -> timedelta:
        """
        Largest [start, end) window this provider accepts in one
        `fetch_candles` call at the given resolution. `MarketDataLoader`
        splits any longer request into chunks of this size. The default is
        deliberately permissive (a decade, i.e. effectively unchunked);
        override it for providers with a real per-request window limit.
        """
        return timedelta(days=3650)

    def recent_latencies_ms(self) -> list[float]:
        """
        Round-trip latency (ms) of this session's most recent successful
        calls, if the provider tracks it. Used by
        `core.market_simulator.LatencyTracker` to calibrate simulated delay
        from a real measurement instead of a guessed constant. Providers
        with no real network round-trip to measure (e.g. a local database)
        can leave this at the default empty list.
        """
        return []
