"""
Exchange-agnostic trading-calendar interface.

`core.market_simulator.MarketSimulator` is written entirely against this
module for one question: is a given moment inside a trading session,
and if not, when is the next one? It holds no knowledge of MOEX or any
other specific exchange's hours, timezone, or trading week. Adding
support for another exchange means writing one new `Exchange`
implementation (see `exchanges/moex.py` for the reference implementation)
and handing it to `MarketSimulator`; nothing else in the live/demo
simulator needs to change.

This is deliberately the counterpart of `core.providers.base.
MarketDataProvider`: a provider answers "where do candles come from",
an exchange answers "when is this market open". The two are independent
- a broker API (provider) can offer access to more than one exchange, and
the same exchange calendar can apply regardless of which broker's API is
used to fetch its data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo


@dataclass(frozen=True)
class TradingSession:
    """A single trading session's open/close time, in the exchange's own local timezone."""

    start: time
    end: time


class Exchange(ABC):
    """
    A trading venue's calendar: timezone, trading days, and session hours.

    Implement this against any exchange (MOEX, another national exchange,
    a crypto venue with a 24/7 calendar, ...) to plug it into
    `MarketSimulator` without changing the simulator itself.
    """

    name: str

    @property
    @abstractmethod
    def timezone(self) -> tzinfo:
        """The timezone `session_hours` is expressed in (e.g. MSK for MOEX)."""

    @abstractmethod
    def session_hours(self, local_date: date) -> TradingSession | None:
        """
        This exchange's trading session on `local_date` (a date in
        `self.timezone`), or `None` if the exchange is closed all day
        (weekend, holiday, ...).
        """

    def is_trading_day(self, local_date: date) -> bool:
        return self.session_hours(local_date) is not None

    def is_open(self, moment: datetime) -> bool:
        """Whether `moment` (any timezone-aware datetime) falls inside a trading session."""
        local = moment.astimezone(self.timezone)
        session = self.session_hours(local.date())
        if session is None:
            return False
        return session.start <= local.time() <= session.end

    def next_open(self, moment: datetime) -> datetime:
        """The next session-open datetime strictly after `moment` (timezone-aware, in `self.timezone`)."""
        local = moment.astimezone(self.timezone)
        candidate_date = local.date()
        while True:
            session = self.session_hours(candidate_date)
            if session is not None:
                candidate = datetime.combine(candidate_date, session.start, tzinfo=self.timezone)
                if candidate > local:
                    return candidate
            candidate_date += timedelta(days=1)
