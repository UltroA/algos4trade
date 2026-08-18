"""
MOEX (Moscow Exchange) trading calendar - the exchange this codebase is
built and validated against by default.

No holiday calendar is modeled: a day is a trading day purely by weekday,
matching this codebase's behavior everywhere else before this module
existed (`SessionConfig.trading_days` was always a plain weekday set,
never checked against an actual MOEX holiday list). A live/demo session
that happens to run on a MOEX holiday will therefore believe the market
is open and simply see no new bars from the data provider - not modeled
as a distinct failure mode. Add a real holiday calendar here (e.g.
`session_hours` returning `None` for specific dates) if that distinction
starts to matter.
"""

from __future__ import annotations

from datetime import date, time, timedelta, timezone, tzinfo

from .base import Exchange, TradingSession

MSK = timezone(timedelta(hours=3))


class MOEXExchange(Exchange):
    """MOEX main trading session, Mon-Fri, 10:00-18:39 MSK by default."""

    name = "MOEX"

    def __init__(
        self,
        session_start: time = time(10, 0),
        session_end: time = time(18, 39),
        trading_weekdays: frozenset[int] = frozenset({0, 1, 2, 3, 4}),
    ):
        self._session = TradingSession(session_start, session_end)
        self._trading_weekdays = trading_weekdays

    @property
    def timezone(self) -> tzinfo:
        return MSK

    def session_hours(self, local_date: date) -> TradingSession | None:
        return self._session if local_date.weekday() in self._trading_weekdays else None
