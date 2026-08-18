"""
Unit tests for core.exchanges - the Exchange/MOEXExchange calendar
abstraction used by the live/demo market simulator (see README section 9,
"How to add your own exchange").
"""

from datetime import date, datetime, time, timezone

from core.exchanges.base import Exchange, TradingSession
from core.exchanges.moex import MOEXExchange


def test_moex_open_on_a_weekday_during_session_hours():
    exchange = MOEXExchange()
    # Monday 2026-08-17, 12:00 MSK == 09:00 UTC.
    moment = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc)
    assert exchange.is_open(moment)


def test_moex_closed_on_weekend():
    exchange = MOEXExchange()
    # Saturday 2026-08-15, 12:00 MSK == 09:00 UTC.
    moment = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)
    assert not exchange.is_open(moment)


def test_moex_closed_before_and_after_session_hours():
    exchange = MOEXExchange()
    before = datetime(2026, 8, 17, 6, 0, tzinfo=timezone.utc)   # 09:00 MSK, before 10:00 open
    after = datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc)   # 19:00 MSK, after 18:39 close
    assert not exchange.is_open(before)
    assert not exchange.is_open(after)


def test_next_open_from_weekend_lands_on_next_trading_day_session_start():
    exchange = MOEXExchange()
    saturday_noon_msk = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)
    next_open = exchange.next_open(saturday_noon_msk)
    assert next_open.date() == date(2026, 8, 17)  # the following Monday
    assert next_open.timetz().hour == 10 and next_open.timetz().minute == 0


def test_custom_exchange_overrides_moex_calendar():
    class AlwaysOpen(Exchange):
        name = "TEST-24-7"

        @property
        def timezone(self):
            return timezone.utc

        def session_hours(self, local_date: date):
            return TradingSession(time(0, 0), time(23, 59, 59))

    exchange = AlwaysOpen()
    saturday = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)
    assert exchange.is_open(saturday)  # closed on MOEX, open on this custom calendar
