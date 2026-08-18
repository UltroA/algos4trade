# NOTE: this package intentionally does not import `.tinvest` here. `core.data_loader`
# imports `core.providers.base` (to stay provider-agnostic), which forces this __init__ to
# run first; `core.providers.tinvest` in turn imports `MarketDataLoader` back from
# `core.data_loader`. Re-exporting `.tinvest` from this __init__ would make that a real
# circular import (`data_loader` -> `providers` -> `tinvest` -> `data_loader`, the last leg
# hitting a not-yet-defined name). Import T-Invest-specific symbols directly from
# `core.providers.tinvest` instead - see `core/__init__.py` for the ordering that avoids this.
from .base import Candle, CandleInterval, Instrument, MarketDataProvider

__all__ = [
    "Candle",
    "CandleInterval",
    "Instrument",
    "MarketDataProvider",
]
