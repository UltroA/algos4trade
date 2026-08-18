"""
Thin REST client for the T-Invest API (T-Bank Investments).

The official gRPC SDK (the ``tinkoff-investments`` package) is not
available on PyPI at the time of writing, so the public REST/JSON gateway
(grpc-gateway) of the T-Invest API is used instead: https://invest-public-api.tinkoff.ru/rest/...

The ``tinkoff.ru`` domain serves a certificate issued by the Russian
Ministry of Digital Development's root certification authority
("Russian Trusted Root CA"), which is not included in system trusted
stores outside Russia by default. So the client verifies the TLS chain
against a combined bundle: standard certifi + the Ministry of Digital
Development's certificates (``certs/russian_trusted_ca.pem``).
"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import certifi
import requests

_BASE_URL = "https://invest-public-api.tinkoff.ru/rest"
_CERTS_DIR = Path(__file__).parent / "certs"
_RU_CA_BUNDLE = _CERTS_DIR / "russian_trusted_ca.pem"


def _build_ca_bundle() -> str:
    """Returns the path to the CA bundle: certifi + Russian Trusted CA (if present)."""
    if not _RU_CA_BUNDLE.exists():
        return certifi.where()

    combined_path = _CERTS_DIR / "_combined_ca_bundle.pem"
    certifi_mtime = os.path.getmtime(certifi.where())
    needs_rebuild = (
        not combined_path.exists()
        or os.path.getmtime(combined_path) < certifi_mtime
        or os.path.getmtime(combined_path) < os.path.getmtime(_RU_CA_BUNDLE)
    )
    if needs_rebuild:
        with open(combined_path, "wb") as out:
            out.write(Path(certifi.where()).read_bytes())
            out.write(b"\n")
            out.write(_RU_CA_BUNDLE.read_bytes())
    return str(combined_path)


class TInvestAPIError(RuntimeError):
    """T-Invest API response error (HTTP status or gRPC status in the body)."""


class TInvestClient:
    """
    Minimal synchronous T-Invest API REST client.

    Implements only the calls needed to fetch historical data and
    instrument metadata (no order placement).
    """

    def __init__(self, token: str | None = None, timeout: float = 30.0, max_retries: int = 3):
        self._token = token or os.environ.get("T_INVEST_TOKEN")
        if not self._token:
            raise ValueError(
                "T-Invest API token is required: pass token=... or set T_INVEST_TOKEN in .env"
            )
        self._timeout = timeout
        self._max_retries = max_retries
        self._verify = _build_ca_bundle()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            }
        )
        # Round-trip latency of the last N successful calls, in ms - the
        # ground truth for "how long does the real T-Invest API actually
        # take", used by core.market_simulator.LatencyTracker to calibrate
        # simulated delay instead of guessing a constant.
        self._latency_log_ms: deque[float] = deque(maxlen=500)

    def call(self, service: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call an arbitrary T-Invest gRPC-gateway method over HTTP/JSON."""
        url = f"{_BASE_URL}/tinkoff.public.invest.api.contract.v1.{service}/{method}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            t0 = time.perf_counter()
            try:
                response = self._session.post(
                    url, json=payload or {}, timeout=self._timeout, verify=self._verify
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
                continue

            if response.status_code == 200:
                self._latency_log_ms.append((time.perf_counter() - t0) * 1000.0)
                return response.json()
            if response.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise TInvestAPIError(
                f"{service}/{method} failed: HTTP {response.status_code}: {response.text[:500]}"
            )
        raise TInvestAPIError(f"{service}/{method} failed after {self._max_retries} retries: {last_error}")

    def recent_latencies_ms(self) -> list[float]:
        """Round-trip latency (ms) of the last successful calls this client made this session."""
        return list(self._latency_log_ms)

    def get_accounts(self) -> list[dict[str, Any]]:
        return self.call("UsersService", "GetAccounts").get("accounts", [])

    def find_share_by_ticker(self, ticker: str, class_code: str = "TQBR") -> dict[str, Any]:
        """Looks up a share by ticker among base instruments (default trading mode is MOEX TQBR)."""
        result = self.call(
            "InstrumentsService",
            "FindInstrument",
            {"query": ticker, "instrumentKind": "INSTRUMENT_TYPE_SHARE", "apiTradeAvailableFlag": False},
        )
        instruments = result.get("instruments", [])
        for inst in instruments:
            if inst.get("ticker") == ticker and inst.get("classCode") == class_code:
                return inst
        for inst in instruments:
            if inst.get("ticker") == ticker:
                return inst
        raise ValueError(f"Instrument with ticker={ticker!r} not found")

    def get_candles(
        self,
        instrument_id: str,
        from_iso: str,
        to_iso: str,
        interval: str = "CANDLE_INTERVAL_DAY",
    ) -> list[dict[str, Any]]:
        """A single page of candles (subject to the API's limits on interval length per request)."""
        candles: list[dict[str, Any]] = []
        page_token = ""
        while True:
            payload = {
                "instrumentId": instrument_id,
                "from": from_iso,
                "to": to_iso,
                "interval": interval,
                "candleSourceType": "CANDLE_SOURCE_INCLUDE_WEEKEND",
            }
            if page_token:
                payload["pageToken"] = page_token
            result = self.call("MarketDataService", "GetCandles", payload)
            candles.extend(result.get("candles", []))
            page_token = result.get("nextPageToken", "")
            if not page_token:
                break
        return candles
