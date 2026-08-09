"""
Тонкий REST-клиент к T-Invest API (Т-Банк Инвестиции).

Официальный gRPC SDK (пакет ``tinkoff-investments``) на момент написания
недоступен на PyPI, поэтому используется публичный REST/JSON-шлюз
(grpc-gateway) T-Invest API: https://invest-public-api.tinkoff.ru/rest/...

Домен ``tinkoff.ru`` отдаёт сертификат, выпущенный корневым
удостоверяющим центром Минцифры России ("Russian Trusted Root CA"),
который по умолчанию не входит в системные доверенные хранilища за
пределами РФ. Поэтому клиент проверяет TLS-цепочку по объединённому
бандлу: стандартный certifi + сертификаты Минцифры (``certs/russian_trusted_ca.pem``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import certifi
import requests

_BASE_URL = "https://invest-public-api.tinkoff.ru/rest"
_CERTS_DIR = Path(__file__).parent / "certs"
_RU_CA_BUNDLE = _CERTS_DIR / "russian_trusted_ca.pem"


def _build_ca_bundle() -> str:
    """Возвращает путь к CA-бандлу: certifi + Russian Trusted CA (если есть)."""
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
    """Ошибка ответа T-Invest API (HTTP-статус или gRPC-status в теле)."""


class TInvestClient:
    """
    Минимальный синхронный REST-клиент T-Invest API.

    Реализует только вызовы, необходимые для получения исторических
    данных и метаданных инструментов (без выставления заявок).
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

    def call(self, service: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Вызов произвольного метода T-Invest gRPC-gateway по HTTP/JSON."""
        url = f"{_BASE_URL}/tinkoff.public.invest.api.contract.v1.{service}/{method}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._session.post(
                    url, json=payload or {}, timeout=self._timeout, verify=self._verify
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
                continue

            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise TInvestAPIError(
                f"{service}/{method} failed: HTTP {response.status_code}: {response.text[:500]}"
            )
        raise TInvestAPIError(f"{service}/{method} failed after {self._max_retries} retries: {last_error}")

    def get_accounts(self) -> list[dict[str, Any]]:
        return self.call("UsersService", "GetAccounts").get("accounts", [])

    def find_share_by_ticker(self, ticker: str, class_code: str = "TQBR") -> dict[str, Any]:
        """Ищет акцию по тикеру среди базовых инструментов (по умолчанию режим торгов MOEX TQBR)."""
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
        """Одна страница свечей (с учётом ограничений API по длине интервала на запрос)."""
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
