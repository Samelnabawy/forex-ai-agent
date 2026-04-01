"""
Supplementary market data feed — instruments we track but don't trade.

Fetches:
  - DXY (US Dollar Index) — anchor for all USD pairs
  - VIX (Volatility Index) — risk sentiment barometer
  - Iron Ore — AUD driver
  - Copper — global growth proxy
  - Natural Gas — energy sector context

These feed into Correlation Agent and Macro Agent as context,
but we don't generate trading signals on them.

Source: Polygon/Massive API (same provider as price feed)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import httpx

from src.config.settings import get_settings
from src.data.storage.cache import set_latest_price

logger = logging.getLogger(__name__)

# Supplementary instruments (not traded, used for context)
SUPPLEMENTARY_INSTRUMENTS: dict[str, dict[str, str]] = {
    "DXY": {
        "polygon_symbol": "I:DXY",
        "description": "US Dollar Index",
        "category": "index",
    },
    "VIX": {
        "polygon_symbol": "I:VIX",
        "description": "CBOE Volatility Index",
        "category": "index",
    },
    "US500": {
        "polygon_symbol": "I:SPX",
        "description": "S&P 500 Index",
        "category": "index",
    },
}


class MarketDataFeed:
    """Fetches supplementary market data for context."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.polygon_api_key.get_secret_value()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def fetch_snapshot(self, ticker: str) -> dict[str, Any] | None:
        """Fetch latest price snapshot from Polygon."""
        if not self._api_key:
            return None

        client = await self._get_client()
        try:
            resp = await client.get(
                f"https://api.polygon.io/v2/snapshot/locale/global/markets/forex/tickers/{ticker}",
                params={"apiKey": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            ticker_data = data.get("ticker", {})
            if not ticker_data:
                return None

            return {
                "bid": ticker_data.get("lastQuote", {}).get("bid"),
                "ask": ticker_data.get("lastQuote", {}).get("ask"),
                "last": ticker_data.get("lastTrade", {}).get("price"),
                "prev_close": ticker_data.get("prevDay", {}).get("c"),
                "change_pct": ticker_data.get("todaysChangePerc"),
            }
        except Exception as e:
            logger.error("Market data fetch failed", extra={"ticker": ticker, "error": str(e)})
            return None

    async def fetch_previous_close(self, ticker: str) -> dict[str, Any] | None:
        """Fetch previous day's OHLCV from Polygon aggs."""
        if not self._api_key:
            return None

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        client = await self._get_client()
        try:
            resp = await client.get(
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{yesterday}/{yesterday}",
                params={"apiKey": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                bar = results[0]
                return {
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                }
        except Exception as e:
            logger.error("Previous close fetch failed", extra={"ticker": ticker, "error": str(e)})
        return None

    async def sync_all(self) -> dict[str, Any]:
        """Fetch all supplementary instruments and cache in Redis."""
        results: dict[str, Any] = {}

        for name, config in SUPPLEMENTARY_INSTRUMENTS.items():
            ticker = config["polygon_symbol"]
            snapshot = await self.fetch_snapshot(ticker)

            if snapshot and snapshot.get("last"):
                price_data = {
                    "instrument": name,
                    "bid": str(snapshot.get("bid", snapshot["last"])),
                    "ask": str(snapshot.get("ask", snapshot["last"])),
                    "mid": str(snapshot["last"]),
                    "spread": "0",
                    "change_pct": str(snapshot.get("change_pct", 0)),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                await set_latest_price(name, price_data)
                results[name] = price_data
                logger.debug("Updated %s: %s", name, snapshot["last"])

        logger.info("Supplementary market data synced", extra={"count": len(results)})
        return results

    async def get_dxy(self) -> float | None:
        """Get current DXY value."""
        from src.data.storage.cache import get_latest_price
        cached = await get_latest_price("DXY")
        if cached:
            return float(cached.get("mid", 0))
        return None

    async def get_vix(self) -> float | None:
        """Get current VIX value."""
        from src.data.storage.cache import get_latest_price
        cached = await get_latest_price("VIX")
        if cached:
            return float(cached.get("mid", 0))
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
