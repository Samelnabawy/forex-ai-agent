"""
Real-time price feed ingestion via WebSocket.

Supports multiple data providers via the PriceProvider protocol.
Handles reconnection, heartbeat, and fan-out to Redis + TimescaleDB.

Architecture:
  WebSocket → PriceProvider.parse() → PriceTick
    → Redis (latest price, pub/sub)
    → CandleAggregator → TimescaleDB (OHLCV candles)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncGenerator

import aiohttp

from src.config.instruments import ALL_SYMBOLS, INSTRUMENTS
from src.config.settings import get_settings
from src.data.storage.cache import (
    CHANNEL_PRICE_UPDATE,
    publish_event,
    set_current_spread,
    set_latest_price,
)
from src.models import PriceTick

logger = logging.getLogger(__name__)


# ── Provider Protocol ─────────────────────────────────────────

class PriceProvider(ABC):
    """Abstract base for price data providers."""

    name: str

    @abstractmethod
    def websocket_url(self) -> str:
        ...

    @abstractmethod
    def subscribe_message(self, symbols: list[str]) -> dict[str, Any]:
        ...

    @abstractmethod
    def parse_message(self, raw: str | bytes) -> list[PriceTick]:
        """Parse a raw WebSocket message into PriceTick(s)."""
        ...

    @abstractmethod
    def is_heartbeat(self, raw: str | bytes) -> bool:
        ...


class PolygonProvider(PriceProvider):
    """Polygon.io forex WebSocket feed."""

    name = "polygon"

    def __init__(self) -> None:
        self._api_key = get_settings().polygon_api_key.get_secret_value()

    def websocket_url(self) -> str:
        return f"wss://socket.polygon.io/forex"

    def subscribe_message(self, symbols: list[str]) -> dict[str, Any]:
        # Polygon uses C.EURUSD format
        polygon_symbols = []
        for sym in symbols:
            inst = INSTRUMENTS.get(sym)
            if inst and inst.polygon_symbol:
                polygon_symbols.append(inst.polygon_symbol)
        return {
            "action": "subscribe",
            "params": ",".join(f"CA.{s}" for s in polygon_symbols),
        }

    def parse_message(self, raw: str | bytes) -> list[PriceTick]:
        data = json.loads(raw)
        ticks: list[PriceTick] = []

        if not isinstance(data, list):
            data = [data]

        for msg in data:
            if msg.get("ev") != "CA":  # CA = forex aggregate
                continue

            # Map Polygon symbol back to our format
            pair = msg.get("pair", "")
            symbol = self._polygon_to_symbol(pair)
            if not symbol:
                continue

            ticks.append(
                PriceTick(
                    instrument=symbol,
                    bid=Decimal(str(msg.get("bp", 0))),
                    ask=Decimal(str(msg.get("ap", 0))),
                    ts=datetime.fromtimestamp(msg.get("t", 0) / 1000, tz=timezone.utc),
                )
            )

        return ticks

    def is_heartbeat(self, raw: str | bytes) -> bool:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data[0].get("ev") == "status"
            return False
        except Exception:
            return False

    @staticmethod
    def _polygon_to_symbol(pair: str) -> str | None:
        """Convert 'EURUSD' → 'EUR/USD'."""
        for sym, inst in INSTRUMENTS.items():
            clean = inst.polygon_symbol.replace("C:", "")
            if pair == clean:
                return sym
        return None


class TwelveDataProvider(PriceProvider):
    """Twelve Data WebSocket feed (free tier: 8 connections)."""

    name = "twelve_data"

    def __init__(self) -> None:
        self._api_key = get_settings().twelve_data_api_key.get_secret_value()

    def websocket_url(self) -> str:
        return "wss://ws.twelvedata.com/v1/quotes/price"

    def subscribe_message(self, symbols: list[str]) -> dict[str, Any]:
        td_symbols = []
        for sym in symbols:
            inst = INSTRUMENTS.get(sym)
            if inst and inst.twelve_data_symbol:
                td_symbols.append(inst.twelve_data_symbol)
        return {
            "action": "subscribe",
            "params": {
                "symbols": ",".join(td_symbols),
                "apikey": self._api_key,
            },
        }

    def parse_message(self, raw: str | bytes) -> list[PriceTick]:
        data = json.loads(raw)

        if data.get("event") != "price":
            return []

        symbol = self._td_to_symbol(data.get("symbol", ""))
        if not symbol:
            return []

        price = Decimal(str(data.get("price", 0)))
        # Twelve Data sends mid-price; approximate bid/ask
        spread = INSTRUMENTS[symbol].avg_spread_pips * INSTRUMENTS[symbol].pip_size
        half_spread = spread / 2

        return [
            PriceTick(
                instrument=symbol,
                bid=price - half_spread,
                ask=price + half_spread,
                ts=datetime.fromtimestamp(data.get("timestamp", 0), tz=timezone.utc),
            )
        ]

    def is_heartbeat(self, raw: str | bytes) -> bool:
        try:
            data = json.loads(raw)
            return data.get("event") == "heartbeat"
        except Exception:
            return False

    @staticmethod
    def _td_to_symbol(td_sym: str) -> str | None:
        for sym, inst in INSTRUMENTS.items():
            if inst.twelve_data_symbol == td_sym:
                return sym
        return None


# ── Feed Manager ──────────────────────────────────────────────

class PriceFeedManager:
    """
    Manages WebSocket connection to a price provider.
    Handles reconnection, fan-out to Redis, and candle aggregation.
    """

    MAX_RECONNECT_DELAY = 60  # seconds
    INITIAL_RECONNECT_DELAY = 1

    def __init__(
        self,
        provider: PriceProvider,
        symbols: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.symbols = symbols or ALL_SYMBOLS
        self._running = False
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
        self._tick_count = 0
        self._last_tick_ts: datetime | None = None

    async def start(self) -> None:
        """Start the WebSocket feed with automatic reconnection."""
        self._running = True
        logger.info(
            "Starting price feed",
            extra={"provider": self.provider.name, "symbols": self.symbols},
        )

        while self._running:
            try:
                await self._connect_and_consume()
            except Exception as e:
                if not self._running:
                    break
                logger.error(
                    "Price feed connection lost, reconnecting",
                    extra={
                        "provider": self.provider.name,
                        "error": str(e),
                        "delay": self._reconnect_delay,
                    },
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY
                )

    async def stop(self) -> None:
        self._running = False
        logger.info("Price feed stopping", extra={"provider": self.provider.name})

    async def _connect_and_consume(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.provider.websocket_url(),
                heartbeat=30,
                timeout=aiohttp.ClientTimeout(total=None),
            ) as ws:
                logger.info("WebSocket connected", extra={"provider": self.provider.name})
                self._reconnect_delay = self.INITIAL_RECONNECT_DELAY

                # Subscribe to symbols
                sub_msg = self.provider.subscribe_message(self.symbols)
                await ws.send_json(sub_msg)
                logger.info(
                    "Subscribed to symbols",
                    extra={"count": len(self.symbols)},
                )

                # Consume messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error("WebSocket error: %s", ws.exception())
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket closed by server")
                        break

    async def _handle_message(self, raw: str) -> None:
        """Parse and fan-out a WebSocket message."""
        if self.provider.is_heartbeat(raw):
            return

        ticks = self.provider.parse_message(raw)

        for tick in ticks:
            self._tick_count += 1
            self._last_tick_ts = tick.ts

            # Fan-out: Redis cache + pub/sub
            price_data = {
                "instrument": tick.instrument,
                "bid": str(tick.bid),
                "ask": str(tick.ask),
                "mid": str((tick.bid + tick.ask) / 2),
                "spread": str(tick.spread),
                "ts": tick.ts.isoformat(),
            }
            await set_latest_price(tick.instrument, price_data)
            await set_current_spread(tick.instrument, float(tick.spread))
            await publish_event(CHANNEL_PRICE_UPDATE, price_data)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "running": self._running,
            "tick_count": self._tick_count,
            "last_tick": self._last_tick_ts.isoformat() if self._last_tick_ts else None,
        }
