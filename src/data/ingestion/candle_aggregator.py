"""
Candle Aggregator — converts real-time price ticks into OHLCV candles.

Subscribes to Redis price update events and builds candles in memory.
On each timeframe boundary, flushes the completed candle to TimescaleDB
and publishes it to Redis for agent consumption.

Timeframes built: 1m, 5m, 15m, 1h, 4h, 1d
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.config.instruments import ALL_SYMBOLS, ALL_TIMEFRAMES
from src.data.storage.cache import get_redis, CHANNEL_PRICE_UPDATE, publish_event
from src.data.storage.timeseries import CandleRecord, upsert_candles
from src.models import PriceTick

logger = logging.getLogger(__name__)

# Timeframe → duration in seconds
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class CandleBuilder:
    """Builds a single candle from ticks."""

    __slots__ = ("instrument", "timeframe", "period_start", "open", "high", "low", "close", "volume", "tick_count")

    def __init__(self, instrument: str, timeframe: str, period_start: datetime) -> None:
        self.instrument = instrument
        self.timeframe = timeframe
        self.period_start = period_start
        self.open: Decimal | None = None
        self.high: Decimal | None = None
        self.low: Decimal | None = None
        self.close: Decimal | None = None
        self.volume: Decimal = Decimal("0")
        self.tick_count: int = 0

    def update(self, mid_price: Decimal, volume: Decimal = Decimal("0")) -> None:
        """Ingest a tick into this candle."""
        if self.open is None:
            self.open = mid_price
            self.high = mid_price
            self.low = mid_price

        self.high = max(self.high, mid_price)  # type: ignore[arg-type]
        self.low = min(self.low, mid_price)  # type: ignore[arg-type]
        self.close = mid_price
        self.volume += volume
        self.tick_count += 1

    @property
    def is_empty(self) -> bool:
        return self.open is None

    def to_record(self) -> CandleRecord:
        assert self.open is not None, "Cannot convert empty candle to record"
        return CandleRecord(
            instrument=self.instrument,
            timeframe=self.timeframe,
            ts=self.period_start,
            open=self.open,
            high=self.high,  # type: ignore[arg-type]
            low=self.low,  # type: ignore[arg-type]
            close=self.close,  # type: ignore[arg-type]
            volume=self.volume,
        )


def align_to_period(ts: datetime, seconds: int) -> datetime:
    """Align a timestamp down to the nearest period boundary."""
    epoch = ts.timestamp()
    aligned = (int(epoch) // seconds) * seconds
    return datetime.fromtimestamp(aligned, tz=timezone.utc)


class CandleAggregator:
    """
    Manages candle builders for all instrument × timeframe combinations.
    Listens to Redis pub/sub for price updates.
    """

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = symbols or ALL_SYMBOLS
        self._builders: dict[str, CandleBuilder] = {}  # "EUR/USD:5m" → builder
        self._running = False
        self._flush_count = 0

    def _key(self, instrument: str, timeframe: str) -> str:
        return f"{instrument}:{timeframe}"

    def _get_or_create_builder(
        self, instrument: str, timeframe: str, ts: datetime
    ) -> CandleBuilder:
        key = self._key(instrument, timeframe)
        seconds = TIMEFRAME_SECONDS[timeframe]
        period_start = align_to_period(ts, seconds)

        existing = self._builders.get(key)
        if existing is not None and existing.period_start == period_start:
            return existing

        # Period rolled over — flush the old candle first
        if existing is not None and not existing.is_empty:
            asyncio.create_task(self._flush_candle(existing))

        builder = CandleBuilder(instrument, timeframe, period_start)
        self._builders[key] = builder
        return builder

    async def _flush_candle(self, builder: CandleBuilder) -> None:
        """Write a completed candle to TimescaleDB and publish to Redis."""
        if builder.is_empty:
            return

        record = builder.to_record()
        await upsert_candles([record])
        self._flush_count += 1

        # Publish completed candle event for agents
        await publish_event(
            f"candle:{builder.instrument}:{builder.timeframe}",
            {
                "instrument": builder.instrument,
                "timeframe": builder.timeframe,
                "ts": record.ts.isoformat(),
                "open": str(record.open),
                "high": str(record.high),
                "low": str(record.low),
                "close": str(record.close),
                "volume": str(record.volume),
            },
        )

        logger.debug(
            "Flushed candle",
            extra={
                "instrument": builder.instrument,
                "timeframe": builder.timeframe,
                "ts": record.ts.isoformat(),
                "ticks": builder.tick_count,
            },
        )

    async def handle_tick(self, price_data: dict[str, Any]) -> None:
        """Process a price tick from Redis pub/sub."""
        instrument = price_data.get("instrument", "")
        if instrument not in self.symbols:
            return

        mid = Decimal(price_data.get("mid", "0"))
        ts_str = price_data.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        # Update all timeframe builders for this instrument
        for tf in ALL_TIMEFRAMES:
            builder = self._get_or_create_builder(instrument, tf, ts)
            builder.update(mid)

    async def start(self) -> None:
        """Subscribe to Redis price updates and aggregate into candles."""
        self._running = True
        r = await get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(CHANNEL_PRICE_UPDATE)

        logger.info(
            "Candle aggregator started",
            extra={"symbols": len(self.symbols), "timeframes": ALL_TIMEFRAMES},
        )

        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    try:
                        data = __import__("json").loads(message["data"])
                        await self.handle_tick(data)
                    except Exception as e:
                        logger.error("Tick processing error", extra={"error": str(e)})
        finally:
            await pubsub.unsubscribe(CHANNEL_PRICE_UPDATE)
            await pubsub.aclose()

    async def stop(self) -> None:
        """Stop aggregator and flush all remaining candles."""
        self._running = False
        # Flush any in-progress candles
        for builder in self._builders.values():
            if not builder.is_empty:
                await self._flush_candle(builder)
        logger.info("Candle aggregator stopped", extra={"flushed": self._flush_count})

    @property
    def stats(self) -> dict[str, Any]:
        active = sum(1 for b in self._builders.values() if not b.is_empty)
        return {
            "running": self._running,
            "active_builders": active,
            "total_builders": len(self._builders),
            "flush_count": self._flush_count,
        }
