"""
TimescaleDB candle storage and retrieval.
Optimized for bulk inserts (backfill) and range queries (agent reads).
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.storage.database import get_session

logger = logging.getLogger(__name__)

# ── Table reference (raw SQL for performance-critical paths) ──
CANDLES_TABLE = sa.table(
    "candles",
    sa.column("instrument", sa.String),
    sa.column("timeframe", sa.String),
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("open", sa.Numeric),
    sa.column("high", sa.Numeric),
    sa.column("low", sa.Numeric),
    sa.column("close", sa.Numeric),
    sa.column("volume", sa.Numeric),
)


class CandleRecord:
    """Lightweight candle data object — no ORM overhead."""

    __slots__ = ("instrument", "timeframe", "ts", "open", "high", "low", "close", "volume")

    def __init__(
        self,
        instrument: str,
        timeframe: str,
        ts: datetime,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal = Decimal("0"),
    ) -> None:
        self.instrument = instrument
        self.timeframe = timeframe
        self.ts = ts
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


async def upsert_candles(candles: Sequence[CandleRecord]) -> int:
    """
    Bulk upsert candles using ON CONFLICT DO UPDATE.
    Returns the number of rows affected.
    Used by both the real-time ingestion and historical backfill.
    """
    if not candles:
        return 0

    rows = [c.to_dict() for c in candles]

    stmt = pg_insert(CANDLES_TABLE).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["instrument", "timeframe", "ts"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
        },
    )

    async with get_session() as session:
        result = await session.execute(stmt)
        count = result.rowcount  # type: ignore[union-attr]
        logger.debug("Upserted %d candles", count)
        return count


async def get_candles(
    instrument: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    limit: int = 10_000,
) -> list[CandleRecord]:
    """
    Retrieve candles for a given instrument/timeframe/date range.
    Returns in chronological order (oldest first).
    """
    query = (
        sa.select(CANDLES_TABLE)
        .where(
            sa.and_(
                CANDLES_TABLE.c.instrument == instrument,
                CANDLES_TABLE.c.timeframe == timeframe,
                CANDLES_TABLE.c.ts >= start,
                CANDLES_TABLE.c.ts <= end,
            )
        )
        .order_by(CANDLES_TABLE.c.ts.asc())
        .limit(limit)
    )

    async with get_session() as session:
        result = await session.execute(query)
        rows = result.fetchall()

    return [
        CandleRecord(
            instrument=row.instrument,
            timeframe=row.timeframe,
            ts=row.ts,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )
        for row in rows
    ]


async def get_latest_candle(
    instrument: str,
    timeframe: str,
) -> CandleRecord | None:
    """Get the most recent candle for an instrument/timeframe."""
    query = (
        sa.select(CANDLES_TABLE)
        .where(
            sa.and_(
                CANDLES_TABLE.c.instrument == instrument,
                CANDLES_TABLE.c.timeframe == timeframe,
            )
        )
        .order_by(CANDLES_TABLE.c.ts.desc())
        .limit(1)
    )

    async with get_session() as session:
        result = await session.execute(query)
        row = result.fetchone()

    if row is None:
        return None

    return CandleRecord(
        instrument=row.instrument,
        timeframe=row.timeframe,
        ts=row.ts,
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
    )


async def get_latest_timestamp(instrument: str, timeframe: str) -> datetime | None:
    """Get the timestamp of the most recent candle. Used by backfill to resume."""
    query = (
        sa.select(CANDLES_TABLE.c.ts)
        .where(
            sa.and_(
                CANDLES_TABLE.c.instrument == instrument,
                CANDLES_TABLE.c.timeframe == timeframe,
            )
        )
        .order_by(CANDLES_TABLE.c.ts.desc())
        .limit(1)
    )

    async with get_session() as session:
        result = await session.execute(query)
        row = result.fetchone()

    return row.ts if row else None


async def get_candle_count(instrument: str, timeframe: str) -> int:
    """Count candles — useful for monitoring backfill progress."""
    query = (
        sa.select(sa.func.count())
        .select_from(CANDLES_TABLE)
        .where(
            sa.and_(
                CANDLES_TABLE.c.instrument == instrument,
                CANDLES_TABLE.c.timeframe == timeframe,
            )
        )
    )

    async with get_session() as session:
        result = await session.execute(query)
        return result.scalar() or 0
