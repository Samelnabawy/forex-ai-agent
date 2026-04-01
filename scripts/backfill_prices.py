"""
Historical price data backfill script.

Fetches 5+ years of OHLCV data for all 9 instruments across all timeframes.
Uses Polygon.io REST API (free tier: 5 API calls/min, 2 years history).
For full 5-year backfill, use a paid Polygon plan or Twelve Data.

Features:
  - Resume-aware: checks last stored timestamp and continues from there
  - Rate-limited: respects API limits with async semaphore
  - Batch inserts: 1000 candles per DB write for efficiency
  - Progress tracking: logs per-instrument, per-timeframe progress

Usage:
  python -m scripts.backfill_prices --instruments EUR/USD GBP/USD --timeframes 1h 4h
  python -m scripts.backfill_prices --all  # all instruments, all timeframes
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import httpx

from src.config.instruments import ALL_SYMBOLS, ALL_TIMEFRAMES, INSTRUMENTS
from src.config.settings import get_settings
from src.data.storage.database import init_db, close_db
from src.data.storage.timeseries import CandleRecord, upsert_candles, get_latest_timestamp
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Polygon timeframe mapping
POLYGON_TIMEFRAMES: dict[str, tuple[int, str]] = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "1h": (1, "hour"),
    "4h": (4, "hour"),
    "1d": (1, "day"),
}

# How far back to fetch per timeframe (from spec Section 1.6)
BACKFILL_YEARS: dict[str, int] = {
    "1m": 1,    # 1 year of 1m data (storage intensive)
    "5m": 5,    # 5 years
    "15m": 5,   # 5 years
    "1h": 10,   # 10 years
    "4h": 10,   # 10 years
    "1d": 20,   # 20 years
}

BATCH_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 2  # Polygon free tier is limited


class HistoricalBackfill:
    """Manages historical data backfill from Polygon.io."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.polygon_api_key.get_secret_value()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._client: httpx.AsyncClient | None = None
        self._total_candles = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _fetch_polygon_aggs(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_ts: datetime,
        to_ts: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch aggregate bars from Polygon.io REST API."""
        inst = INSTRUMENTS[symbol]
        polygon_ticker = inst.polygon_symbol  # e.g. "C:EURUSD"

        from_str = from_ts.strftime("%Y-%m-%d")
        to_str = to_ts.strftime("%Y-%m-%d")

        async with self._semaphore:
            client = await self._get_client()
            try:
                resp = await client.get(
                    f"https://api.polygon.io/v2/aggs/ticker/{polygon_ticker}/range/{multiplier}/{timespan}/{from_str}/{to_str}",
                    params={
                        "adjusted": "true",
                        "sort": "asc",
                        "limit": 50000,
                        "apiKey": self._api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") != "OK":
                    logger.warning(
                        "Polygon non-OK response",
                        extra={"status": data.get("status"), "symbol": symbol},
                    )
                    return []

                return data.get("results", [])

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limited by Polygon, waiting 60s")
                    await asyncio.sleep(60)
                    return await self._fetch_polygon_aggs(
                        symbol, multiplier, timespan, from_ts, to_ts
                    )
                raise
            except Exception as e:
                logger.error(
                    "Polygon fetch failed",
                    extra={"symbol": symbol, "error": str(e)},
                )
                return []

    async def backfill_instrument_timeframe(
        self,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Backfill one instrument × timeframe. Returns candle count."""
        multiplier, timespan = POLYGON_TIMEFRAMES[timeframe]
        years = BACKFILL_YEARS[timeframe]

        # Check where we left off
        last_ts = await get_latest_timestamp(symbol, timeframe)
        if last_ts:
            start = last_ts + timedelta(minutes=1)
            logger.info(
                "Resuming backfill",
                extra={"symbol": symbol, "timeframe": timeframe, "from": start.isoformat()},
            )
        else:
            start = datetime.now(timezone.utc) - timedelta(days=years * 365)

        end = datetime.now(timezone.utc)
        total_inserted = 0

        # Fetch in 30-day chunks to stay within API limits
        chunk_days = 30 if timeframe in ("1m", "5m") else 180
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)

            raw_bars = await self._fetch_polygon_aggs(
                symbol, multiplier, timespan, current, chunk_end
            )

            if raw_bars:
                candles = [
                    CandleRecord(
                        instrument=symbol,
                        timeframe=timeframe,
                        ts=datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc),
                        open=Decimal(str(bar["o"])),
                        high=Decimal(str(bar["h"])),
                        low=Decimal(str(bar["l"])),
                        close=Decimal(str(bar["c"])),
                        volume=Decimal(str(bar.get("v", 0))),
                    )
                    for bar in raw_bars
                ]

                # Batch insert
                for i in range(0, len(candles), BATCH_SIZE):
                    batch = candles[i : i + BATCH_SIZE]
                    await upsert_candles(batch)
                    total_inserted += len(batch)

                logger.info(
                    "Backfill progress",
                    extra={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "chunk_end": chunk_end.isoformat(),
                        "bars_in_chunk": len(raw_bars),
                        "total": total_inserted,
                    },
                )

            current = chunk_end + timedelta(minutes=1)

            # Rate limit courtesy: 200ms between requests
            await asyncio.sleep(0.2)

        self._total_candles += total_inserted
        return total_inserted

    async def backfill_all(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Backfill all requested instrument × timeframe combinations.
        Returns {symbol:timeframe → candle_count}.
        """
        symbols = symbols or ALL_SYMBOLS
        timeframes = timeframes or ALL_TIMEFRAMES

        results: dict[str, int] = {}

        # Process higher timeframes first (less data, quick validation)
        sorted_tfs = sorted(timeframes, key=lambda tf: BACKFILL_YEARS.get(tf, 0))

        for tf in sorted_tfs:
            for symbol in symbols:
                key = f"{symbol}:{tf}"
                logger.info("Starting backfill", extra={"key": key})
                count = await self.backfill_instrument_timeframe(symbol, tf)
                results[key] = count
                logger.info("Completed backfill", extra={"key": key, "candles": count})

        return results

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


async def main() -> None:
    """CLI entry point for backfill."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill historical price data")
    parser.add_argument("--instruments", nargs="+", default=None, help="Instruments to backfill")
    parser.add_argument("--timeframes", nargs="+", default=None, help="Timeframes to backfill")
    parser.add_argument("--all", action="store_true", help="Backfill everything")
    args = parser.parse_args()

    setup_logging()

    symbols = args.instruments if not args.all else ALL_SYMBOLS
    timeframes = args.timeframes if not args.all else ALL_TIMEFRAMES

    if not symbols:
        symbols = ALL_SYMBOLS
    if not timeframes:
        timeframes = ALL_TIMEFRAMES

    logger.info(
        "Starting historical backfill",
        extra={"symbols": symbols, "timeframes": timeframes},
    )

    await init_db()

    backfill = HistoricalBackfill()
    try:
        results = await backfill.backfill_all(symbols, timeframes)
        total = sum(results.values())
        logger.info(
            "Backfill complete",
            extra={"total_candles": total, "breakdown": results},
        )
    finally:
        await backfill.close()
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
