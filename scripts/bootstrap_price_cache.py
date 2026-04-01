"""
Bootstrap Redis price cache from latest daily candle closes in TimescaleDB.
Run once to seed the real-time price cache when WebSocket feeds haven't
populated it yet.

Usage:
  python -m scripts.bootstrap_price_cache
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

from src.config.instruments import ALL_SYMBOLS, INSTRUMENTS
from src.config.settings import get_settings
from src.data.storage.database import init_db, close_db, get_session
from src.data.storage.cache import get_redis, set_latest_price
from src.logging_config import setup_logging

import sqlalchemy as sa

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_logging()
    await init_db()
    r = await get_redis()

    logger.info("Bootstrapping Redis price cache from DB...")

    count = 0
    async with get_session() as session:
        for symbol in ALL_SYMBOLS:
            result = await session.execute(
                sa.text("""
                    SELECT ts, open, high, low, close
                    FROM candles
                    WHERE instrument = :instrument AND timeframe = '1d'
                    ORDER BY ts DESC
                    LIMIT 1
                """),
                {"instrument": symbol},
            )
            row = result.fetchone()
            if not row:
                logger.warning("No candles for %s", symbol)
                continue

            ts, open_, high, low, close = row
            mid = float(close)
            inst = INSTRUMENTS.get(symbol)
            spread = float(inst.avg_spread_pips * inst.pip_size) if inst else 0.0002
            half_spread = spread / 2

            price_data = {
                "instrument": symbol,
                "bid": str(round(mid - half_spread, 6)),
                "ask": str(round(mid + half_spread, 6)),
                "mid": str(round(mid, 6)),
                "spread": str(round(spread, 6)),
                "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "source": "db_bootstrap",
            }

            await set_latest_price(symbol, price_data)
            count += 1
            logger.info("Cached %s: mid=%s", symbol, price_data["mid"])

    logger.info("Bootstrapped %d/%d instruments into Redis", count, len(ALL_SYMBOLS))
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
