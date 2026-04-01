"""
Bootstrap Redis price cache from latest daily candle closes in TimescaleDB.

Sets a 1-hour TTL so agents have price data to work with until
Twelve Data WebSocket takes over with real-time updates.

Usage:
  python -m scripts.bootstrap_prices
"""

from __future__ import annotations

import asyncio
import json
import logging

import sqlalchemy as sa

from src.config.instruments import ALL_SYMBOLS, INSTRUMENTS
from src.data.storage.cache import get_redis
from src.data.storage.database import close_db, get_session, init_db
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

BOOTSTRAP_TTL = 3600  # 1 hour


async def main() -> None:
    setup_logging()
    await init_db()
    r = await get_redis()

    count = 0
    async with get_session() as session:
        for symbol in ALL_SYMBOLS:
            result = await session.execute(
                sa.text("""
                    SELECT ts, close
                    FROM candles
                    WHERE instrument = :instrument AND timeframe = '1d'
                    ORDER BY ts DESC
                    LIMIT 1
                """),
                {"instrument": symbol},
            )
            row = result.fetchone()
            if not row:
                print(f"  {symbol:10} — no candles in DB, skipped")
                continue

            ts, close = row
            mid = float(close)
            inst = INSTRUMENTS.get(symbol)
            spread = float(inst.avg_spread_pips * inst.pip_size) if inst else 0.0002
            half = spread / 2

            price_data = {
                "instrument": symbol,
                "bid": str(round(mid - half, 6)),
                "ask": str(round(mid + half, 6)),
                "mid": str(round(mid, 6)),
                "spread": str(round(spread, 6)),
                "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "source": "db_bootstrap",
            }

            key = f"price:{symbol}"
            await r.set(key, json.dumps(price_data, default=str), ex=BOOTSTRAP_TTL)
            count += 1
            print(f"  {symbol:10} mid={price_data['mid']:>12}  spread={price_data['spread']}  TTL=1h")

    print(f"\nBootstrapped {count}/{len(ALL_SYMBOLS)} instruments into Redis (1-hour TTL)")
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
