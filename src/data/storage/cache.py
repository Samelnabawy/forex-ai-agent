"""
Redis cache and pub/sub for real-time data pipeline.

Key schema:
  price:{instrument}          → latest price JSON
  spread:{instrument}         → current spread
  agent:state:{agent_name}    → last agent output
  queue:signals               → BullMQ-style signal queue
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

import redis.asyncio as aioredis

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Get or create Redis connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
    return _pool


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
    logger.info("Redis connection closed")


# ── Price Cache ───────────────────────────────────────────────

async def set_latest_price(instrument: str, price_data: dict[str, Any]) -> None:
    """Cache latest price for an instrument. TTL = 60s (stale data protection)."""
    r = await get_redis()
    key = f"price:{instrument}"
    await r.set(key, json.dumps(price_data, default=str), ex=60)


async def get_latest_price(instrument: str) -> dict[str, Any] | None:
    """Retrieve latest cached price."""
    r = await get_redis()
    raw = await r.get(f"price:{instrument}")
    return json.loads(raw) if raw else None


async def get_all_prices() -> dict[str, dict[str, Any]]:
    """Retrieve latest prices for all instruments. Used by Correlation Agent."""
    r = await get_redis()
    pipe = r.pipeline()
    from src.config.instruments import ALL_SYMBOLS
    for symbol in ALL_SYMBOLS:
        pipe.get(f"price:{symbol}")
    results = await pipe.execute()

    prices = {}
    for symbol, raw in zip(ALL_SYMBOLS, results, strict=True):
        if raw:
            prices[symbol] = json.loads(raw)
    return prices


# ── Spread Cache ──────────────────────────────────────────────

async def set_current_spread(instrument: str, spread: float) -> None:
    r = await get_redis()
    await r.set(f"spread:{instrument}", str(spread), ex=60)


async def get_current_spread(instrument: str) -> float | None:
    r = await get_redis()
    raw = await r.get(f"spread:{instrument}")
    return float(raw) if raw else None


# ── Agent State Cache ─────────────────────────────────────────

async def set_agent_state(agent_name: str, state: dict[str, Any]) -> None:
    """Cache the latest output from an agent. TTL = 1 hour."""
    r = await get_redis()
    key = f"agent:state:{agent_name}"
    await r.set(key, json.dumps(state, default=str), ex=3600)


async def get_agent_state(agent_name: str) -> dict[str, Any] | None:
    r = await get_redis()
    raw = await r.get(f"agent:state:{agent_name}")
    return json.loads(raw) if raw else None


# ── Signal Queue (lightweight BullMQ-style) ───────────────────

SIGNAL_QUEUE = "queue:signals"


async def enqueue_signal(signal: dict[str, Any]) -> None:
    """Push a signal to the processing queue."""
    r = await get_redis()
    await r.rpush(SIGNAL_QUEUE, json.dumps(signal, default=str))
    logger.debug("Enqueued signal: %s %s", signal.get("instrument"), signal.get("signal"))


async def dequeue_signal(timeout: int = 5) -> dict[str, Any] | None:
    """Pop next signal from the queue. Blocks up to `timeout` seconds."""
    r = await get_redis()
    result = await r.blpop(SIGNAL_QUEUE, timeout=timeout)
    if result:
        _, raw = result
        return json.loads(raw)
    return None


async def get_queue_length() -> int:
    r = await get_redis()
    return await r.llen(SIGNAL_QUEUE)


# ── Pub/Sub for real-time events ──────────────────────────────

CHANNEL_PRICE_UPDATE = "events:price_update"
CHANNEL_NEWS_ALERT = "events:news_alert"
CHANNEL_SIGNAL = "events:signal"


async def publish_event(channel: str, data: dict[str, Any]) -> None:
    r = await get_redis()
    await r.publish(channel, json.dumps(data, default=str))


async def init_redis() -> None:
    """Verify Redis connectivity on startup."""
    r = await get_redis()
    pong = await r.ping()
    assert pong is True
    logger.info("Redis connection verified")
