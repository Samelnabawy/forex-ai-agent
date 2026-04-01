"""
MarketState Builder.

Assembles the MarketState snapshot that every agent consumes.
Pulls from:
  - TimescaleDB → candles (all instruments × all timeframes)
  - Redis → latest prices, spreads
  - Redis → current agent states (macro regime, etc.)
  - PostgreSQL → open trades, daily/weekly P&L

This is the single source of truth for "what does the market look like right now?"
Called before each agent execution cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.config.instruments import ALL_SYMBOLS, ALL_TIMEFRAMES, EXECUTION_TIMEFRAMES, CONTEXT_TIMEFRAMES
from src.data.storage.cache import get_all_prices, get_agent_state, get_latest_price
from src.data.storage.timeseries import get_candles, CandleRecord
from src.models import Candle, MacroRegime, MarketState, PriceTick

logger = logging.getLogger(__name__)

# How many candles to fetch per timeframe (enough for indicator computation)
CANDLE_LOOKBACK: dict[str, int] = {
    "1m": 60,     # 1 hour of 1m candles
    "5m": 300,    # ~25 hours
    "15m": 300,   # ~3 days
    "1h": 300,    # ~12.5 days
    "4h": 300,    # ~50 days
    "1d": 300,    # ~300 days
}


def _record_to_candle(record: CandleRecord) -> Candle:
    """Convert a storage CandleRecord to a Pydantic Candle model."""
    return Candle(
        instrument=record.instrument,
        timeframe=record.timeframe,
        ts=record.ts,
        open=record.open,
        high=record.high,
        low=record.low,
        close=record.close,
        volume=record.volume,
    )


async def build_market_state(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> MarketState:
    """
    Build a complete MarketState snapshot.
    Called before each agent analysis cycle.

    Args:
        symbols: instruments to include (default: all 9)
        timeframes: timeframes to fetch (default: execution + context)
    """
    symbols = symbols or ALL_SYMBOLS
    timeframes = timeframes or (EXECUTION_TIMEFRAMES + CONTEXT_TIMEFRAMES)
    now = datetime.now(timezone.utc)

    # 1. Fetch latest prices from Redis
    prices: dict[str, PriceTick] = {}
    try:
        raw_prices = await get_all_prices()
        for symbol, data in raw_prices.items():
            if symbol in symbols:
                try:
                    prices[symbol] = PriceTick(
                        instrument=symbol,
                        bid=Decimal(data.get("bid", "0")),
                        ask=Decimal(data.get("ask", "0")),
                        ts=datetime.fromisoformat(data["ts"]) if "ts" in data else now,
                    )
                except Exception as e:
                    logger.debug("Failed to parse price for %s: %s", symbol, e)
    except Exception as e:
        logger.error("Failed to fetch prices from Redis", extra={"error": str(e)})

    # 2. Fetch candles from TimescaleDB
    candles: dict[str, dict[str, list[Candle]]] = {}
    for symbol in symbols:
        candles[symbol] = {}
        for tf in timeframes:
            lookback = CANDLE_LOOKBACK.get(tf, 200)
            tf_seconds = {
                "1m": 60, "5m": 300, "15m": 900,
                "1h": 3600, "4h": 14400, "1d": 86400,
            }
            seconds = tf_seconds.get(tf, 3600)
            start = now - timedelta(seconds=seconds * lookback)

            try:
                records = await get_candles(
                    instrument=symbol,
                    timeframe=tf,
                    start=start,
                    end=now,
                    limit=lookback,
                )
                candles[symbol][tf] = [_record_to_candle(r) for r in records]
            except Exception as e:
                logger.debug("No candles for %s:%s — %s", symbol, tf, e)
                candles[symbol][tf] = []

    # 3. Get macro regime from Macro Agent's last output
    macro_regime = MacroRegime.NEUTRAL
    try:
        macro_state = await get_agent_state("macro_analyst")
        if macro_state:
            regime_str = macro_state.get("macro_regime", "neutral")
            regime_map = {
                "risk_on": MacroRegime.RISK_ON,
                "risk_off": MacroRegime.RISK_OFF,
                "neutral": MacroRegime.NEUTRAL,
                "transitioning": MacroRegime.TRANSITIONING,
            }
            macro_regime = regime_map.get(regime_str, MacroRegime.NEUTRAL)
    except Exception as e:
        logger.debug("Failed to fetch macro regime: %s", e)

    # 4. Get daily/weekly P&L from risk state
    daily_pnl = Decimal("0")
    weekly_pnl = Decimal("0")
    try:
        from src.data.storage.database import get_session
        import sqlalchemy as sa

        risk_table = sa.table(
            "risk_state",
            sa.column("daily_pnl_pct", sa.Numeric),
            sa.column("weekly_pnl_pct", sa.Numeric),
        )
        async with get_session() as session:
            result = await session.execute(sa.select(risk_table).where(sa.text("id = 1")))
            row = result.fetchone()
            if row:
                daily_pnl = Decimal(str(row.daily_pnl_pct))
                weekly_pnl = Decimal(str(row.weekly_pnl_pct))
    except Exception as e:
        logger.debug("Failed to fetch risk state: %s", e)

    state = MarketState(
        ts=now,
        prices=prices,
        candles=candles,
        macro_regime=macro_regime,
        daily_pnl_pct=daily_pnl,
        weekly_pnl_pct=weekly_pnl,
    )

    logger.debug(
        "MarketState built",
        extra={
            "instruments_with_prices": len(prices),
            "instruments_with_candles": sum(
                1 for s in candles.values() if any(len(c) > 0 for c in s.values())
            ),
            "macro_regime": macro_regime.value,
        },
    )

    return state
