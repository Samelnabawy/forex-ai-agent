"""
Trade Logger — records every trade decision to PostgreSQL.

Manages:
  - Trade creation (pending → open)
  - Trade updates (price changes, partial closes)
  - Trade closure (closed_win / closed_loss)
  - Risk state updates (daily/weekly P&L, consecutive losses, open trades)
  - Daily performance aggregation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa

from src.agents.evidence import TradeProposal
from src.data.storage.database import get_session

logger = logging.getLogger(__name__)

TRADES_TABLE = sa.table(
    "trades",
    sa.column("trade_id", sa.String),
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("instrument", sa.String),
    sa.column("direction", sa.String),
    sa.column("entry_price", sa.Numeric),
    sa.column("stop_loss", sa.Numeric),
    sa.column("take_profit_1", sa.Numeric),
    sa.column("take_profit_2", sa.Numeric),
    sa.column("position_size", sa.Numeric),
    sa.column("confidence", sa.Numeric),
    sa.column("status", sa.String),
    sa.column("exit_price", sa.Numeric),
    sa.column("exit_ts", sa.DateTime(timezone=True)),
    sa.column("pnl_pips", sa.Numeric),
    sa.column("pnl_percent", sa.Numeric),
    sa.column("agent_reasoning", sa.JSON),
)

RISK_STATE = sa.table(
    "risk_state",
    sa.column("id", sa.Integer),
    sa.column("daily_pnl_pct", sa.Numeric),
    sa.column("weekly_pnl_pct", sa.Numeric),
    sa.column("consecutive_losses", sa.Integer),
    sa.column("open_trade_count", sa.Integer),
    sa.column("open_instruments", sa.ARRAY(sa.String)),
    sa.column("circuit_breaker_until", sa.DateTime(timezone=True)),
    sa.column("daily_halt", sa.Boolean),
    sa.column("weekly_halt", sa.Boolean),
    sa.column("last_updated", sa.DateTime(timezone=True)),
)


async def log_new_trade(
    decision: dict[str, Any],
    proposal: TradeProposal,
) -> None:
    """Record a new trade in the database."""
    now = datetime.now(timezone.utc)

    # Extract reasoning for audit trail
    reasoning = {
        "debate": decision.get("debate", {}),
        "risk": decision.get("risk", {}),
        "matrix": decision.get("matrix", {}),
        "decision_type": decision.get("decision_type", ""),
    }

    async with get_session() as session:
        await session.execute(
            TRADES_TABLE.insert().values(
                trade_id=proposal.trade_id,
                ts=now,
                instrument=proposal.instrument,
                direction=proposal.direction,
                entry_price=proposal.entry,
                stop_loss=proposal.stop_loss,
                take_profit_1=proposal.take_profit_1,
                take_profit_2=proposal.take_profit_2,
                position_size=decision.get("final_position_size", 0),
                confidence=decision.get("confidence", 0),
                status="open",
                agent_reasoning=reasoning,
            )
        )

        # Update risk state
        await session.execute(
            RISK_STATE.update()
            .where(RISK_STATE.c.id == 1)
            .values(
                open_trade_count=RISK_STATE.c.open_trade_count + 1,
                open_instruments=sa.func.array_append(
                    RISK_STATE.c.open_instruments, proposal.instrument
                ),
                last_updated=now,
            )
        )

    logger.info("Trade logged", extra={"trade_id": proposal.trade_id})


async def close_trade(
    trade_id: str,
    exit_price: float,
    entry_price: float,
    direction: str,
    position_size: float,
    pip_size: float = 0.0001,
) -> dict[str, Any]:
    """Close a trade and update P&L."""
    now = datetime.now(timezone.utc)

    pnl_pips = (exit_price - entry_price) / pip_size if direction == "LONG" else (entry_price - exit_price) / pip_size
    pnl_pct = pnl_pips * position_size / 100  # Simplified

    status = "closed_win" if pnl_pips > 0 else "closed_loss"

    async with get_session() as session:
        # Update trade record
        await session.execute(
            TRADES_TABLE.update()
            .where(TRADES_TABLE.c.trade_id == trade_id)
            .values(
                status=status,
                exit_price=exit_price,
                exit_ts=now,
                pnl_pips=round(pnl_pips, 2),
                pnl_percent=round(pnl_pct, 3),
            )
        )

        # Update risk state
        update_values: dict[str, Any] = {
            "daily_pnl_pct": RISK_STATE.c.daily_pnl_pct + pnl_pct,
            "weekly_pnl_pct": RISK_STATE.c.weekly_pnl_pct + pnl_pct,
            "open_trade_count": sa.func.greatest(RISK_STATE.c.open_trade_count - 1, 0),
            "last_updated": now,
        }

        if status == "closed_loss":
            update_values["consecutive_losses"] = RISK_STATE.c.consecutive_losses + 1
        else:
            update_values["consecutive_losses"] = 0

        await session.execute(
            RISK_STATE.update()
            .where(RISK_STATE.c.id == 1)
            .values(**update_values)
        )

    logger.info(
        "Trade closed",
        extra={"trade_id": trade_id, "status": status,
               "pnl_pips": round(pnl_pips, 2), "pnl_pct": round(pnl_pct, 3)},
    )

    return {
        "trade_id": trade_id,
        "status": status,
        "pnl_pips": round(pnl_pips, 2),
        "pnl_pct": round(pnl_pct, 3),
    }


async def get_open_trades() -> list[dict[str, Any]]:
    """Get all open trades."""
    query = (
        sa.select(TRADES_TABLE)
        .where(TRADES_TABLE.c.status == "open")
        .order_by(TRADES_TABLE.c.ts.desc())
    )

    async with get_session() as session:
        result = await session.execute(query)
        return [dict(row._mapping) for row in result.fetchall()]


async def get_daily_trades(date: datetime | None = None) -> list[dict[str, Any]]:
    """Get all trades for a given day."""
    date = date or datetime.now(timezone.utc)
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(hour=23, minute=59, second=59)

    query = (
        sa.select(TRADES_TABLE)
        .where(sa.and_(TRADES_TABLE.c.ts >= start, TRADES_TABLE.c.ts <= end))
        .order_by(TRADES_TABLE.c.ts.asc())
    )

    async with get_session() as session:
        result = await session.execute(query)
        return [dict(row._mapping) for row in result.fetchall()]


async def reset_daily_pnl() -> None:
    """Reset daily P&L at start of new trading day."""
    async with get_session() as session:
        await session.execute(
            RISK_STATE.update()
            .where(RISK_STATE.c.id == 1)
            .values(daily_pnl_pct=0, daily_halt=False, last_updated=datetime.now(timezone.utc))
        )


async def reset_weekly_pnl() -> None:
    """Reset weekly P&L at start of new trading week (Monday)."""
    async with get_session() as session:
        await session.execute(
            RISK_STATE.update()
            .where(RISK_STATE.c.id == 1)
            .values(weekly_pnl_pct=0, weekly_halt=False, consecutive_losses=0,
                    last_updated=datetime.now(timezone.utc))
        )
