"""
Telegram Signal Delivery.

Sends formatted trade signals to the Telegram channel.
Also sends alerts for system events (regime changes, circuit breakers, etc).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.agents.evidence import TradeProposal
from src.config.instruments import INSTRUMENTS
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def format_trade_signal(
    decision: dict[str, Any],
    proposal: TradeProposal,
) -> str:
    """Format a trade decision as a Telegram message."""
    direction = proposal.direction
    emoji = "🔵" if direction == "LONG" else "🔴"
    inst = INSTRUMENTS.get(proposal.instrument)
    pip_size = float(inst.pip_size) if inst else 0.0001

    risk_pips = round(abs(proposal.entry - proposal.stop_loss) / pip_size, 1)
    tp1_pips = round(abs(proposal.take_profit_1 - proposal.entry) / pip_size, 1)
    tp2_pips = round(abs((proposal.take_profit_2 or proposal.take_profit_1) - proposal.entry) / pip_size, 1)

    rr1 = round(tp1_pips / risk_pips, 1) if risk_pips > 0 else 0
    rr2 = round(tp2_pips / risk_pips, 1) if risk_pips > 0 else 0

    size_pct = decision.get("final_position_size", 0)
    settings = get_settings()
    capital = settings.initial_capital
    risk_amount = capital * size_pct / 100

    # Approximate lot size (simplified)
    pip_value = float(inst.pip_value_per_lot) if inst else 10.0
    lots = round(risk_amount / (risk_pips * pip_value), 2) if risk_pips > 0 and pip_value > 0 else 0

    confluence = decision.get("matrix", {}).get("agreement_ratio", 0) * 100
    debate = decision.get("debate", {})
    ev_pips = debate.get("expected_value", {}).get("expected_value_pips", 0)

    # Build reasoning summary
    reasoning = debate.get("reasoning", "")
    if not reasoning:
        bull = debate.get("bull_summary", {}).get("key_points", "")
        reasoning = bull[:200] if bull else "Multi-agent consensus"

    # Risk summary
    risks = debate.get("bear_summary", {}).get("key_risks", [])
    risk_text = risks[0] if risks else "Standard market risk"

    # Invalidation
    invalidation = debate.get("bull_summary", {}).get("invalidation", [])
    invalidation_text = invalidation[0] if invalidation else f"Price breaks {'below' if direction == 'LONG' else 'above'} SL"

    msg = f"""{emoji} {direction} {proposal.instrument}

Entry:  {proposal.entry:.5f}
SL:     {proposal.stop_loss:.5f} (-{risk_pips:.0f} pips)
TP1:    {proposal.take_profit_1:.5f} (+{tp1_pips:.0f} pips)"""

    if proposal.take_profit_2:
        msg += f"\nTP2:    {proposal.take_profit_2:.5f} (+{tp2_pips:.0f} pips)"

    msg += f"""

Size:   {size_pct:.2f}% risk ({lots:.2f} lots on ${capital:,.0f})
R:R:    1:{rr1}"""
    if proposal.take_profit_2:
        msg += f" / 1:{rr2}"

    msg += f"""

📊 Confluence: {confluence:.0f}/100
🎯 EV: {ev_pips:+.1f} pips per trade

Why: {reasoning[:300]}

⚠️ Risk: {risk_text[:150]}
🚫 Bail if: {invalidation_text[:150]}

#{proposal.instrument.replace('/', '')} #{direction} #Forex"""

    return msg


def format_alert(alert_type: str, message: str, urgency: str = "normal") -> str:
    """Format a system alert for Telegram."""
    emoji_map = {
        "regime_change": "🔄",
        "circuit_breaker": "🔴",
        "daily_halt": "⛔",
        "weekly_halt": "⛔",
        "system_error": "❌",
        "trade_closed": "✅" if "win" in message.lower() else "❎",
    }
    urgency_prefix = {
        "normal": "",
        "high": "⚠️ ",
        "critical": "🚨 CRITICAL: ",
    }

    emoji = emoji_map.get(alert_type, "📢")
    prefix = urgency_prefix.get(urgency, "")

    return f"{emoji} {prefix}{message}"


async def send_trade_signal(
    decision: dict[str, Any],
    proposal: TradeProposal,
) -> bool:
    """Send a trade signal to Telegram."""
    settings = get_settings()
    token = settings.telegram_bot_token.get_secret_value()
    chat_id = settings.telegram_chat_id

    if not token or not chat_id:
        logger.warning("Telegram not configured — signal not sent")
        return False

    message = format_trade_signal(decision, proposal)

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            logger.info("Telegram signal sent", extra={"trade_id": proposal.trade_id})
            return True
    except Exception as e:
        logger.error("Telegram send failed", extra={"error": str(e)})
        return False


async def send_alert(
    alert_type: str,
    message: str,
    urgency: str = "normal",
) -> bool:
    """Send a system alert to Telegram."""
    settings = get_settings()
    token = settings.telegram_bot_token.get_secret_value()
    chat_id = settings.telegram_chat_id

    if not token or not chat_id:
        return False

    formatted = format_alert(alert_type, message, urgency)

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": formatted},
                timeout=10.0,
            )
            return True
    except Exception:
        return False
