"""
Seed the Historical Brain with foundational market events.

Categories seeded:
  1. Major central bank decisions (last 20 years of key events)
  2. Significant economic releases and their market impact
  3. Geopolitical events mapped to price action
  4. Regime transitions (risk-on → risk-off and back)
  5. Notable technical pattern outcomes

Run once during initial setup, then incrementally as new events occur.

Usage:
  python -m scripts.seed_knowledge_base
"""

from __future__ import annotations

import asyncio
import logging

from src.brain.knowledge_base import HistoricalBrain
from src.data.storage.database import init_db, close_db
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ── Seed Events ───────────────────────────────────────────────

SEED_EVENTS: list[dict] = [
    # Central Bank — Fed
    {
        "event_type": "central_bank",
        "event_name": "Fed Emergency Rate Cut to Zero — COVID",
        "description": "Federal Reserve cuts rates to 0-0.25% in emergency meeting, announces unlimited QE. Largest single-day policy action in Fed history.",
        "ts": "2020-03-15T00:00:00Z",
        "affected_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"],
        "market_context": {"regime": "crisis", "vix": 82, "trigger": "COVID-19 pandemic"},
        "price_impact": {
            "EUR/USD": {"pips": 150, "direction": "up", "duration_hours": 48},
            "XAU/USD": {"pips": 5000, "direction": "up", "duration_hours": 168},
            "USD/JPY": {"pips": -300, "direction": "down", "duration_hours": 72},
        },
        "tags": ["fed", "rate_cut", "emergency", "covid", "qe"],
    },
    {
        "event_type": "central_bank",
        "event_name": "Fed First Rate Hike Post-COVID",
        "description": "Federal Reserve raises rates by 25bps for the first time since COVID, beginning tightening cycle. Signals more hikes ahead.",
        "ts": "2022-03-16T00:00:00Z",
        "affected_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"],
        "market_context": {"regime": "transitioning", "cpi": 7.9, "trigger": "inflation"},
        "price_impact": {
            "EUR/USD": {"pips": -80, "direction": "down", "duration_hours": 168},
            "USD/JPY": {"pips": 500, "direction": "up", "duration_hours": 720},
            "XAU/USD": {"pips": -3000, "direction": "down", "duration_hours": 720},
        },
        "tags": ["fed", "rate_hike", "tightening_cycle", "inflation"],
    },
    # Central Bank — ECB
    {
        "event_type": "central_bank",
        "event_name": "ECB Ends Negative Rates",
        "description": "ECB raises rates by 50bps, ending 8 years of negative interest rates. Bigger than expected hike signals determination to fight inflation.",
        "ts": "2022-07-21T00:00:00Z",
        "affected_pairs": ["EUR/USD"],
        "market_context": {"regime": "transitioning", "eu_cpi": 8.6},
        "price_impact": {
            "EUR/USD": {"pips": 120, "direction": "up", "duration_hours": 48},
        },
        "tags": ["ecb", "rate_hike", "negative_rates", "inflation"],
    },
    # Central Bank — BOJ
    {
        "event_type": "central_bank",
        "event_name": "BOJ Yield Curve Control Adjustment",
        "description": "Bank of Japan widens YCC band to ±0.50%, effectively allowing rates to rise. Surprise move catches market off-guard.",
        "ts": "2022-12-20T00:00:00Z",
        "affected_pairs": ["USD/JPY"],
        "market_context": {"regime": "risk_off", "usdjpy": 137},
        "price_impact": {
            "USD/JPY": {"pips": -400, "direction": "down", "duration_hours": 24},
        },
        "tags": ["boj", "ycc", "surprise", "policy_shift"],
    },
    # Geopolitical
    {
        "event_type": "geopolitical",
        "event_name": "Russia Invades Ukraine",
        "description": "Russia launches full-scale invasion of Ukraine. Global markets enter risk-off mode. Energy prices spike. Safe havens bid.",
        "ts": "2022-02-24T00:00:00Z",
        "affected_pairs": ["EUR/USD", "XAU/USD", "WTI", "USD/CHF", "USD/JPY"],
        "market_context": {"regime": "crisis", "vix": 37, "trigger": "war"},
        "price_impact": {
            "EUR/USD": {"pips": -200, "direction": "down", "duration_hours": 720},
            "XAU/USD": {"pips": 8000, "direction": "up", "duration_hours": 168},
            "WTI": {"pips": 3000, "direction": "up", "duration_hours": 168},
            "USD/CHF": {"pips": -100, "direction": "down", "duration_hours": 48},
        },
        "tags": ["geopolitical", "war", "crisis", "energy", "safe_haven"],
    },
    {
        "event_type": "geopolitical",
        "event_name": "SVB Collapse — Banking Crisis",
        "description": "Silicon Valley Bank collapses, triggering fears of broader banking crisis. Flight to safety. Fed pause expectations surge.",
        "ts": "2023-03-10T00:00:00Z",
        "affected_pairs": ["EUR/USD", "XAU/USD", "USD/CHF", "USD/JPY"],
        "market_context": {"regime": "crisis", "vix": 26, "trigger": "banking_crisis"},
        "price_impact": {
            "EUR/USD": {"pips": 200, "direction": "up", "duration_hours": 168},
            "XAU/USD": {"pips": 10000, "direction": "up", "duration_hours": 720},
            "USD/JPY": {"pips": -500, "direction": "down", "duration_hours": 168},
        },
        "tags": ["banking_crisis", "safe_haven", "fed_pause", "svb"],
    },
    # Economic Releases
    {
        "event_type": "economic_release",
        "event_name": "US CPI 9.1% — Inflation Peak",
        "description": "US CPI hits 9.1% YoY, the highest in 40 years. USD surges as markets price in aggressive Fed hiking.",
        "ts": "2022-06-10T00:00:00Z",
        "affected_pairs": ["EUR/USD", "GBP/USD", "XAU/USD"],
        "market_context": {"regime": "risk_off", "fed_funds": 1.5},
        "price_impact": {
            "EUR/USD": {"pips": -150, "direction": "down", "duration_hours": 168},
            "XAU/USD": {"pips": -5000, "direction": "down", "duration_hours": 168},
        },
        "tags": ["cpi", "inflation", "fed", "usd_strength"],
    },
    # Regime Transitions
    {
        "event_type": "regime_change",
        "event_name": "Risk-On to Risk-Off — COVID Crash",
        "description": "Global markets shift from complacent risk-on to panic risk-off as COVID spreads globally. VIX from 15 to 82 in 3 weeks.",
        "ts": "2020-02-20T00:00:00Z",
        "affected_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "WTI"],
        "market_context": {"from_regime": "risk_on", "to_regime": "crisis", "vix_from": 15, "vix_to": 82},
        "price_impact": {
            "AUD/USD": {"pips": -1200, "direction": "down", "duration_hours": 720},
            "XAU/USD": {"pips": -15000, "direction": "down", "duration_hours": 168},
            "WTI": {"pips": -6000, "direction": "down", "duration_hours": 720},
            "USD/JPY": {"pips": -800, "direction": "down", "duration_hours": 720},
        },
        "tags": ["regime_change", "crisis", "covid", "risk_off", "vix_spike"],
    },
    # Oil Supply Shock
    {
        "event_type": "commodity",
        "event_name": "Oil Negative — WTI Below Zero",
        "description": "WTI crude trades below zero for the first time in history as storage capacity runs out during COVID demand collapse.",
        "ts": "2020-04-20T00:00:00Z",
        "affected_pairs": ["WTI", "USD/CAD", "XAU/USD"],
        "market_context": {"regime": "crisis", "trigger": "demand_collapse"},
        "price_impact": {
            "WTI": {"pips": -50000, "direction": "down", "duration_hours": 24},
            "USD/CAD": {"pips": 200, "direction": "up", "duration_hours": 48},
        },
        "tags": ["oil", "negative_price", "demand_shock", "covid", "historic"],
    },
    # SNB Floor Removal
    {
        "event_type": "central_bank",
        "event_name": "SNB Removes EUR/CHF Floor",
        "description": "Swiss National Bank unexpectedly removes the 1.20 EUR/CHF floor. CHF surges 30% in minutes. Brokers go bankrupt.",
        "ts": "2015-01-15T00:00:00Z",
        "affected_pairs": ["USD/CHF", "EUR/USD"],
        "market_context": {"regime": "crisis", "trigger": "central_bank_surprise"},
        "price_impact": {
            "USD/CHF": {"pips": -2800, "direction": "down", "duration_hours": 1},
        },
        "tags": ["snb", "intervention", "flash_crash", "historic", "chf"],
    },
]


async def seed_knowledge_base() -> int:
    """Seed the Brain with foundational events. Returns count of events stored."""
    brain = HistoricalBrain()
    count = 0

    for event in SEED_EVENTS:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(event["ts"].replace("Z", "+00:00"))

            await brain.store_event(
                event_type=event["event_type"],
                event_name=event["event_name"],
                description=event["description"],
                affected_pairs=event["affected_pairs"],
                market_context=event["market_context"],
                price_impact=event["price_impact"],
                tags=event.get("tags", []),
                ts=ts,
            )
            count += 1
            logger.info("Seeded: %s", event["event_name"])

        except Exception as e:
            logger.error("Failed to seed: %s — %s", event["event_name"], str(e))

    logger.info("Knowledge base seeding complete: %d events", count)
    return count


async def main() -> None:
    setup_logging()
    await init_db()

    logger.info("Starting knowledge base seeding")
    count = await seed_knowledge_base()
    logger.info("Seeded %d foundational events", count)

    stats = await HistoricalBrain().get_stats()
    logger.info("Brain stats: %s", stats)

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
