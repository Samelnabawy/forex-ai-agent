"""
FastAPI monitoring dashboard.
Provides health checks, agent status, and Prometheus metrics.

Endpoints:
  GET  /health          → system health
  GET  /api/v1/status   → all agent states + pipeline status
  GET  /api/v1/prices   → latest prices for all instruments
  GET  /api/v1/trades   → recent trades
  GET  /metrics         → Prometheus metrics
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.config import ALL_SYMBOLS, RISK_RULES, get_settings, validate_rules_integrity
from src.data.storage.cache import get_all_prices, get_agent_state, get_queue_length, init_redis
from src.data.storage.database import init_db, close_db
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────

AGENT_EXECUTIONS = Counter(
    "forex_agent_executions_total",
    "Total agent executions",
    ["agent_name", "status"],
)
SIGNAL_LATENCY = Histogram(
    "forex_signal_latency_seconds",
    "Signal generation latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
TRADE_DECISIONS = Counter(
    "forex_trade_decisions_total",
    "Trade decisions made",
    ["decision", "instrument"],
)

AGENTS = [
    "technical_analyst",
    "macro_analyst",
    "correlation_agent",
    "sentiment_agent",
    "bull_researcher",
    "bear_researcher",
    "risk_manager",
    "portfolio_manager",
]


# ── App Lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic."""
    setup_logging()
    logger.info("Starting Forex AI Trading Agent")

    # Validate risk rules haven't been tampered with
    validate_rules_integrity()
    logger.info("Risk rules integrity verified")

    # Connect to databases
    await init_db()
    await init_redis()

    yield

    # Cleanup
    from src.data.storage.cache import close_redis
    await close_db()
    await close_redis()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Forex AI Trading Agent",
    version="0.1.0",
    description="Multi-agent AI system for forex day-trading signals",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    """Basic health check — used by Docker healthcheck and load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "paper_trading": get_settings().paper_trading,
    }


# ── Status ────────────────────────────────────────────────────

@app.get("/api/v1/status")
async def system_status() -> dict[str, Any]:
    """Full system status — agents, pipeline, risk state."""
    agent_states = {}
    for agent in AGENTS:
        state = await get_agent_state(agent)
        agent_states[agent] = {
            "active": state is not None,
            "last_output": state,
        }

    queue_len = await get_queue_length()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents": agent_states,
        "pipeline": {
            "signal_queue_length": queue_len,
        },
        "risk_rules": {
            "daily_limit": str(RISK_RULES.daily_loss_limit_pct),
            "weekly_limit": str(RISK_RULES.weekly_loss_limit_pct),
            "max_risk_per_trade": str(RISK_RULES.max_risk_per_trade_pct),
            "max_concurrent_trades": RISK_RULES.max_concurrent_trades,
        },
    }


# ── Prices ────────────────────────────────────────────────────

@app.get("/api/v1/prices")
async def latest_prices() -> dict[str, Any]:
    """Latest cached prices for all instruments."""
    prices = await get_all_prices()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instruments": len(prices),
        "prices": prices,
    }


@app.get("/api/v1/prices/{instrument}")
async def instrument_price(instrument: str) -> dict[str, Any]:
    """Latest price for a single instrument."""
    # Normalize: accept EURUSD or EUR/USD
    normalized = instrument.replace("_", "/").upper()
    if normalized not in ALL_SYMBOLS:
        # Try adding slash
        if len(normalized) == 6:
            normalized = f"{normalized[:3]}/{normalized[3:]}"

    prices = await get_all_prices()
    price = prices.get(normalized)
    if not price:
        raise HTTPException(status_code=404, detail=f"No price data for {normalized}")

    return {"instrument": normalized, **price}


# ── Prometheus Metrics ────────────────────────────────────────

@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
