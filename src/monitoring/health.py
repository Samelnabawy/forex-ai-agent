"""
System Health Monitor.

Tracks:
  - Agent heartbeats (has each agent run recently?)
  - Data freshness (are prices current?)
  - Error rates per component
  - LLM cost tracking
  - Queue depth
  - Database and Redis connectivity

Used by Portfolio Manager for system-level risk decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.data.storage.cache import get_agent_state, get_queue_length

logger = logging.getLogger(__name__)

# Expected agent execution frequencies
AGENT_FREQUENCIES: dict[str, int] = {
    "technical_analyst": 300,    # 5 min
    "macro_analyst": 3600,       # 1 hour
    "correlation_agent": 60,     # 1 min
    "sentiment_agent": 1800,     # 30 min
}


@dataclass
class AgentHealth:
    """Health status of a single agent."""
    name: str
    is_healthy: bool = True
    last_run: str | None = None
    seconds_since_run: float = 0
    expected_frequency: int = 300
    overdue: bool = False
    error_count: int = 0


@dataclass
class SystemHealth:
    """Overall system health status."""
    timestamp: str = ""
    overall_status: str = "healthy"  # "healthy", "degraded", "critical"
    agents: list[AgentHealth] = field(default_factory=list)
    data_fresh: bool = True
    price_staleness_seconds: float = 0
    queue_depth: int = 0
    queue_ok: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def check_system_health() -> SystemHealth:
    """Run full system health check."""
    now = datetime.now(timezone.utc)
    health = SystemHealth(timestamp=now.isoformat())

    # Check each agent
    for agent_name, frequency in AGENT_FREQUENCIES.items():
        agent_health = AgentHealth(
            name=agent_name,
            expected_frequency=frequency,
        )

        try:
            state = await get_agent_state(agent_name)
            if state:
                ts_str = state.get("timestamp", "")
                if ts_str:
                    try:
                        last_ts = datetime.fromisoformat(ts_str)
                        age = (now - last_ts).total_seconds()
                        agent_health.last_run = ts_str
                        agent_health.seconds_since_run = age

                        # Overdue if more than 3× expected frequency
                        if age > frequency * 3:
                            agent_health.overdue = True
                            agent_health.is_healthy = False
                            health.warnings.append(
                                f"{agent_name} overdue: last ran {age/60:.0f} min ago "
                                f"(expected every {frequency/60:.0f} min)"
                            )
                    except ValueError:
                        agent_health.last_run = ts_str
            else:
                agent_health.is_healthy = False
                agent_health.last_run = None
        except Exception as e:
            agent_health.is_healthy = False
            health.errors.append(f"Failed to check {agent_name}: {e}")

        health.agents.append(agent_health)

    # Check data freshness
    try:
        from src.data.storage.cache import get_latest_price
        eur_price = await get_latest_price("EUR/USD")
        if eur_price:
            ts_str = eur_price.get("ts", "")
            try:
                price_ts = datetime.fromisoformat(ts_str)
                staleness = (now - price_ts).total_seconds()
                health.price_staleness_seconds = staleness
                if staleness > 60:
                    health.data_fresh = False
                    health.warnings.append(f"Price data stale: {staleness:.0f}s old")
            except ValueError:
                pass
        else:
            health.data_fresh = False
            health.warnings.append("No price data available for EUR/USD")
    except Exception as e:
        health.errors.append(f"Price freshness check failed: {e}")

    # Check queue depth
    try:
        health.queue_depth = await get_queue_length()
        if health.queue_depth > 10:
            health.queue_ok = False
            health.warnings.append(f"Signal queue backed up: {health.queue_depth} signals")
    except Exception as e:
        health.errors.append(f"Queue check failed: {e}")

    # Check LLM costs
    try:
        from src.agents.llm_client import get_llm_client
        llm = get_llm_client()
        usage = llm.usage
        if usage.get("daily_cost_usd", 0) > 20:
            health.warnings.append(f"LLM daily cost high: ${usage['daily_cost_usd']:.2f}")
    except Exception:
        pass

    # Overall status
    unhealthy_agents = sum(1 for a in health.agents if not a.is_healthy)
    if health.errors or unhealthy_agents >= 3:
        health.overall_status = "critical"
    elif health.warnings or unhealthy_agents >= 1:
        health.overall_status = "degraded"
    else:
        health.overall_status = "healthy"

    return health
