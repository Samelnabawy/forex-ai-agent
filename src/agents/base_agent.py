"""
Base Agent — all 9 agents inherit from this.

Provides:
  - Structured logging with agent context
  - Historical Brain query interface
  - Decision persistence to PostgreSQL
  - Execution timing and metrics
  - Standard lifecycle (init → analyze → log)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.instruments import INSTRUMENTS
from src.data.storage.cache import set_agent_state
from src.data.storage.database import get_session
from src.logging_config import get_agent_logger
from src.models import BrainQuery, HistoricalMatch, MarketState

# Table reference for agent decisions
AGENT_DECISIONS = sa.table(
    "agent_decisions",
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("agent_name", sa.String),
    sa.column("instrument", sa.String),
    sa.column("decision", sa.JSON),
    sa.column("confidence", sa.Numeric),
    sa.column("execution_ms", sa.Integer),
)


class BaseAgent(ABC):
    """
    Abstract base for all trading agents.

    Subclasses must implement:
      - analyze(market_state) → dict (the agent's structured output)
      - name (class variable)
      - description (class variable)
    """

    name: ClassVar[str]
    description: ClassVar[str]
    execution_frequency: ClassVar[str]  # "5m", "1h", "30m", etc.

    def __init__(self) -> None:
        self.logger = get_agent_logger(self.name)
        self._last_execution: datetime | None = None
        self._execution_count: int = 0
        self._total_execution_ms: int = 0

    # ── Core Interface ────────────────────────────────────────

    @abstractmethod
    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """
        Core analysis logic. Each agent implements this differently.
        Must return a structured dict matching the agent's output schema.
        """
        ...

    async def run(self, market_state: MarketState) -> dict[str, Any]:
        """
        Execute the agent's analysis cycle with timing, logging, and persistence.
        This is the method called by the orchestrator.
        """
        start = time.monotonic()
        self.logger.info("Agent starting analysis")

        try:
            result = await self.analyze(market_state)

            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._execution_count += 1
            self._total_execution_ms += elapsed_ms
            self._last_execution = datetime.now(timezone.utc)

            # Extract confidence if present
            confidence = result.get("confidence")
            instrument = result.get("instrument")

            # Persist decision
            await self._log_decision(
                instrument=instrument,
                decision=result,
                confidence=Decimal(str(confidence)) if confidence else None,
                execution_ms=elapsed_ms,
            )

            # Cache latest state in Redis
            await set_agent_state(self.name, result)

            self.logger.info(
                "Agent analysis complete",
                extra={
                    "execution_ms": elapsed_ms,
                    "instrument": instrument,
                    "confidence": confidence,
                },
            )

            return result

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self.logger.error(
                "Agent analysis failed",
                extra={"error": str(e), "execution_ms": elapsed_ms},
                exc_info=True,
            )
            raise

    # ── Historical Brain Interface ────────────────────────────

    async def query_brain(
        self,
        query_text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[HistoricalMatch]:
        """
        Query the Historical Brain for precedents.
        All agents can call this to find "what happened last time?"
        """
        self.logger.debug(
            "Brain query",
            extra={"query": query_text, "top_k": top_k},
        )

        try:
            from src.brain.knowledge_base import get_brain
            brain = get_brain()
            filters = filters or {}
            return await brain.query(
                query_text=query_text,
                top_k=top_k,
                event_type=filters.get("event_type"),
                affected_pairs=filters.get("affected_pairs"),
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                tags=filters.get("tags"),
                requesting_agent=self.name,
            )
        except Exception as e:
            self.logger.error("Brain query failed", extra={"error": str(e)})
            return []

    # ── Decision Persistence ──────────────────────────────────

    async def _log_decision(
        self,
        instrument: str | None,
        decision: dict[str, Any],
        confidence: Decimal | None,
        execution_ms: int,
    ) -> None:
        """Log every agent decision to PostgreSQL for audit trail."""
        try:
            async with get_session() as session:
                await session.execute(
                    AGENT_DECISIONS.insert().values(
                        ts=datetime.now(timezone.utc),
                        agent_name=self.name,
                        instrument=instrument,
                        decision=decision,
                        confidence=confidence,
                        execution_ms=execution_ms,
                    )
                )
        except Exception as e:
            # Don't let logging failures crash the agent
            self.logger.error(
                "Failed to log decision",
                extra={"error": str(e)},
            )

    # ── Utilities ─────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        avg_ms = (
            self._total_execution_ms / self._execution_count
            if self._execution_count > 0
            else 0
        )
        return {
            "agent": self.name,
            "executions": self._execution_count,
            "avg_execution_ms": round(avg_ms),
            "last_execution": (
                self._last_execution.isoformat() if self._last_execution else None
            ),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
