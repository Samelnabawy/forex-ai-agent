"""
Historical Brain Knowledge Base — Agent 09

"The Historian" — knows everything that ever happened in markets
and can recall it instantly.

This is the main interface all agents use to query the Brain.
Not a decision-making agent — it's a service that ENHANCES all other agents.

Combines:
  - Semantic similarity search (pgvector)
  - Pattern matching (cataloged patterns with win rates)
  - Event precedent queries (what happened last time?)
  - Regime transition history
  - Central bank decision history

All agents call: brain.query("RSI < 30 on EUR/USD during ECB hawkish cycle")
Brain returns: 47 matches, 34 rallied (72%), avg move +0.6% in 48h
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.brain.embeddings import encode_market_moment, generate_embedding
from src.brain.patterns import ALL_PATTERNS, Pattern, find_matching_patterns
from src.brain.query_engine import (
    get_brain_stats,
    query_by_setup,
    query_central_bank_precedents,
    query_event_precedents,
    query_regime_precedents,
    similarity_search,
)
from src.data.storage.database import get_session
from src.models import BrainQuery, HistoricalMatch

logger = logging.getLogger(__name__)


class HistoricalBrain:
    """
    Agent 09: Historical Brain (RAG Knowledge Base)

    The MEMORY of the entire system. All other agents query it
    to find historical precedents for current market conditions.
    """

    def __init__(self) -> None:
        self._query_count = 0
        self._avg_query_ms = 0.0

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        event_type: str | None = None,
        affected_pairs: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        tags: list[str] | None = None,
        requesting_agent: str = "",
    ) -> list[HistoricalMatch]:
        """
        Main query interface — all agents use this.

        Args:
            query_text: Natural language description of the market condition
            top_k: Number of results to return
            event_type: Filter by event type (central_bank, economic_release, etc.)
            affected_pairs: Filter by affected instruments
            date_from/date_to: Date range filter
            tags: Filter by tags
            requesting_agent: Name of the agent making the query (for logging)

        Returns:
            List of HistoricalMatch objects, sorted by similarity score
        """
        self._query_count += 1

        logger.info(
            "Brain query",
            extra={
                "query": query_text[:100],
                "top_k": top_k,
                "requesting_agent": requesting_agent,
                "filters": {
                    "event_type": event_type,
                    "pairs": affected_pairs,
                    "tags": tags,
                },
            },
        )

        matches = await similarity_search(
            query_text=query_text,
            top_k=top_k,
            event_type=event_type,
            affected_pairs=affected_pairs,
            date_from=date_from,
            date_to=date_to,
            tags=tags,
        )

        return matches

    async def query_setup(
        self,
        instrument: str,
        indicators: dict[str, Any],
        macro_regime: str,
        outcome: str | None = None,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """
        Query for setups similar to the current one.
        Returns matches + pattern analysis + statistical summary.

        Used by Bull (outcome="win") and Bear (outcome="loss").
        """
        # Similarity search
        matches = await query_by_setup(
            instrument=instrument,
            indicators=indicators,
            macro_regime=macro_regime,
            outcome_filter=outcome,
            top_k=top_k,
        )

        # Pattern matching
        pattern_matches = find_matching_patterns(
            indicators=indicators,
            macro_context={"macro_regime": macro_regime},
        )

        # Statistical summary
        if matches:
            win_matches = [m for m in matches if m.price_impact.get(instrument, {}).get("pips", 0) > 0]
            loss_matches = [m for m in matches if m.price_impact.get(instrument, {}).get("pips", 0) < 0]

            avg_win = sum(
                m.price_impact.get(instrument, {}).get("pips", 0) for m in win_matches
            ) / len(win_matches) if win_matches else 0

            avg_loss = sum(
                m.price_impact.get(instrument, {}).get("pips", 0) for m in loss_matches
            ) / len(loss_matches) if loss_matches else 0

            stats = {
                "total_matches": len(matches),
                "wins": len(win_matches),
                "losses": len(loss_matches),
                "win_rate": round(len(win_matches) / len(matches), 2) if matches else 0,
                "avg_win_pips": round(avg_win, 1),
                "avg_loss_pips": round(avg_loss, 1),
                "avg_similarity": round(
                    sum(m.similarity_score for m in matches) / len(matches), 3
                ),
            }
        else:
            stats = {"total_matches": 0, "wins": 0, "losses": 0, "win_rate": 0.5}

        return {
            "matches": [
                {
                    "event_name": m.event_name,
                    "date": m.ts.isoformat(),
                    "similarity": m.similarity_score,
                    "price_impact": m.price_impact,
                    "description": m.description[:200],
                }
                for m in matches[:10]
            ],
            "patterns": [
                {
                    "name": p.name,
                    "match_score": round(score, 2),
                    "historical_win_rate": p.outcome.win_rate,
                    "avg_win_pips": p.outcome.avg_win_pips,
                    "avg_loss_pips": p.outcome.avg_loss_pips,
                    "strengtheners": p.outcome.strengtheners,
                    "weakeners": p.outcome.weakeners,
                }
                for p, score in pattern_matches[:5]
            ],
            "statistics": stats,
        }

    async def query_event(
        self,
        event_name: str,
        conditions: str = "",
        top_k: int = 20,
    ) -> dict[str, Any]:
        """
        Query for specific event type precedents.

        Example: brain.query_event("NFP miss", "greater than 50K during rate cut cycle")
        Returns: matches with price impacts and statistical summary
        """
        matches = await query_event_precedents(event_name, conditions, top_k)

        # Aggregate impacts across instruments
        impact_summary: dict[str, dict[str, Any]] = {}
        for m in matches:
            for instrument, impact in m.price_impact.items():
                if instrument not in impact_summary:
                    impact_summary[instrument] = {"moves": [], "count": 0}
                pips = impact.get("pips", 0)
                impact_summary[instrument]["moves"].append(pips)
                impact_summary[instrument]["count"] += 1

        for instrument, data in impact_summary.items():
            moves = data["moves"]
            if moves:
                data["avg_pips"] = round(sum(moves) / len(moves), 1)
                data["max_pips"] = round(max(moves), 1)
                data["min_pips"] = round(min(moves), 1)
                data["positive_pct"] = round(sum(1 for m in moves if m > 0) / len(moves) * 100, 0)

        return {
            "event": event_name,
            "conditions": conditions,
            "total_precedents": len(matches),
            "impact_by_instrument": impact_summary,
            "matches": [
                {
                    "date": m.ts.isoformat(),
                    "event": m.event_name,
                    "similarity": m.similarity_score,
                    "impact": m.price_impact,
                }
                for m in matches[:10]
            ],
        }

    async def store_event(
        self,
        event_type: str,
        event_name: str,
        description: str,
        affected_pairs: list[str],
        market_context: dict[str, Any],
        price_impact: dict[str, Any],
        tags: list[str] | None = None,
        ts: datetime | None = None,
    ) -> int:
        """
        Store a new market event in the knowledge base.
        Generates and stores the embedding.
        """
        ts = ts or datetime.now(timezone.utc)
        tags = tags or []

        # Generate embedding
        embed_text = f"{event_type}: {event_name}. {description}"
        embedding = await generate_embedding(embed_text)

        async with get_session() as session:
            result = await session.execute(
                sa.text("""
                    INSERT INTO market_events
                        (ts, event_type, event_name, description, affected_pairs,
                         market_context, price_impact, tags, embedding)
                    VALUES
                        (:ts, :event_type, :event_name, :description, :affected_pairs,
                         :market_context, :price_impact, :tags, :embedding::vector)
                    RETURNING id
                """),
                {
                    "ts": ts,
                    "event_type": event_type,
                    "event_name": event_name,
                    "description": description,
                    "affected_pairs": affected_pairs,
                    "market_context": __import__("json").dumps(market_context),
                    "price_impact": __import__("json").dumps(price_impact),
                    "tags": tags,
                    "embedding": str(embedding),
                },
            )
            event_id = result.scalar()

        logger.info(
            "Stored event in Brain",
            extra={"event_id": event_id, "event_type": event_type, "event_name": event_name},
        )
        return event_id or 0

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        db_stats = await get_brain_stats()
        return {
            **db_stats,
            "query_count": self._query_count,
            "pattern_catalog_size": len(ALL_PATTERNS),
        }


# ── Singleton ─────────────────────────────────────────────────

_brain: HistoricalBrain | None = None


def get_brain() -> HistoricalBrain:
    """Get or create the Historical Brain singleton."""
    global _brain
    if _brain is None:
        _brain = HistoricalBrain()
    return _brain
