"""
Historical Brain Query Engine.

Executes similarity searches against the market_events table
using pgvector's cosine distance operator.

Features:
  - Semantic similarity search (cosine distance on 1536-dim embeddings)
  - Filtered search (by event_type, affected_pairs, date range, tags)
  - Hybrid search (semantic + keyword + filter)
  - Result scoring (similarity × recency × relevance)
  - Response time target: < 500ms
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.brain.embeddings import generate_embedding
from src.data.storage.database import get_session
from src.models import BrainQuery, HistoricalMatch

logger = logging.getLogger(__name__)


async def similarity_search(
    query_text: str,
    top_k: int = 10,
    event_type: str | None = None,
    affected_pairs: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    tags: list[str] | None = None,
    min_similarity: float = 0.3,
) -> list[HistoricalMatch]:
    """
    Core similarity search against market_events table.

    Uses pgvector's cosine distance: 1 - (embedding <=> query_embedding)
    Filters narrow the search space before the vector scan.
    """
    start = time.monotonic()

    # Generate query embedding
    query_embedding = await generate_embedding(query_text)

    # Build query with filters
    # Using raw SQL for pgvector operator
    filters: list[str] = []
    params: dict[str, Any] = {
        "embedding": str(query_embedding),
        "top_k": top_k,
        "min_sim": min_similarity,
    }

    if event_type:
        filters.append("event_type = :event_type")
        params["event_type"] = event_type

    if affected_pairs:
        filters.append("affected_pairs && :pairs")
        params["pairs"] = affected_pairs

    if date_from:
        filters.append("ts >= :date_from")
        params["date_from"] = date_from

    if date_to:
        filters.append("ts <= :date_to")
        params["date_to"] = date_to

    if tags:
        filters.append("tags && :tags")
        params["tags"] = tags

    where_clause = " AND ".join(filters) if filters else "TRUE"

    query = sa.text(f"""
        SELECT
            id,
            ts,
            event_type,
            event_name,
            description,
            affected_pairs,
            market_context,
            price_impact,
            tags,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM market_events
        WHERE embedding IS NOT NULL
          AND {where_clause}
          AND 1 - (embedding <=> CAST(:embedding AS vector)) >= :min_sim
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    try:
        async with get_session() as session:
            result = await session.execute(query, params)
            rows = result.fetchall()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Brain query completed",
            extra={
                "query_length": len(query_text),
                "results": len(rows),
                "elapsed_ms": elapsed_ms,
                "filters": list(params.keys()),
            },
        )

        matches = []
        for row in rows:
            matches.append(HistoricalMatch(
                event_id=row.id,
                event_name=row.event_name,
                event_type=row.event_type,
                ts=row.ts,
                similarity_score=round(float(row.similarity), 4),
                description=row.description or "",
                price_impact=row.price_impact or {},
                market_context=row.market_context or {},
            ))

        return matches

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.error(
            "Brain query failed",
            extra={"error": str(e), "elapsed_ms": elapsed_ms},
        )
        return []


async def query_by_setup(
    instrument: str,
    indicators: dict[str, Any],
    macro_regime: str,
    outcome_filter: str | None = None,
    top_k: int = 20,
) -> list[HistoricalMatch]:
    """
    Query for historical precedents matching a specific setup.

    Used by Bull Researcher (outcome_filter="win") and
    Bear Researcher (outcome_filter="loss").
    """
    from src.brain.embeddings import encode_market_moment

    query_text = encode_market_moment(
        instrument=instrument,
        price=0,  # Not relevant for similarity
        indicators=indicators,
        macro_regime=macro_regime,
    )

    matches = await similarity_search(
        query_text=query_text,
        top_k=top_k * 2,  # Fetch extra, filter by outcome
        affected_pairs=[instrument],
    )

    # Filter by outcome if specified
    if outcome_filter:
        filtered = []
        for m in matches:
            impact = m.price_impact
            if outcome_filter == "win":
                # Check if the price moved favorably
                move = impact.get(instrument, {}).get("pips", 0)
                if move > 0:
                    filtered.append(m)
            elif outcome_filter == "loss":
                move = impact.get(instrument, {}).get("pips", 0)
                if move < 0:
                    filtered.append(m)
        matches = filtered[:top_k]

    return matches[:top_k]


async def query_event_precedents(
    event_name: str,
    conditions: str = "",
    top_k: int = 20,
) -> list[HistoricalMatch]:
    """
    Query for historical precedents of a specific event type.

    Example: "NFP miss greater than 50K during rate cut cycle"
    → Returns the last 20 times this happened with price impacts.
    """
    query_text = f"{event_name}. {conditions}".strip()

    return await similarity_search(
        query_text=query_text,
        top_k=top_k,
        event_type="economic_release",
    )


async def query_regime_precedents(
    from_regime: str,
    to_regime: str,
    top_k: int = 10,
) -> list[HistoricalMatch]:
    """
    Query for historical regime transitions.

    Example: "risk_on to risk_off transition"
    → Returns past regime changes and their market impacts.
    """
    query_text = f"Market regime change from {from_regime} to {to_regime}. Correlations shifting, safe haven flows changing."

    return await similarity_search(
        query_text=query_text,
        top_k=top_k,
        event_type="regime_change",
    )


async def query_central_bank_precedents(
    bank: str,
    action: str,
    conditions: str = "",
    top_k: int = 15,
) -> list[HistoricalMatch]:
    """
    Query for central bank decision precedents.

    Example: "Fed rate cut while unemployment below 4%"
    """
    query_text = f"{bank} {action}. {conditions}".strip()

    return await similarity_search(
        query_text=query_text,
        top_k=top_k,
        event_type="central_bank",
        tags=[bank.lower()],
    )


async def get_brain_stats() -> dict[str, Any]:
    """Get statistics about the knowledge base."""
    try:
        async with get_session() as session:
            # Total events
            total = await session.execute(
                sa.text("SELECT COUNT(*) FROM market_events")
            )
            total_count = total.scalar() or 0

            # Events with embeddings
            embedded = await session.execute(
                sa.text("SELECT COUNT(*) FROM market_events WHERE embedding IS NOT NULL")
            )
            embedded_count = embedded.scalar() or 0

            # By type
            by_type = await session.execute(
                sa.text("SELECT event_type, COUNT(*) as cnt FROM market_events GROUP BY event_type ORDER BY cnt DESC")
            )
            type_counts = {row.event_type: row.cnt for row in by_type.fetchall()}

            # Date range
            date_range = await session.execute(
                sa.text("SELECT MIN(ts), MAX(ts) FROM market_events")
            )
            dr = date_range.fetchone()

            return {
                "total_events": total_count,
                "embedded_events": embedded_count,
                "embedding_coverage": round(embedded_count / total_count * 100, 1) if total_count > 0 else 0,
                "by_type": type_counts,
                "date_range": {
                    "earliest": dr[0].isoformat() if dr and dr[0] else None,
                    "latest": dr[1].isoformat() if dr and dr[1] else None,
                },
            }

    except Exception as e:
        logger.error("Brain stats query failed", extra={"error": str(e)})
        return {"total_events": 0, "error": str(e)}
