"""
Central Bank Statement Feed.

Fetches and stores central bank statements/press releases.
Used by Sentiment Agent for language delta analysis (hawkish/dovish shift detection).

Sources: Central bank websites via RSS, FRED speech data
Stores: PostgreSQL for historical diffing

The key insight: the CHANGE in language between consecutive statements
is one of the most reliable predictors of future rate actions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.settings import get_settings
from src.data.storage.database import get_session

logger = logging.getLogger(__name__)

# Central bank RSS/API sources
CB_SOURCES: dict[str, dict[str, str]] = {
    "Fed": {
        "rss": "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "currency": "USD",
    },
    "ECB": {
        "rss": "https://www.ecb.europa.eu/rss/press.html",
        "currency": "EUR",
    },
    "BOE": {
        "rss": "https://www.bankofengland.co.uk/rss/news",
        "currency": "GBP",
    },
    "BOJ": {
        "rss": "https://www.boj.or.jp/en/rss/whatsnew.xml",
        "currency": "JPY",
    },
    "SNB": {
        "url": "https://www.snb.ch/en",
        "currency": "CHF",
    },
    "RBA": {
        "rss": "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
        "currency": "AUD",
    },
    "RBNZ": {
        "url": "https://www.rbnz.govt.nz",
        "currency": "NZD",
    },
}

# Table for storing statements
CB_STATEMENTS = sa.table(
    "market_events",  # Reuse market_events table with event_type='central_bank_statement'
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("event_type", sa.String),
    sa.column("event_name", sa.String),
    sa.column("description", sa.Text),
    sa.column("affected_pairs", sa.ARRAY(sa.String)),
    sa.column("market_context", sa.JSON),
    sa.column("price_impact", sa.JSON),
    sa.column("tags", sa.ARRAY(sa.String)),
)


class CentralBankFeed:
    """Fetches and stores central bank statements."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._client

    async def store_statement(
        self,
        bank: str,
        statement_text: str,
        statement_date: datetime,
        title: str = "",
    ) -> None:
        """Store a central bank statement in the market_events table."""
        source = CB_SOURCES.get(bank, {})
        currency = source.get("currency", "")
        affected = self._get_affected_pairs(currency)

        async with get_session() as session:
            await session.execute(
                CB_STATEMENTS.insert().values(
                    ts=statement_date,
                    event_type="central_bank_statement",
                    event_name=f"{bank} Statement: {title}",
                    description=statement_text[:10000],  # Cap at 10K chars
                    affected_pairs=affected,
                    market_context={"bank": bank, "currency": currency},
                    price_impact={},  # Filled after the fact
                    tags=[bank.lower(), "central_bank", "statement", currency.lower()],
                )
            )

        logger.info("Stored %s statement: %s", bank, title[:50])

    async def get_latest_statements(
        self,
        bank: str,
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        """Get the N most recent statements from a central bank."""
        query = (
            sa.select(CB_STATEMENTS)
            .where(
                sa.and_(
                    CB_STATEMENTS.c.event_type == "central_bank_statement",
                    CB_STATEMENTS.c.market_context["bank"].astext == bank,
                )
            )
            .order_by(CB_STATEMENTS.c.ts.desc())
            .limit(limit)
        )

        async with get_session() as session:
            result = await session.execute(query)
            rows = result.fetchall()

        return [
            {
                "ts": row.ts.isoformat(),
                "bank": bank,
                "title": row.event_name,
                "statement": row.description,
            }
            for row in rows
        ]

    async def get_statement_pair_for_diff(
        self, bank: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Get the two most recent statements for delta analysis."""
        statements = await self.get_latest_statements(bank, limit=2)
        current = statements[0] if len(statements) > 0 else None
        previous = statements[1] if len(statements) > 1 else None
        return previous, current

    @staticmethod
    def _get_affected_pairs(currency: str) -> list[str]:
        """Map a currency to affected trading pairs."""
        pair_map = {
            "USD": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"],
            "EUR": ["EUR/USD"],
            "GBP": ["GBP/USD"],
            "JPY": ["USD/JPY"],
            "CHF": ["USD/CHF"],
            "AUD": ["AUD/USD"],
            "CAD": ["USD/CAD"],
            "NZD": ["NZD/USD"],
        }
        return pair_map.get(currency, [])

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
