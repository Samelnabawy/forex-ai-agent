"""
Economic calendar feed.
Fetches upcoming economic events for:
  - Risk Manager: news blackout rule (±15 min of HIGH impact events)
  - Macro Agent: context for currency analysis
  - Sentiment Agent: expected vs actual data

Sources: Forex Factory scraper, Finnhub calendar API
Execution: Every 1 hour + on-demand refresh
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import sqlalchemy as sa

from src.config.settings import get_settings
from src.data.storage.database import get_session

logger = logging.getLogger(__name__)

# Currency → country mapping for calendar filtering
CURRENCY_COUNTRY_MAP: dict[str, str] = {
    "USD": "US",
    "EUR": "EU",
    "GBP": "GB",
    "JPY": "JP",
    "CHF": "CH",
    "AUD": "AU",
    "CAD": "CA",
    "NZD": "NZ",
}

# Table reference
ECON_EVENTS = sa.table(
    "economic_events",
    sa.column("id", sa.Integer),
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("country", sa.String),
    sa.column("event_name", sa.String),
    sa.column("impact", sa.String),
    sa.column("forecast", sa.String),
    sa.column("previous", sa.String),
    sa.column("actual", sa.String),
    sa.column("currency", sa.String),
    sa.column("source", sa.String),
)


class EconomicCalendarFeed:
    """Fetches and stores economic calendar events."""

    def __init__(self) -> None:
        settings = get_settings()
        self._finnhub_key = settings.finnhub_api_key.get_secret_value()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def fetch_finnhub_calendar(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch economic calendar from Finnhub API."""
        if not self._finnhub_key:
            logger.warning("Finnhub API key not configured, skipping calendar fetch")
            return []

        now = datetime.now(timezone.utc)
        from_str = (from_date or now).strftime("%Y-%m-%d")
        to_str = (to_date or now + timedelta(days=7)).strftime("%Y-%m-%d")

        client = await self._get_client()
        try:
            resp = await client.get(
                "https://finnhub.io/api/v1/calendar/economic",
                params={
                    "from": from_str,
                    "to": to_str,
                    "token": self._finnhub_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            events = data.get("economicCalendar", [])
            logger.info("Fetched %d economic events from Finnhub", len(events))
            return events
        except Exception as e:
            logger.error("Finnhub calendar fetch failed", extra={"error": str(e)})
            return []

    async def sync_calendar(self) -> int:
        """Fetch events and upsert into database. Returns count."""
        raw_events = await self.fetch_finnhub_calendar()
        if not raw_events:
            return 0

        rows = []
        for evt in raw_events:
            country = evt.get("country", "")
            if country not in CURRENCY_COUNTRY_MAP.values():
                continue

            # Map impact: Finnhub uses 1/2/3
            impact_map = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
            impact = impact_map.get(evt.get("impact", 1), "LOW")

            rows.append({
                "ts": datetime.fromisoformat(evt["time"]) if "time" in evt else datetime.now(timezone.utc),
                "country": country,
                "event_name": evt.get("event", "Unknown"),
                "impact": impact,
                "forecast": str(evt.get("estimate", "")),
                "previous": str(evt.get("prev", "")),
                "actual": str(evt.get("actual", "")),
                "currency": evt.get("currency", ""),
                "source": "finnhub",
            })

        if not rows:
            return 0

        # Upsert using ON CONFLICT
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(ECON_EVENTS).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="economic_events_ts_country_event_name_key",
            set_={
                "actual": stmt.excluded.actual,
                "forecast": stmt.excluded.forecast,
            },
        )

        async with get_session() as session:
            await session.execute(stmt)

        logger.info("Synced %d economic events", len(rows))
        return len(rows)

    async def get_upcoming_high_impact(
        self, within_minutes: int = 60
    ) -> list[dict[str, Any]]:
        """
        Get HIGH impact events within the next N minutes.
        Used by Risk Manager for news blackout rule.
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=within_minutes)

        query = (
            sa.select(ECON_EVENTS)
            .where(
                sa.and_(
                    ECON_EVENTS.c.impact == "HIGH",
                    ECON_EVENTS.c.ts >= now,
                    ECON_EVENTS.c.ts <= cutoff,
                )
            )
            .order_by(ECON_EVENTS.c.ts.asc())
        )

        async with get_session() as session:
            result = await session.execute(query)
            rows = result.fetchall()

        return [
            {
                "ts": row.ts.isoformat(),
                "country": row.country,
                "event_name": row.event_name,
                "impact": row.impact,
                "currency": row.currency,
                "forecast": row.forecast,
                "previous": row.previous,
            }
            for row in rows
        ]

    async def is_news_blackout(self, blackout_minutes: int = 15) -> bool:
        """
        Check if we're within the news blackout window.
        Returns True if any HIGH impact event is within ±blackout_minutes.
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=blackout_minutes)
        window_end = now + timedelta(minutes=blackout_minutes)

        query = (
            sa.select(sa.func.count())
            .select_from(ECON_EVENTS)
            .where(
                sa.and_(
                    ECON_EVENTS.c.impact == "HIGH",
                    ECON_EVENTS.c.ts >= window_start,
                    ECON_EVENTS.c.ts <= window_end,
                )
            )
        )

        async with get_session() as session:
            result = await session.execute(query)
            count = result.scalar() or 0

        return count > 0

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
