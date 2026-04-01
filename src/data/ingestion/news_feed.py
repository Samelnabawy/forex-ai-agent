"""
News feed ingestion for the Sentiment Agent.

Sources: Finnhub news API, NewsAPI
Stores raw headlines in PostgreSQL for LLM-based sentiment classification.
Publishes breaking news alerts to Redis for real-time reaction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.instruments import INSTRUMENTS
from src.config.settings import get_settings
from src.data.storage.cache import CHANNEL_NEWS_ALERT, publish_event
from src.data.storage.database import get_session

logger = logging.getLogger(__name__)

# Currency keywords for headline → currency mapping
CURRENCY_KEYWORDS: dict[str, list[str]] = {
    "USD": ["fed", "federal reserve", "fomc", "us economy", "nonfarm", "nfp", "treasury", "dollar"],
    "EUR": ["ecb", "european central", "eurozone", "euro area", "lagarde"],
    "GBP": ["boe", "bank of england", "uk economy", "sterling", "pound"],
    "JPY": ["boj", "bank of japan", "yen", "japanese economy", "ueda"],
    "CHF": ["snb", "swiss national", "swiss franc"],
    "AUD": ["rba", "reserve bank of australia", "australian economy", "aussie"],
    "CAD": ["boc", "bank of canada", "canadian economy", "loonie"],
    "NZD": ["rbnz", "reserve bank of new zealand", "kiwi dollar"],
    "XAU": ["gold", "precious metal", "bullion", "safe haven"],
    "WTI": ["oil", "crude", "opec", "petroleum", "barrel", "brent"],
}

NEWS_HEADLINES = sa.table(
    "news_headlines",
    sa.column("ts", sa.DateTime(timezone=True)),
    sa.column("source", sa.String),
    sa.column("headline", sa.String),
    sa.column("url", sa.String),
    sa.column("currencies", sa.ARRAY(sa.String)),
    sa.column("processed", sa.Boolean),
)


def classify_currencies(headline: str) -> list[str]:
    """Map a headline to affected currencies based on keyword matching."""
    lower = headline.lower()
    affected = []
    for currency, keywords in CURRENCY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            affected.append(currency)
    return affected


class NewsFeed:
    """Fetches and stores financial news headlines."""

    def __init__(self) -> None:
        settings = get_settings()
        self._finnhub_key = settings.finnhub_api_key.get_secret_value()
        self._newsapi_key = settings.newsapi_key.get_secret_value()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def fetch_finnhub_news(self, category: str = "forex") -> list[dict[str, Any]]:
        """Fetch latest news from Finnhub."""
        if not self._finnhub_key:
            return []

        client = await self._get_client()
        try:
            resp = await client.get(
                "https://finnhub.io/api/v1/news",
                params={"category": category, "token": self._finnhub_key},
            )
            resp.raise_for_status()
            articles = resp.json()
            return [
                {
                    "ts": datetime.fromtimestamp(a.get("datetime", 0), tz=timezone.utc),
                    "source": a.get("source", "finnhub"),
                    "headline": a.get("headline", ""),
                    "url": a.get("url", ""),
                }
                for a in articles
                if a.get("headline")
            ]
        except Exception as e:
            logger.error("Finnhub news fetch failed", extra={"error": str(e)})
            return []

    async def fetch_newsapi_headlines(self) -> list[dict[str, Any]]:
        """Fetch business headlines from NewsAPI."""
        if not self._newsapi_key:
            return []

        client = await self._get_client()
        try:
            resp = await client.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "category": "business",
                    "language": "en",
                    "pageSize": 50,
                    "apiKey": self._newsapi_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "ts": datetime.fromisoformat(
                        a["publishedAt"].replace("Z", "+00:00")
                    ) if a.get("publishedAt") else datetime.now(timezone.utc),
                    "source": a.get("source", {}).get("name", "newsapi"),
                    "headline": a.get("title", ""),
                    "url": a.get("url", ""),
                }
                for a in data.get("articles", [])
                if a.get("title")
            ]
        except Exception as e:
            logger.error("NewsAPI fetch failed", extra={"error": str(e)})
            return []

    async def sync_news(self) -> int:
        """Fetch from all sources, deduplicate, store. Returns new headline count."""
        all_articles = []
        all_articles.extend(await self.fetch_finnhub_news("forex"))
        all_articles.extend(await self.fetch_finnhub_news("general"))
        all_articles.extend(await self.fetch_newsapi_headlines())

        if not all_articles:
            return 0

        rows = []
        seen_headlines: set[str] = set()
        for article in all_articles:
            headline = article["headline"].strip()
            if not headline or headline in seen_headlines:
                continue
            seen_headlines.add(headline)

            currencies = classify_currencies(headline)
            rows.append({
                "ts": article["ts"],
                "source": article["source"],
                "headline": headline,
                "url": article.get("url", ""),
                "currencies": currencies,
                "processed": False,
            })

        if not rows:
            return 0

        # Insert, skip duplicates (same headline at same time)
        stmt = pg_insert(NEWS_HEADLINES).values(rows)
        stmt = stmt.on_conflict_do_nothing()

        async with get_session() as session:
            result = await session.execute(stmt)
            count = result.rowcount  # type: ignore[union-attr]

        # Publish breaking news for high-relevance headlines
        for row in rows:
            if row["currencies"]:
                await publish_event(CHANNEL_NEWS_ALERT, {
                    "headline": row["headline"],
                    "currencies": row["currencies"],
                    "source": row["source"],
                    "ts": row["ts"].isoformat(),
                })

        logger.info("Synced %d new headlines (of %d fetched)", count, len(rows))
        return count

    async def get_unprocessed(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get headlines not yet processed by the Sentiment Agent."""
        query = (
            sa.select(NEWS_HEADLINES)
            .where(NEWS_HEADLINES.c.processed == False)  # noqa: E712
            .order_by(NEWS_HEADLINES.c.ts.desc())
            .limit(limit)
        )

        async with get_session() as session:
            result = await session.execute(query)
            return [dict(row._mapping) for row in result.fetchall()]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
