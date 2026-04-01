"""
Application entry point.
Starts all services: data feeds, candle aggregator, agent scheduler, and API.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from src.config import get_settings, validate_rules_integrity
from src.data.ingestion.candle_aggregator import CandleAggregator
from src.data.ingestion.calendar_feed import EconomicCalendarFeed
from src.data.ingestion.news_feed import NewsFeed
from src.data.ingestion.price_feed import PriceFeedManager, TwelveDataProvider, PolygonProvider
from src.data.storage.cache import init_redis, close_redis
from src.data.storage.database import init_db, close_db
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


class Application:
    """Main application orchestrator — manages all service lifecycles."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Services
        self._price_feed: PriceFeedManager | None = None
        self._candle_aggregator: CandleAggregator | None = None
        self._calendar_feed: EconomicCalendarFeed | None = None
        self._news_feed: NewsFeed | None = None

    async def start(self) -> None:
        """Initialize all services and start the event loop."""
        setup_logging()
        logger.info("=" * 60)
        logger.info("FOREX AI TRADING AGENT — Starting")
        logger.info("=" * 60)

        # Validate risk rules on every startup
        validate_rules_integrity()
        logger.info("Risk rules integrity: VERIFIED")
        logger.info("Paper trading: %s", self.settings.paper_trading)

        # Connect to databases
        await init_db()
        await init_redis()

        # Select price provider based on available API keys
        if self.settings.polygon_api_key.get_secret_value():
            provider = PolygonProvider()
        elif self.settings.twelve_data_api_key.get_secret_value():
            provider = TwelveDataProvider()
        else:
            logger.warning("No price data provider configured — price feed disabled")
            provider = None

        # Start services
        if provider:
            self._price_feed = PriceFeedManager(provider)
            self._tasks.append(asyncio.create_task(self._price_feed.start()))

        self._candle_aggregator = CandleAggregator()
        self._tasks.append(asyncio.create_task(self._candle_aggregator.start()))

        self._calendar_feed = EconomicCalendarFeed()
        self._tasks.append(asyncio.create_task(self._calendar_sync_loop()))

        self._news_feed = NewsFeed()
        self._tasks.append(asyncio.create_task(self._news_sync_loop()))

        # Start FastAPI via uvicorn in a separate task
        self._tasks.append(asyncio.create_task(self._start_api()))

        logger.info("All services started — waiting for shutdown signal")
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Graceful shutdown of all services."""
        logger.info("Shutdown initiated")

        if self._price_feed:
            await self._price_feed.stop()
        if self._candle_aggregator:
            await self._candle_aggregator.stop()
        if self._calendar_feed:
            await self._calendar_feed.close()
        if self._news_feed:
            await self._news_feed.close()

        # Cancel remaining tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        await close_redis()
        await close_db()

        logger.info("Shutdown complete")

    async def _calendar_sync_loop(self) -> None:
        """Sync economic calendar every hour."""
        while not self._shutdown_event.is_set():
            try:
                assert self._calendar_feed is not None
                count = await self._calendar_feed.sync_calendar()
                logger.info("Calendar sync: %d events", count)
            except Exception as e:
                logger.error("Calendar sync failed", extra={"error": str(e)})
            await asyncio.sleep(3600)  # 1 hour

    async def _news_sync_loop(self) -> None:
        """Sync news headlines every 15 minutes."""
        while not self._shutdown_event.is_set():
            try:
                assert self._news_feed is not None
                count = await self._news_feed.sync_news()
                logger.info("News sync: %d headlines", count)
            except Exception as e:
                logger.error("News sync failed", extra={"error": str(e)})
            await asyncio.sleep(900)  # 15 minutes

    async def _start_api(self) -> None:
        """Start the FastAPI server."""
        import uvicorn

        config = uvicorn.Config(
            "src.monitoring.dashboard:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()


def main() -> None:
    """CLI entry point."""
    app = Application()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle SIGTERM/SIGINT for graceful shutdown
    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        app._shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(app.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(app.stop())
        loop.close()


if __name__ == "__main__":
    main()
