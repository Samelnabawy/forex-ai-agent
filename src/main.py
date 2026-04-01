"""
Application entry point.
Starts all services: data feeds, candle aggregator, agent scheduler, and API.

Service lifecycle:
  1. Verify risk rules integrity
  2. Connect to Postgres + Redis
  3. Start data feeds (price, news, calendar)
  4. Start supplementary feeds (FRED, DXY/VIX, oil/commodities)
  5. Start agent scheduler (Technical 5m, Macro 1h)
  6. Start FastAPI dashboard
  7. Wait for shutdown signal
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime, timezone
from typing import Any

from src.config import get_settings, validate_rules_integrity
from src.data.ingestion.candle_aggregator import CandleAggregator
from src.data.ingestion.calendar_feed import EconomicCalendarFeed
from src.data.ingestion.fred_feed import FREDFeed
from src.data.ingestion.market_feed import MarketDataFeed
from src.data.ingestion.news_feed import NewsFeed
from src.data.ingestion.oil_commodity_feed import OilCommodityFeed
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

        # Data feeds
        self._price_feed: PriceFeedManager | None = None
        self._candle_aggregator: CandleAggregator | None = None
        self._calendar_feed: EconomicCalendarFeed | None = None
        self._news_feed: NewsFeed | None = None

        # Supplementary feeds (shared across agents)
        self._fred_feed: FREDFeed | None = None
        self._market_feed: MarketDataFeed | None = None
        self._oil_feed: OilCommodityFeed | None = None

        # Agents
        self._technical_agent: Any = None
        self._macro_agent: Any = None
        self._correlation_agent: Any = None

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

        # ── Data Feeds ────────────────────────────────────────

        # Price feed (WebSocket)
        if self.settings.polygon_api_key.get_secret_value():
            provider = PolygonProvider()
        elif self.settings.twelve_data_api_key.get_secret_value():
            provider = TwelveDataProvider()
        else:
            logger.warning("No price data provider configured — price feed disabled")
            provider = None

        if provider:
            self._price_feed = PriceFeedManager(provider)
            self._tasks.append(asyncio.create_task(self._price_feed.start()))

        self._candle_aggregator = CandleAggregator()
        self._tasks.append(asyncio.create_task(self._candle_aggregator.start()))

        self._calendar_feed = EconomicCalendarFeed()
        self._tasks.append(asyncio.create_task(self._loop("calendar", self._sync_calendar, 3600)))

        self._news_feed = NewsFeed()
        self._tasks.append(asyncio.create_task(self._loop("news", self._sync_news, 900)))

        # ── Supplementary Feeds (shared with agents) ──────────

        self._fred_feed = FREDFeed()
        self._tasks.append(asyncio.create_task(self._loop("fred", self._sync_fred, 3600)))

        self._market_feed = MarketDataFeed()
        self._tasks.append(asyncio.create_task(self._loop("market_data", self._sync_market_data, 300)))

        self._oil_feed = OilCommodityFeed()
        self._tasks.append(asyncio.create_task(self._loop("oil_commodity", self._sync_oil, 3600)))

        # ── Agents ────────────────────────────────────────────

        # Technical Analyst — runs every 5 minutes
        from src.agents.technical import TechnicalAnalystAgent
        self._technical_agent = TechnicalAnalystAgent()
        self._tasks.append(asyncio.create_task(
            self._loop("technical_analyst", self._run_technical_agent, 300)
        ))

        # Macro Analyst — runs every 1 hour, receives shared feed instances
        from src.agents.macro import MacroAnalystAgent
        self._macro_agent = MacroAnalystAgent(
            fred_feed=self._fred_feed,
            oil_feed=self._oil_feed,
            market_feed=self._market_feed,
            news_feed=self._news_feed,
        )
        self._tasks.append(asyncio.create_task(
            self._loop("macro_analyst", self._run_macro_agent, 3600)
        ))

        # Correlation Agent — runs every 1 minute (lightweight math, no LLM)
        from src.agents.correlation import CorrelationAgent
        self._correlation_agent = CorrelationAgent()
        self._tasks.append(asyncio.create_task(
            self._loop("correlation_agent", self._run_correlation_agent, 60)
        ))

        # ── API Server ────────────────────────────────────────

        self._tasks.append(asyncio.create_task(self._start_api()))

        logger.info("All services started — waiting for shutdown signal")
        await self._shutdown_event.wait()

    # ── Generic Loop Runner ───────────────────────────────────

    async def _loop(self, name: str, func: Any, interval_seconds: int) -> None:
        """Generic service loop with error handling."""
        # Small initial delay to let infrastructure settle
        await asyncio.sleep(2)
        while not self._shutdown_event.is_set():
            try:
                await func()
            except Exception as e:
                logger.error(f"{name} loop error", extra={"error": str(e)})
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=interval_seconds
                )
                break  # Shutdown was signaled
            except asyncio.TimeoutError:
                pass  # Normal — interval elapsed, run again

    # ── Feed Sync Functions ───────────────────────────────────

    async def _sync_calendar(self) -> None:
        assert self._calendar_feed is not None
        count = await self._calendar_feed.sync_calendar()
        logger.info("Calendar sync: %d events", count)

    async def _sync_news(self) -> None:
        assert self._news_feed is not None
        count = await self._news_feed.sync_news()
        logger.info("News sync: %d headlines", count)

    async def _sync_fred(self) -> None:
        assert self._fred_feed is not None
        data = await self._fred_feed.get_bond_yields()
        non_null = sum(1 for v in data.values() if v is not None)
        logger.info("FRED sync: %d/%d yields fetched", non_null, len(data))

    async def _sync_market_data(self) -> None:
        assert self._market_feed is not None
        results = await self._market_feed.sync_all()
        logger.info("Market data sync: %d instruments", len(results))

    async def _sync_oil(self) -> None:
        assert self._oil_feed is not None
        data = await self._oil_feed.get_oil_data()
        logger.info("Oil/Commodity sync: WTI=%s", data.get("wti_price"))

    # ── Agent Execution Functions ─────────────────────────────

    async def _run_technical_agent(self) -> None:
        """Run Technical Analyst on all instruments."""
        from src.orchestration.market_state_builder import build_market_state

        market_state = await build_market_state()
        result = await self._technical_agent.run(market_state)

        signals = result.get("signals_generated", 0)
        if signals > 0:
            logger.info(
                "Technical Agent: %d signals generated",
                signals,
                extra={"signals": result.get("signals", [])},
            )

    async def _run_macro_agent(self) -> None:
        """Run Macro Analyst."""
        from src.orchestration.market_state_builder import build_market_state

        market_state = await build_market_state()
        result = await self._macro_agent.run(market_state)

        regime = result.get("macro_regime", "unknown")
        logger.info(
            "Macro Agent: regime=%s",
            regime,
            extra={
                "currency_scores": result.get("currency_scores", {}),
                "preferred_pairs": result.get("preferred_pairs", []),
            },
        )

    async def _run_correlation_agent(self) -> None:
        """Run Correlation Agent."""
        from src.orchestration.market_state_builder import build_market_state

        market_state = await build_market_state()
        result = await self._correlation_agent.run(market_state)

        cascades = result.get("active_cascades", [])
        anomalies = result.get("anomalies", [])
        regime = result.get("market_regime", {}).get("regime", "unknown")

        if cascades or anomalies:
            logger.info(
                "Correlation Agent: regime=%s, cascades=%d, anomalies=%d",
                regime, len(cascades), len(anomalies),
            )

    # ── API Server ────────────────────────────────────────────

    async def _start_api(self) -> None:
        import uvicorn
        config = uvicorn.Config(
            "src.monitoring.dashboard:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    # ── Shutdown ──────────────────────────────────────────────

    async def stop(self) -> None:
        """Graceful shutdown of all services."""
        logger.info("Shutdown initiated")

        # Stop data feeds
        if self._price_feed:
            await self._price_feed.stop()
        if self._candle_aggregator:
            await self._candle_aggregator.stop()

        # Close HTTP clients
        for feed in [self._calendar_feed, self._news_feed, self._fred_feed,
                     self._market_feed, self._oil_feed]:
            if feed and hasattr(feed, 'close'):
                await feed.close()

        # Cancel remaining tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        await close_redis()
        await close_db()

        logger.info("Shutdown complete")


def main() -> None:
    """CLI entry point."""
    app = Application()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

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
