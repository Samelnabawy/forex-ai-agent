"""
Macro Analyst Agent — Agent 02

"The Economist" — thinks in economic theory, monetary policy cycles,
and fundamental value. Long-term context, not short-term noise.

Execution: Every 1 hour + triggered on breaking news / economic releases.

Inputs:
  - Bond yields and yield curve (FRED)
  - Rate differentials between economies
  - Real interest rates (TIPS yields)
  - Economic calendar (upcoming + recent releases)
  - News headlines (recent)
  - Commodity prices (oil, iron ore, copper)
  - DXY, VIX (market indices)
  - Session context (NFP day, CB day, etc.)
  - Geopolitical risk assessment

Outputs:
  - Currency scores (-5 to +5) for all 8 currencies
  - Macro regime classification (risk-on/off/neutral/transitioning)
  - Preferred pairs with direction and conviction
  - Gold and oil outlooks
  - Event risk warnings
  - Regime change risk assessment

Decision Authority:
  Sets the macro directional bias. If Macro says "EUR bullish" and Technical says
  "EUR bearish," the Bull/Bear debate must resolve the conflict.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar

from src.agents.base_agent import BaseAgent
from src.agents.geopolitical import GeopoliticalTracker
from src.agents.llm_client import get_llm_client
from src.agents.prompts.macro import (
    CURRENT_VERSION,
    build_event_impact_prompt,
    build_macro_analysis_prompt,
)
from src.config.sessions import get_session_context
from src.data.storage.cache import (
    CHANNEL_NEWS_ALERT,
    get_all_prices,
    publish_event,
)
from src.models import MacroOutlook, MarketState

logger = logging.getLogger(__name__)

# Currencies we score
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]


class MacroAnalystAgent(BaseAgent):
    """
    Agent 02: Macro Analyst

    LLM-powered analysis of global macro conditions.
    Produces currency scores and regime classification.
    """

    name: ClassVar[str] = "macro_analyst"
    description: ClassVar[str] = "Central bank policy, economic data, rate expectations"
    execution_frequency: ClassVar[str] = "1h"

    def __init__(self) -> None:
        super().__init__()
        self._llm = get_llm_client()
        self._geo_tracker = GeopoliticalTracker()
        self._last_regime: str = "neutral"
        self._regime_change_count: int = 0

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """
        Core macro analysis cycle.

        1. Gather all data sources
        2. Build structured prompt
        3. Call Claude Sonnet
        4. Validate and parse response
        5. Check for regime change → publish alert if detected
        6. Return MacroOutlook
        """

        # Step 1: Gather context
        context = await self._gather_context(market_state)

        # Step 2: Build prompt
        system_prompt, user_prompt = build_macro_analysis_prompt(
            yield_curve=context["yield_curve"],
            rate_differentials=context["rate_differentials"],
            real_rates=context["real_rates"],
            economic_data=context["economic_data"],
            commodity_data=context["commodity_data"],
            session_context=context["session_context"],
            recent_news=context["recent_news"],
            dxy=context.get("dxy"),
            vix=context.get("vix"),
            prices=context.get("prices"),
            version=CURRENT_VERSION,
        )

        # Step 3: Call Claude
        raw_result = await self._llm.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=self.name,
            prompt_version=CURRENT_VERSION,
        )

        # Step 4: Validate and enrich
        result = self._validate_and_enrich(raw_result, context)

        # Step 5: Check for regime change
        new_regime = result.get("macro_regime", "neutral")
        if new_regime != self._last_regime and self._last_regime != "neutral":
            self._regime_change_count += 1
            self.logger.warning(
                "MACRO REGIME CHANGE DETECTED",
                extra={
                    "from": self._last_regime,
                    "to": new_regime,
                    "count": self._regime_change_count,
                },
            )
            # Publish regime change alert
            await publish_event("events:regime_change", {
                "from": self._last_regime,
                "to": new_regime,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": self.name,
            })

        self._last_regime = new_regime

        return result

    async def _gather_context(self, market_state: MarketState) -> dict[str, Any]:
        """Gather all data sources for the macro analysis prompt."""
        context: dict[str, Any] = {
            "yield_curve": {},
            "rate_differentials": {},
            "real_rates": {},
            "economic_data": {},
            "commodity_data": {},
            "session_context": {},
            "recent_news": [],
            "dxy": None,
            "vix": None,
            "prices": {},
            "geopolitical": {},
        }

        # Bond yields and rate differentials (FRED)
        try:
            from src.data.ingestion.fred_feed import FREDFeed
            fred = FREDFeed()
            macro_data = await fred.get_full_macro_data()
            context["yield_curve"] = macro_data.get("yield_curve", {})
            context["rate_differentials"] = macro_data.get("rate_differentials", {})
            context["real_rates"] = macro_data.get("real_rates", {})
            context["economic_data"] = macro_data.get("economic_data", {})
            await fred.close()
        except Exception as e:
            self.logger.error("FRED data fetch failed", extra={"error": str(e)})

        # Commodity data
        try:
            from src.data.ingestion.oil_commodity_feed import OilCommodityFeed
            oil_feed = OilCommodityFeed()
            context["commodity_data"] = await oil_feed.get_commodity_context()
            await oil_feed.close()
        except Exception as e:
            self.logger.error("Commodity data fetch failed", extra={"error": str(e)})

        # DXY and VIX
        try:
            from src.data.ingestion.market_feed import MarketDataFeed
            market_feed = MarketDataFeed()
            await market_feed.sync_all()
            context["dxy"] = await market_feed.get_dxy()
            context["vix"] = await market_feed.get_vix()
            await market_feed.close()
        except Exception as e:
            self.logger.error("Market data fetch failed", extra={"error": str(e)})

        # Current prices from Redis
        try:
            context["prices"] = await get_all_prices()
        except Exception as e:
            self.logger.error("Price cache read failed", extra={"error": str(e)})

        # Session context
        session_ctx = get_session_context()
        context["session_context"] = {
            "active_session": session_ctx.active_session.value,
            "day_type": session_ctx.day_type.value,
            "special_notes": session_ctx.special_notes,
            "upcoming_cb_meetings": session_ctx.upcoming_cb_meetings,
            "month_end": session_ctx.month_end_rebalancing,
            "jpn_fiscal_yearend": session_ctx.jpn_fiscal_yearend,
            "aus_fiscal_yearend": session_ctx.aus_fiscal_yearend,
        }

        # Recent news headlines
        try:
            from src.data.ingestion.news_feed import NewsFeed
            news = NewsFeed()
            unprocessed = await news.get_unprocessed(limit=20)
            context["recent_news"] = [h.get("headline", "") for h in unprocessed if h.get("headline")]
            await news.close()
        except Exception as e:
            self.logger.error("News fetch failed", extra={"error": str(e)})

        # Geopolitical risk
        try:
            geo_result = await self._geo_tracker.assess_risk(context["recent_news"])
            context["geopolitical"] = geo_result
        except Exception as e:
            self.logger.error("Geopolitical assessment failed", extra={"error": str(e)})

        return context

    def _validate_and_enrich(
        self, raw: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate LLM output and add metadata."""

        # Handle fallback case
        if raw.get("_fallback") or raw.get("_no_cache"):
            return self._build_fallback_output(context)

        # Validate currency scores are in range
        scores = raw.get("currency_scores", {})
        for currency in CURRENCIES:
            if currency not in scores:
                scores[currency] = 0
            score = scores[currency]
            if not isinstance(score, (int, float)) or score < -5 or score > 5:
                scores[currency] = 0
        raw["currency_scores"] = scores

        # Validate macro regime
        valid_regimes = {"risk_on", "risk_off", "neutral", "transitioning"}
        if raw.get("macro_regime") not in valid_regimes:
            raw["macro_regime"] = "neutral"

        # Add metadata
        raw["timestamp"] = datetime.now(timezone.utc).isoformat()
        raw["agent"] = self.name
        raw["prompt_version"] = CURRENT_VERSION
        raw["data_quality"] = self._assess_data_quality(context)

        # Add geopolitical context
        geo = context.get("geopolitical", {})
        raw["geopolitical_risk_score"] = geo.get("overall_risk_score", 0)
        raw["geopolitical_risk_level"] = geo.get("risk_level", "calm")

        # Add session awareness
        session = context.get("session_context", {})
        raw["session"] = session.get("active_session", "unknown")
        raw["day_type"] = session.get("day_type", "normal")

        # Derive pair preferences from currency scores if not provided
        if not raw.get("preferred_pairs"):
            raw["preferred_pairs"] = self._derive_preferred_pairs(scores)

        return raw

    def _derive_preferred_pairs(
        self, scores: dict[str, int | float]
    ) -> list[dict[str, Any]]:
        """Derive preferred pairs from currency scores."""
        pairs = []
        pair_map = {
            "EUR/USD": ("EUR", "USD"),
            "GBP/USD": ("GBP", "USD"),
            "USD/JPY": ("USD", "JPY"),
            "USD/CHF": ("USD", "CHF"),
            "AUD/USD": ("AUD", "USD"),
            "USD/CAD": ("USD", "CAD"),
            "NZD/USD": ("NZD", "USD"),
        }

        for pair, (base, quote) in pair_map.items():
            base_score = scores.get(base, 0)
            quote_score = scores.get(quote, 0)
            diff = base_score - quote_score

            if abs(diff) < 2:
                continue  # Not enough divergence

            direction = "LONG" if diff > 0 else "SHORT"
            conviction = "high" if abs(diff) >= 4 else "medium" if abs(diff) >= 3 else "low"

            pairs.append({
                "pair": pair,
                "direction": direction,
                "conviction": conviction,
                "score_differential": diff,
            })

        # Sort by absolute score differential
        pairs.sort(key=lambda p: abs(p["score_differential"]), reverse=True)
        return pairs[:5]  # Top 5 pairs

    def _assess_data_quality(self, context: dict[str, Any]) -> dict[str, str]:
        """Assess quality of each data source."""
        quality: dict[str, str] = {}

        quality["yields"] = "good" if context.get("yield_curve", {}).get("yields") else "missing"
        quality["rates"] = "good" if context.get("rate_differentials") else "missing"
        quality["real_rates"] = "good" if any(
            v is not None for v in context.get("real_rates", {}).values()
        ) else "missing"
        quality["commodities"] = "good" if context.get("commodity_data", {}).get("oil") else "missing"
        quality["dxy"] = "good" if context.get("dxy") else "missing"
        quality["vix"] = "good" if context.get("vix") else "missing"
        quality["prices"] = "good" if context.get("prices") else "missing"
        quality["news"] = "good" if context.get("recent_news") else "sparse"

        return quality

    def _build_fallback_output(self, context: dict[str, Any]) -> dict[str, Any]:
        """Build a conservative output when LLM is unavailable."""
        self.logger.warning("Using fallback macro output — LLM unavailable")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self.name,
            "macro_regime": "neutral",
            "regime_confidence": 0.3,
            "usd_outlook": "neutral",
            "reasoning": "LLM unavailable — defaulting to neutral stance",
            "currency_scores": {c: 0 for c in CURRENCIES},
            "preferred_pairs": [],
            "gold_outlook": "neutral",
            "oil_outlook": "neutral",
            "risk_events": [],
            "_fallback": True,
            "data_quality": self._assess_data_quality(context),
        }

    # ── Event-Triggered Analysis ──────────────────────────────

    async def assess_event_impact(
        self,
        event_name: str,
        actual: str,
        forecast: str,
        previous: str,
        currency: str,
    ) -> dict[str, Any]:
        """
        Quick event impact assessment using Haiku.
        Called when a high-impact economic release occurs.
        """
        system, user = build_event_impact_prompt(
            event_name=event_name,
            actual=actual,
            forecast=forecast,
            previous=previous,
            currency=currency,
        )

        result = await self._llm.classify(
            system_prompt=system,
            user_prompt=user,
            agent_name=f"{self.name}:event_impact",
            prompt_version="v1",
        )

        self.logger.info(
            "Event impact assessed",
            extra={
                "event": event_name,
                "surprise": result.get("surprise", "unknown"),
                "magnitude": result.get("magnitude", "unknown"),
            },
        )

        return result

    @property
    def stats(self) -> dict[str, Any]:
        base = super().stats
        base["last_regime"] = self._last_regime
        base["regime_changes"] = self._regime_change_count
        base["llm_usage"] = self._llm.usage
        return base
