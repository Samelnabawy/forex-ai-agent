"""
Sentiment Agent — Agent 04

"The Street Listener" — reads between the lines, detects fear and greed,
knows what the crowd is thinking AND where they're about to be wrong.

10 Analytical Layers:
  1. News Sentiment (LLM — Haiku batch classification)
  2. COT Positioning Intelligence (speculator/commercial decomposition)
  3. Economic Surprise Index (decay-weighted actual vs forecast)
  4. Sentiment-Price Divergence (the contrarian reversal signal)
  5. Central Bank Language Delta (statement diffing for hawkish/dovish shifts)
  6. Attention Model (what the market cares about RIGHT NOW)
  7. Intermarket Signals (credit spreads, Cu/Au, CNH)
  8. Seasonal Patterns (NFP compression, month-end, Golden Week)
  9. Reflexivity Detection (self-reinforcing loops about to break)
  10. Feedback Loop (self-improving hit rate tracking)

Execution: Every 30 minutes + triggered on breaking news.

Decision Authority:
  Can elevate or reduce confidence scores from other agents.
  If sentiment is extreme (fear or greed), it flags potential reversals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import get_llm_client
from src.agents.prompts.sentiment import (
    build_cb_delta_prompt,
    build_headline_classification_prompt,
    build_narrative_detection_prompt,
    build_event_scenario_prompt,
)
from src.agents.sentiment_models import (
    AttentionRegime,
    COTDecomposition,
    FearGreedIndex,
    FeedbackTracker,
    ReflexivitySignal,
    SentimentPriceDivergence,
    SurpriseIndex,
    analyze_cot_positioning,
    classify_attention_regime,
    compute_fear_greed,
    compute_sentiment_decay,
    compute_surprise_index,
    detect_reflexivity,
    detect_seasonal_patterns,
    detect_sentiment_price_divergence,
)
from src.config.instruments import ALL_SYMBOLS, INSTRUMENTS
from src.data.storage.cache import get_agent_state, get_all_prices
from src.models import MarketState

logger = logging.getLogger(__name__)

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]


class SentimentAgent(BaseAgent):
    """
    Agent 04: Sentiment Agent

    Orchestrates 10 analytical layers to produce a composite
    sentiment picture with contrarian signal detection.
    """

    name: ClassVar[str] = "sentiment_agent"
    description: ClassVar[str] = "News NLP, COT positioning, fear/greed, contrarian signals"
    execution_frequency: ClassVar[str] = "30m"

    def __init__(
        self,
        news_feed: Any | None = None,
        cb_feed: Any | None = None,
    ) -> None:
        super().__init__()
        self._llm = get_llm_client()
        self._news_feed = news_feed
        self._cb_feed = cb_feed
        self._feedback = FeedbackTracker()
        # Rolling state
        self._headline_sentiment_history: list[tuple[float, datetime]] = []
        self._last_narrative_analysis: dict[str, Any] = {}

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """
        Run all 10 sentiment layers and produce composite output.
        """
        now = datetime.now(timezone.utc)
        result: dict[str, Any] = {"timestamp": now.isoformat(), "agent": self.name}

        # Gather inputs
        headlines = await self._get_recent_headlines()
        prices = self._get_current_prices(market_state)
        vix = await self._get_vix()
        macro_state = await get_agent_state("macro_analyst")

        # ── Layer 1: News Sentiment ───────────────────────────
        news_sentiment = await self._layer_news_sentiment(headlines)
        result["news_sentiment"] = news_sentiment

        # ── Layer 2: COT Positioning ──────────────────────────
        cot_analysis = await self._layer_cot_positioning()
        result["cot_analysis"] = cot_analysis

        # ── Layer 3: Economic Surprise Index ──────────────────
        surprise_indices = await self._layer_economic_surprise()
        result["economic_surprise"] = surprise_indices

        # ── Layer 4: Sentiment-Price Divergence ───────────────
        divergences = self._layer_divergence(
            prices, news_sentiment, cot_analysis, surprise_indices, market_state
        )
        result["divergences"] = divergences

        # ── Layer 5: Central Bank Language Delta ──────────────
        cb_deltas = await self._layer_cb_delta()
        result["central_bank_deltas"] = cb_deltas

        # ── Layer 6: Attention Model ──────────────────────────
        headline_texts = [h.get("headline", "") for h in headlines]
        geo_state = await get_agent_state("geopolitical_tracker")
        geo_score = geo_state.get("overall_risk_score", 0) if geo_state else 0
        attention = classify_attention_regime(headline_texts, vix, geo_score)
        result["attention"] = {
            "regime": attention.regime.value,
            "confidence": attention.confidence,
            "what_matters": attention.what_matters_now,
            "what_is_noise": attention.what_is_noise,
        }

        # ── Layer 7: Intermarket Signals ──────────────────────
        intermarket = await self._layer_intermarket(vix)
        result["intermarket"] = intermarket

        # ── Layer 8: Seasonal Patterns ────────────────────────
        seasonals = detect_seasonal_patterns(now)
        result["seasonal_patterns"] = [
            {
                "pattern": s.pattern_name,
                "instruments": s.affected_instruments,
                "effect": s.expected_effect,
                "confidence": s.confidence,
            }
            for s in seasonals
        ]

        # ── Layer 9: Reflexivity Detection ────────────────────
        reflexivity = self._layer_reflexivity(market_state, macro_state)
        result["reflexivity"] = reflexivity

        # ── Layer 10: Feedback Loop ───────────────────────────
        self._feedback.update_weights()
        result["feedback"] = {
            "source_weights": self._feedback.weights,
            "hit_rates": self._feedback.get_hit_rates(window_days=30),
        }

        # ── Composite Fear & Greed Index ──────────────────────
        fear_greed = self._compute_composite(
            news_sentiment, cot_analysis, surprise_indices,
            vix, divergences, intermarket,
        )
        result["fear_greed_index"] = {
            "score": fear_greed.score,
            "label": fear_greed.label,
            "components": fear_greed.components,
            "contrarian_signal": fear_greed.contrarian_signal,
            "active_divergences": fear_greed.active_divergences,
        }

        # ── Per-Currency Sentiment Summary ────────────────────
        result["currency_sentiment"] = self._build_currency_sentiment(
            news_sentiment, cot_analysis, surprise_indices, cb_deltas
        )

        # ── Confidence ────────────────────────────────────────
        result["confidence"] = round(min(
            0.3 + len(headlines) / 50 +
            (0.2 if cot_analysis else 0) +
            (0.1 if surprise_indices else 0),
            0.95,
        ), 2)

        return result

    # ── Layer 1: News Sentiment ───────────────────────────────

    async def _layer_news_sentiment(
        self, headlines: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Batch classify headlines via Haiku and compute decay-weighted scores."""
        if not headlines:
            return {"currency_scores": {}, "narrative": "", "consensus_risk": "none"}

        # Batch classify via LLM
        system, user = build_headline_classification_prompt(headlines)
        raw = await self._llm.classify(
            system_prompt=system,
            user_prompt=user,
            agent_name=f"{self.name}:headlines",
            prompt_version="v1",
        )

        if raw.get("_fallback"):
            return {"currency_scores": {}, "narrative": "", "consensus_risk": "none"}

        # Aggregate per-currency scores with decay weighting
        currency_scores: dict[str, list[tuple[float, datetime]]] = {c: [] for c in CURRENCIES}
        now = datetime.now(timezone.utc)

        for classification in raw.get("classifications", []):
            scores = classification.get("scores", {})
            credibility = classification.get("credibility", "medium")
            cred_multiplier = {"high": 1.5, "medium": 1.0, "low": 0.5}.get(credibility, 1.0)

            for currency, score in scores.items():
                if currency in currency_scores:
                    currency_scores[currency].append((score * cred_multiplier, now))

        # Decay-weighted averages
        decayed_scores: dict[str, float] = {}
        for currency, score_list in currency_scores.items():
            all_scores = score_list + [
                (s, t) for s, t in self._headline_sentiment_history
            ]
            if all_scores:
                decayed_scores[currency] = round(compute_sentiment_decay(all_scores, half_life_hours=4.0), 2)
            else:
                decayed_scores[currency] = 0

        # Store for next iteration's decay
        for currency, score_list in currency_scores.items():
            self._headline_sentiment_history.extend(score_list)
        # Trim history
        cutoff = now.timestamp() - 24 * 3600
        self._headline_sentiment_history = [
            (s, t) for s, t in self._headline_sentiment_history
            if t.timestamp() > cutoff
        ]

        return {
            "currency_scores": decayed_scores,
            "narrative": raw.get("overall_narrative", ""),
            "narrative_consistency": raw.get("narrative_consistency", 0),
            "consensus_risk": raw.get("consensus_risk", "none"),
            "headlines_processed": len(raw.get("classifications", [])),
        }

    # ── Layer 2: COT Positioning ──────────────────────────────

    async def _layer_cot_positioning(self) -> dict[str, Any]:
        """Analyze COT data with speculator/commercial decomposition."""
        try:
            from src.data.ingestion.cot_feed import COTFeed
            cot = COTFeed()
            latest = await cot.fetch_latest_cot()
            await cot.close()
        except Exception as e:
            logger.error("COT fetch failed", extra={"error": str(e)})
            return {}

        if not latest:
            return {}

        results: dict[str, Any] = {}
        for position_data in latest:
            instrument = position_data.get("instrument", "")
            # Simplified: use current data as both current and history placeholder
            decomp = analyze_cot_positioning(position_data, [position_data])
            results[instrument] = {
                "speculator_net": decomp.speculator_net,
                "speculator_percentile": decomp.speculator_percentile,
                "positioning_extreme": decomp.positioning_extreme.value,
                "crowded_trade": decomp.crowded_trade,
                "reversal_probability": round(decomp.reversal_probability, 2),
                "contrarian_signal": decomp.contrarian_signal,
                "momentum": decomp.speculator_momentum,
                "reasoning": decomp.reasoning,
            }

        return results

    # ── Layer 3: Economic Surprise ────────────────────────────

    async def _layer_economic_surprise(self) -> dict[str, Any]:
        """Compute economic surprise index per economy."""
        try:
            import sqlalchemy as sa
            from src.data.storage.database import get_session

            econ_table = sa.table(
                "economic_events",
                sa.column("ts", sa.DateTime(timezone=True)),
                sa.column("country", sa.String),
                sa.column("event_name", sa.String),
                sa.column("actual", sa.String),
                sa.column("forecast", sa.String),
                sa.column("previous", sa.String),
            )

            query = (
                sa.select(econ_table)
                .where(econ_table.c.actual.isnot(None))
                .where(econ_table.c.actual != "")
                .order_by(econ_table.c.ts.desc())
                .limit(200)
            )

            async with get_session() as session:
                result = await session.execute(query)
                rows = result.fetchall()

            # Group by country
            by_country: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                country = row.country
                by_country.setdefault(country, []).append({
                    "event_name": row.event_name,
                    "actual": row.actual,
                    "forecast": row.forecast,
                    "previous": row.previous,
                    "ts": row.ts,
                })

            indices: dict[str, Any] = {}
            for country, releases in by_country.items():
                idx = compute_surprise_index(releases, country)
                indices[country] = {
                    "score": idx.score,
                    "trend": idx.trend,
                    "momentum_shift": idx.momentum_shift,
                    "streak": idx.streak,
                }

            return indices

        except Exception as e:
            logger.error("Surprise index computation failed", extra={"error": str(e)})
            return {}

    # ── Layer 4: Divergence ───────────────────────────────────

    def _layer_divergence(
        self,
        prices: dict[str, float],
        news_sentiment: dict[str, Any],
        cot_analysis: dict[str, Any],
        surprise_indices: dict[str, Any],
        market_state: MarketState,
    ) -> list[dict[str, Any]]:
        """Detect sentiment-price divergences across all instruments."""
        divergences: list[dict[str, Any]] = []
        news_scores = news_sentiment.get("currency_scores", {})

        for symbol in ALL_SYMBOLS:
            inst = INSTRUMENTS.get(symbol)
            if not inst:
                continue

            # Determine price trend from candles
            candles = market_state.candles.get(symbol, {}).get("1h", [])
            price_trend = "flat"
            if len(candles) >= 10:
                recent = [float(c.close if hasattr(c, "close") else c.get("close", 0)) for c in candles[-10:]]
                if recent[-1] > recent[0] * 1.002:
                    price_trend = "up"
                elif recent[-1] < recent[0] * 0.998:
                    price_trend = "down"

            # Get sentiment inputs for this instrument's base currency
            base_currency = inst.base_currency
            news_score = news_scores.get(base_currency, 0)

            cot_data = cot_analysis.get(symbol, {})
            cot_momentum = cot_data.get("momentum", "flat")

            # Map country to surprise trend
            country_map = {"USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP",
                           "AUD": "AU", "CAD": "CA", "CHF": "CH", "NZD": "NZ"}
            country = country_map.get(base_currency, "")
            surprise_data = surprise_indices.get(country, {})
            surprise_trend = surprise_data.get("trend", "flat")

            div = detect_sentiment_price_divergence(
                instrument=symbol,
                price_trend=price_trend,
                news_sentiment=news_score,
                cot_momentum=cot_momentum,
                surprise_trend=surprise_trend,
            )

            if div:
                divergences.append({
                    "instrument": div.instrument,
                    "type": div.divergence_type.value,
                    "direction": div.direction,
                    "severity": div.severity,
                    "components": div.components,
                    "reasoning": div.reasoning,
                })

                # Record prediction for feedback loop
                prediction = "bearish" if div.direction == "bearish_divergence" else "bullish"
                price = prices.get(symbol, 0)
                if price:
                    self._feedback.record_prediction(symbol, prediction, "divergence", div.severity / 100, price)

        return divergences

    # ── Layer 5: Central Bank Delta ───────────────────────────

    async def _layer_cb_delta(self) -> list[dict[str, Any]]:
        """Analyze language changes in central bank statements."""
        if not self._cb_feed:
            return []

        deltas: list[dict[str, Any]] = []

        for bank, config in [("Fed", "USD"), ("ECB", "EUR"), ("BOE", "GBP"), ("BOJ", "JPY")]:
            try:
                previous, current = await self._cb_feed.get_statement_pair_for_diff(bank)
                if not previous or not current:
                    continue

                system, user = build_cb_delta_prompt(
                    bank=bank,
                    currency=config,
                    previous_statement=previous["statement"],
                    current_statement=current["statement"],
                    previous_date=previous["ts"],
                    current_date=current["ts"],
                )

                # Use Sonnet for CB analysis (nuanced language)
                result = await self._llm.analyze(
                    system_prompt=system,
                    user_prompt=user,
                    agent_name=f"{self.name}:cb_delta:{bank}",
                    prompt_version="v1",
                )

                if not result.get("_fallback"):
                    deltas.append(result)

            except Exception as e:
                logger.error("CB delta failed for %s", bank, extra={"error": str(e)})

        return deltas

    # ── Layer 7: Intermarket ──────────────────────────────────

    async def _layer_intermarket(self, vix: float | None) -> dict[str, Any]:
        """Get intermarket sentiment signals."""
        try:
            from src.data.ingestion.intermarket_feed import compute_intermarket_signals
            signals = await compute_intermarket_signals(vix=vix)
            return {
                "risk_score": signals.risk_score,
                "copper_gold_signal": signals.copper_gold_signal,
                "credit_signal": signals.credit_signal,
                "credit_vix_divergence": signals.credit_vix_divergence,
                "signals": signals.signals,
            }
        except Exception as e:
            logger.error("Intermarket signals failed", extra={"error": str(e)})
            return {"risk_score": 50, "signals": []}

    # ── Layer 9: Reflexivity ──────────────────────────────────

    def _layer_reflexivity(
        self, market_state: MarketState, macro_state: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Detect self-reinforcing loops across instruments."""
        results: list[dict[str, Any]] = []
        currency_scores = (macro_state or {}).get("currency_scores", {})

        for symbol in ALL_SYMBOLS:
            candles = market_state.candles.get(symbol, {}).get("1d", [])
            if len(candles) < 30:
                continue

            closes = [float(c.close if hasattr(c, "close") else c.get("close", 0)) for c in candles]
            if not closes or closes[-30] == 0:
                continue

            price_30d = ((closes[-1] - closes[-30]) / closes[-30]) * 100
            price_7d = ((closes[-1] - closes[-7]) / closes[-7]) * 100 if len(closes) >= 7 and closes[-7] != 0 else 0
            price_1d = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) >= 2 and closes[-2] != 0 else 0

            inst = INSTRUMENTS.get(symbol)
            base = inst.base_currency if inst else ""
            score = currency_scores.get(base, 0)
            fundamental_dir = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"

            signal = detect_reflexivity(
                instrument=symbol,
                price_change_30d_pct=price_30d,
                price_change_7d_pct=price_7d,
                price_change_1d_pct=price_1d,
                fundamental_direction=fundamental_dir,
                cot_momentum="flat",  # Would come from COT layer
            )

            if signal:
                results.append({
                    "instrument": symbol,
                    "loop_type": signal.loop_type,
                    "phase": signal.phase,
                    "strength": signal.strength,
                    "fundamental_support": signal.fundamental_support,
                    "reasoning": signal.reasoning,
                })

        return results

    # ── Composite Computation ─────────────────────────────────

    def _compute_composite(
        self,
        news_sentiment: dict[str, Any],
        cot_analysis: dict[str, Any],
        surprise_indices: dict[str, Any],
        vix: float | None,
        divergences: list[dict[str, Any]],
        intermarket: dict[str, Any],
    ) -> FearGreedIndex:
        """Compute the composite Fear & Greed Index."""
        # Average news sentiment across currencies
        news_scores = news_sentiment.get("currency_scores", {})
        news_avg = sum(news_scores.values()) / len(news_scores) if news_scores else 0

        # Average COT percentile
        cot_pcts = [
            v.get("speculator_percentile", 50) for v in cot_analysis.values()
            if isinstance(v, dict)
        ]
        cot_avg = sum(cot_pcts) / len(cot_pcts) if cot_pcts else 50

        # Average surprise score
        surprise_scores = [
            v.get("score", 0) for v in surprise_indices.values()
            if isinstance(v, dict)
        ]
        surprise_avg = sum(surprise_scores) / len(surprise_scores) if surprise_scores else 0

        intermarket_risk = intermarket.get("risk_score", 50)

        return compute_fear_greed(
            news_sentiment_avg=news_avg,
            cot_percentile_avg=cot_avg,
            surprise_score_avg=surprise_avg,
            vix=vix,
            divergence_count=len(divergences),
            intermarket_risk_score=intermarket_risk,
            source_weights=self._feedback.weights,
        )

    # ── Per-Currency Summary ──────────────────────────────────

    def _build_currency_sentiment(
        self,
        news: dict[str, Any],
        cot: dict[str, Any],
        surprise: dict[str, Any],
        cb_deltas: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Build per-currency sentiment summary."""
        country_map = {"USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP",
                       "AUD": "AU", "CAD": "CA", "CHF": "CH", "NZD": "NZ"}
        sentiment: dict[str, dict[str, Any]] = {}

        for currency in CURRENCIES:
            entry: dict[str, Any] = {"score": 0, "narrative": ""}

            # News score
            news_score = news.get("currency_scores", {}).get(currency, 0)
            entry["news_score"] = news_score

            # Surprise
            country = country_map.get(currency, "")
            surprise_data = surprise.get(country, {})
            entry["surprise_trend"] = surprise_data.get("trend", "flat")
            entry["surprise_score"] = surprise_data.get("score", 0)

            # CB delta
            for delta in cb_deltas:
                if delta.get("bank", "") and delta.get("currency_impact", {}).get(currency):
                    entry["cb_shift"] = delta.get("hawkish_dovish_shift", 0)
                    entry["rate_path"] = delta.get("rate_path_implication", "uncertain")

            # Composite score (-5 to +5)
            composite = news_score * 0.4 + entry.get("surprise_score", 0) * 0.01 * 3
            entry["score"] = round(max(-5, min(5, composite)), 1)

            sentiment[currency] = entry

        return sentiment

    # ── Helpers ────────────────────────────────────────────────

    async def _get_recent_headlines(self) -> list[dict[str, Any]]:
        if self._news_feed:
            try:
                return await self._news_feed.get_unprocessed(limit=30)
            except Exception:
                pass
        return []

    def _get_current_prices(self, market_state: MarketState) -> dict[str, float]:
        return {
            sym: float((tick.bid + tick.ask) / 2)
            for sym, tick in market_state.prices.items()
        }

    async def _get_vix(self) -> float | None:
        try:
            from src.data.storage.cache import get_latest_price
            cached = await get_latest_price("VIX")
            return float(cached.get("mid", 0)) if cached else None
        except Exception:
            return None

    @property
    def stats(self) -> dict[str, Any]:
        base = super().stats
        base["feedback_weights"] = self._feedback.weights
        base["headline_history_size"] = len(self._headline_sentiment_history)
        return base
