"""
Sentiment Computation Models.

Pure computation — no LLM, no database, no side effects.
10 analytical layers for the Sentiment Agent.

Layer 2: COT Positioning Intelligence (speculator vs commercial decomposition)
Layer 3: Economic Surprise Index (actual vs forecast with decay + revisions)
Layer 4: Sentiment-Price Divergence Detector
Layer 6: Attention Model (what the market cares about RIGHT NOW)
Layer 8: Seasonal Patterns
Layer 9: Reflexivity Detection
Layer 10: Feedback Loop (self-improving hit rate tracking)

Layers 1, 5, 7 are LLM-powered (in sentiment.py and prompts/sentiment.py)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Layer 2: COT Positioning Intelligence ─────────────────────

class PositioningExtreme(StrEnum):
    EXTREME_LONG = "extreme_long"     # >90th percentile
    ELEVATED_LONG = "elevated_long"   # 75-90th percentile
    NEUTRAL = "neutral"               # 25-75th percentile
    ELEVATED_SHORT = "elevated_short" # 10-25th percentile
    EXTREME_SHORT = "extreme_short"   # <10th percentile


@dataclass
class COTDecomposition:
    """Full COT analysis for one instrument."""
    instrument: str
    # Raw positioning
    speculator_net: int = 0
    commercial_net: int = 0
    open_interest: int = 0
    # Percentile rankings (0-100, vs 2-year history)
    speculator_percentile: float = 50.0
    commercial_percentile: float = 50.0
    # The gap — when speculators and commercials disagree strongly, commercials win
    spec_commercial_gap: float = 0.0
    gap_percentile: float = 50.0
    # Momentum (3-week change direction)
    speculator_momentum: str = "flat"  # "building", "unwinding", "flat"
    weeks_in_direction: int = 0
    # Classification
    positioning_extreme: PositioningExtreme = PositioningExtreme.NEUTRAL
    crowded_trade: bool = False
    reversal_probability: float = 0.0  # 0-1
    # Contrarian signal
    contrarian_signal: str = "none"  # "contrarian_buy", "contrarian_sell", "none"
    reasoning: str = ""


def analyze_cot_positioning(
    current: dict[str, Any],
    history: list[dict[str, Any]],
) -> COTDecomposition:
    """
    Full COT decomposition with speculator/commercial split.

    current: latest COT data for one instrument
    history: last 104 weeks (2 years) of COT data for percentile ranking
    """
    instrument = current.get("instrument", "unknown")

    spec_net = current.get("non_commercial_long", 0) - current.get("non_commercial_short", 0)
    # Commercials = total OI - speculators - non-reportable
    comm_long = current.get("commercial_long", 0)
    comm_short = current.get("commercial_short", 0)
    comm_net = comm_long - comm_short
    oi = current.get("open_interest", 1)

    result = COTDecomposition(
        instrument=instrument,
        speculator_net=spec_net,
        commercial_net=comm_net,
        open_interest=oi,
    )

    if not history:
        return result

    # Percentile rankings
    hist_spec_nets = [h.get("net_position", 0) for h in history]
    hist_comm_nets = [
        h.get("commercial_long", 0) - h.get("commercial_short", 0)
        for h in history
    ]

    if hist_spec_nets:
        result.speculator_percentile = _percentile_rank(spec_net, hist_spec_nets)
    if hist_comm_nets:
        result.commercial_percentile = _percentile_rank(comm_net, hist_comm_nets)

    # Spec-Commercial gap
    result.spec_commercial_gap = (result.speculator_percentile - result.commercial_percentile)
    gap_history = [
        _percentile_rank(h.get("net_position", 0), hist_spec_nets) -
        _percentile_rank(
            h.get("commercial_long", 0) - h.get("commercial_short", 0),
            hist_comm_nets,
        )
        for h in history
        if hist_spec_nets and hist_comm_nets
    ]
    if gap_history:
        result.gap_percentile = _percentile_rank(result.spec_commercial_gap, gap_history)

    # Momentum (3-week direction)
    if len(hist_spec_nets) >= 3:
        recent_3 = hist_spec_nets[-3:]
        if all(recent_3[i] < recent_3[i + 1] for i in range(len(recent_3) - 1)):
            result.speculator_momentum = "building"
        elif all(recent_3[i] > recent_3[i + 1] for i in range(len(recent_3) - 1)):
            result.speculator_momentum = "unwinding"
        else:
            result.speculator_momentum = "flat"

    # Classify extremes
    pct = result.speculator_percentile
    if pct >= 90:
        result.positioning_extreme = PositioningExtreme.EXTREME_LONG
        result.crowded_trade = True
    elif pct >= 75:
        result.positioning_extreme = PositioningExtreme.ELEVATED_LONG
    elif pct <= 10:
        result.positioning_extreme = PositioningExtreme.EXTREME_SHORT
        result.crowded_trade = True
    elif pct <= 25:
        result.positioning_extreme = PositioningExtreme.ELEVATED_SHORT
    else:
        result.positioning_extreme = PositioningExtreme.NEUTRAL

    # Reversal probability
    if result.crowded_trade:
        # Base: 30% at 90th percentile, scales up to 50% at 99th
        base_prob = 0.30
        extreme_factor = (max(pct, 100 - pct) - 90) / 10 * 0.20
        # Boost if commercials disagree
        if abs(result.spec_commercial_gap) > 40:
            extreme_factor += 0.15
        # Boost if momentum is unwinding
        if result.speculator_momentum == "unwinding":
            extreme_factor += 0.10
        result.reversal_probability = min(base_prob + extreme_factor, 0.80)

    # Contrarian signal
    if result.positioning_extreme == PositioningExtreme.EXTREME_LONG and result.reversal_probability > 0.35:
        result.contrarian_signal = "contrarian_sell"
        result.reasoning = (
            f"Speculators at {pct:.0f}th percentile (extreme long). "
            f"Commercial gap: {result.spec_commercial_gap:.0f}. "
            f"Reversal probability: {result.reversal_probability:.0%}"
        )
    elif result.positioning_extreme == PositioningExtreme.EXTREME_SHORT and result.reversal_probability > 0.35:
        result.contrarian_signal = "contrarian_buy"
        result.reasoning = (
            f"Speculators at {pct:.0f}th percentile (extreme short). "
            f"Commercial gap: {result.spec_commercial_gap:.0f}. "
            f"Reversal probability: {result.reversal_probability:.0%}"
        )

    return result


# ── Layer 3: Economic Surprise Index ──────────────────────────

@dataclass
class EconomicSurprise:
    """Single economic release surprise."""
    event_name: str
    country: str
    actual: float
    forecast: float
    previous: float
    surprise_pct: float  # (actual - forecast) / |forecast|
    revision: float  # current previous - last reported previous (revision adjustment)
    quality_score: float  # 0-1, are internals strong?
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decay_weight: float = 1.0  # Decays over time


@dataclass
class SurpriseIndex:
    """Rolling economic surprise index for one economy."""
    country: str
    score: float = 0.0  # -100 (all missing) to +100 (all beating)
    trend: str = "flat"  # "improving", "deteriorating", "flat"
    momentum_shift: bool = False  # Was beating, now missing (or vice versa)
    streak: int = 0  # Consecutive beats (positive) or misses (negative)
    recent_surprises: list[EconomicSurprise] = field(default_factory=list)


def compute_surprise_index(
    releases: list[dict[str, Any]],
    country: str,
    decay_half_life_hours: float = 72.0,
) -> SurpriseIndex:
    """
    Compute rolling economic surprise index with time decay.

    Recent surprises matter more. A beat from 1 hour ago is more relevant
    than a beat from last week. Exponential decay with 72-hour half-life.
    """
    now = datetime.now(timezone.utc)
    index = SurpriseIndex(country=country)

    if not releases:
        return index

    surprises: list[EconomicSurprise] = []
    for r in releases:
        try:
            actual = float(r.get("actual", 0))
            forecast = float(r.get("forecast", 0))
            previous = float(r.get("previous", 0))
        except (ValueError, TypeError):
            continue

        if forecast == 0:
            continue

        surprise_pct = ((actual - forecast) / abs(forecast)) * 100

        # Time decay
        ts = r.get("ts", now)
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = now
        hours_ago = max((now - ts).total_seconds() / 3600, 0)
        decay = math.exp(-0.693 * hours_ago / decay_half_life_hours)  # 0.693 = ln(2)

        surprises.append(EconomicSurprise(
            event_name=r.get("event_name", ""),
            country=country,
            actual=actual,
            forecast=forecast,
            previous=previous,
            surprise_pct=round(surprise_pct, 2),
            revision=0,  # Would need previous report's "previous" to compute
            quality_score=1.0,  # Placeholder until we parse internals
            ts=ts,
            decay_weight=round(decay, 4),
        ))

    if not surprises:
        return index

    # Weighted surprise score
    weighted_sum = sum(s.surprise_pct * s.decay_weight for s in surprises)
    weight_total = sum(s.decay_weight for s in surprises)
    index.score = round(weighted_sum / weight_total, 2) if weight_total > 0 else 0

    # Clamp to -100, +100
    index.score = max(-100, min(100, index.score))

    # Trend detection (last 5 vs previous 5)
    sorted_surprises = sorted(surprises, key=lambda s: s.ts)
    if len(sorted_surprises) >= 10:
        recent_5 = sorted_surprises[-5:]
        prior_5 = sorted_surprises[-10:-5]
        recent_avg = sum(s.surprise_pct for s in recent_5) / 5
        prior_avg = sum(s.surprise_pct for s in prior_5) / 5
        if recent_avg > prior_avg + 1:
            index.trend = "improving"
        elif recent_avg < prior_avg - 1:
            index.trend = "deteriorating"
        else:
            index.trend = "flat"

        # Momentum shift: was positive, now negative (or vice versa)
        if (prior_avg > 0 and recent_avg < 0) or (prior_avg < 0 and recent_avg > 0):
            index.momentum_shift = True

    # Streak counting
    streak = 0
    for s in reversed(sorted_surprises):
        if s.surprise_pct > 0.5:
            if streak >= 0:
                streak += 1
            else:
                break
        elif s.surprise_pct < -0.5:
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break
    index.streak = streak

    index.recent_surprises = sorted_surprises[-10:]
    return index


# ── Layer 4: Sentiment-Price Divergence ───────────────────────

class DivergenceType(StrEnum):
    NEWS_PRICE = "news_price"
    POSITIONING_PRICE = "positioning_price"
    SURPRISE_PRICE = "surprise_price"
    TRIPLE = "triple"  # All three diverging simultaneously


@dataclass
class SentimentPriceDivergence:
    """Detected divergence between sentiment and price action."""
    instrument: str
    divergence_type: DivergenceType
    direction: str  # "bearish_divergence" (price up, sentiment down) or "bullish_divergence"
    severity: float  # 0-100
    components: dict[str, str] = field(default_factory=dict)
    reasoning: str = ""


def detect_sentiment_price_divergence(
    instrument: str,
    price_trend: str,  # "up", "down", "flat" (from recent price action)
    news_sentiment: float,  # -5 to +5
    cot_momentum: str,  # "building", "unwinding", "flat"
    surprise_trend: str,  # "improving", "deteriorating", "flat"
) -> SentimentPriceDivergence | None:
    """
    Detect divergence between sentiment layers and price action.

    Bearish divergence: price rising but sentiment/positioning/data weakening
    Bullish divergence: price falling but sentiment/positioning/data improving

    Triple divergence (all 3 diverge from price) = highest probability reversal.
    """
    divergences: dict[str, str] = {}

    # News-Price divergence
    if price_trend == "up" and news_sentiment < -1:
        divergences["news_price"] = "bearish"
    elif price_trend == "down" and news_sentiment > 1:
        divergences["news_price"] = "bullish"

    # Positioning-Price divergence
    if price_trend == "up" and cot_momentum == "unwinding":
        divergences["positioning_price"] = "bearish"
    elif price_trend == "down" and cot_momentum == "building":
        divergences["positioning_price"] = "bullish"

    # Surprise-Price divergence
    if price_trend == "up" and surprise_trend == "deteriorating":
        divergences["surprise_price"] = "bearish"
    elif price_trend == "down" and surprise_trend == "improving":
        divergences["surprise_price"] = "bullish"

    if not divergences:
        return None

    # Classify
    bearish_count = sum(1 for v in divergences.values() if v == "bearish")
    bullish_count = sum(1 for v in divergences.values() if v == "bullish")
    total = len(divergences)

    if total >= 3:
        div_type = DivergenceType.TRIPLE
        severity = 90.0
    elif "news_price" in divergences and "positioning_price" in divergences:
        div_type = DivergenceType.NEWS_PRICE  # Dominant type
        severity = 70.0
    elif "positioning_price" in divergences:
        div_type = DivergenceType.POSITIONING_PRICE
        severity = 60.0
    elif "surprise_price" in divergences:
        div_type = DivergenceType.SURPRISE_PRICE
        severity = 50.0
    else:
        div_type = DivergenceType.NEWS_PRICE
        severity = 40.0

    direction = "bearish_divergence" if bearish_count > bullish_count else "bullish_divergence"

    return SentimentPriceDivergence(
        instrument=instrument,
        divergence_type=div_type,
        direction=direction,
        severity=severity,
        components=divergences,
        reasoning=f"{total} divergence(s) detected: {', '.join(divergences.keys())}",
    )


# ── Layer 6: Attention Model ─────────────────────────────────

class AttentionRegime(StrEnum):
    """What the market is focused on RIGHT NOW."""
    RATE_CYCLE = "rate_cycle"         # CB speeches, rate expectations dominate
    CRISIS = "crisis"                 # Geopolitics, safe haven flows dominate
    GROWTH_SCARE = "growth_scare"     # PMI, employment, GDP dominate
    INFLATION_PANIC = "inflation_panic"  # CPI, PPI, wage growth dominate
    LIQUIDITY = "liquidity"           # QE/QT, balance sheet, M2 dominate
    ELECTION = "election"             # Political uncertainty dominates
    MIXED = "mixed"                   # No clear dominant theme


@dataclass
class AttentionState:
    """Current market attention classification."""
    regime: AttentionRegime
    confidence: float  # 0-1
    dominant_keywords: list[str] = field(default_factory=list)
    what_matters_now: list[str] = field(default_factory=list)
    what_is_noise: list[str] = field(default_factory=list)


def classify_attention_regime(
    recent_headlines: list[str],
    vix: float | None = None,
    geo_risk_score: float = 0,
) -> AttentionState:
    """
    Classify what the market cares about based on headline keyword frequency.
    This tells all other agents WHICH data points matter today.
    """
    # Keyword banks
    rate_keywords = {"rate", "fed", "ecb", "boj", "boe", "cut", "hike", "hawkish", "dovish", "fomc", "policy"}
    crisis_keywords = {"war", "conflict", "sanctions", "crisis", "attack", "military", "escalation", "nuclear"}
    growth_keywords = {"gdp", "recession", "pmi", "employment", "jobs", "growth", "slowdown", "contraction"}
    inflation_keywords = {"cpi", "inflation", "prices", "ppi", "wage", "cost", "deflation", "stagflation"}
    election_keywords = {"election", "vote", "poll", "president", "parliament", "coalition", "campaign"}

    combined = " ".join(h.lower() for h in recent_headlines)

    scores = {
        AttentionRegime.RATE_CYCLE: sum(1 for k in rate_keywords if k in combined),
        AttentionRegime.CRISIS: sum(1 for k in crisis_keywords if k in combined),
        AttentionRegime.GROWTH_SCARE: sum(1 for k in growth_keywords if k in combined),
        AttentionRegime.INFLATION_PANIC: sum(1 for k in inflation_keywords if k in combined),
        AttentionRegime.ELECTION: sum(1 for k in election_keywords if k in combined),
    }

    # Boost crisis if geo risk is high or VIX is elevated
    if geo_risk_score > 50:
        scores[AttentionRegime.CRISIS] += 5
    if vix and vix > 25:
        scores[AttentionRegime.CRISIS] += 3

    total = sum(scores.values()) or 1
    best_regime = max(scores, key=lambda k: scores[k])
    best_score = scores[best_regime]

    if best_score < 2 or (best_score / total) < 0.3:
        regime = AttentionRegime.MIXED
        confidence = 0.3
    else:
        regime = best_regime
        confidence = min(best_score / total, 0.95)

    # What matters vs noise
    matters = {
        AttentionRegime.RATE_CYCLE: ["CB speeches", "rate decisions", "bond yields", "forward guidance"],
        AttentionRegime.CRISIS: ["geopolitical events", "safe haven flows", "oil supply", "defense"],
        AttentionRegime.GROWTH_SCARE: ["PMI", "employment", "GDP", "retail sales"],
        AttentionRegime.INFLATION_PANIC: ["CPI", "PPI", "wage growth", "commodity prices"],
        AttentionRegime.ELECTION: ["poll results", "policy proposals", "coalition talks"],
        AttentionRegime.MIXED: ["all data relevant", "no dominant theme"],
    }

    noise = {
        AttentionRegime.RATE_CYCLE: ["GDP revisions", "trade balance"],
        AttentionRegime.CRISIS: ["routine economic data", "corporate earnings"],
        AttentionRegime.GROWTH_SCARE: ["CB rhetoric", "trade policy"],
        AttentionRegime.INFLATION_PANIC: ["housing data", "consumer confidence"],
        AttentionRegime.ELECTION: ["routine economic releases"],
        AttentionRegime.MIXED: [],
    }

    return AttentionState(
        regime=regime,
        confidence=round(confidence, 2),
        dominant_keywords=sorted(scores, key=lambda k: scores[k], reverse=True)[:3],
        what_matters_now=matters.get(regime, []),
        what_is_noise=noise.get(regime, []),
    )


# ── Layer 8: Seasonal Patterns ────────────────────────────────

@dataclass
class SeasonalSignal:
    """Active seasonal pattern."""
    pattern_name: str
    affected_instruments: list[str]
    expected_effect: str
    confidence: float
    start_date: str
    end_date: str


def detect_seasonal_patterns(ts: datetime | None = None) -> list[SeasonalSignal]:
    """Detect active calendar-driven patterns."""
    ts = ts or datetime.now(timezone.utc)
    month, day, weekday = ts.month, ts.day, ts.weekday()
    signals: list[SeasonalSignal] = []

    # NFP week compression (Mon-Thu before first Friday)
    if weekday < 4 and day <= 7:
        # Check if this week contains the first Friday
        days_to_friday = 4 - weekday
        friday_day = day + days_to_friday
        if friday_day <= 7:
            signals.append(SeasonalSignal(
                pattern_name="nfp_week_compression",
                affected_instruments=["EUR/USD", "GBP/USD", "USD/JPY"],
                expected_effect="Ranges compress Mon-Thu, breakout Friday 13:30 UTC",
                confidence=0.70,
                start_date=ts.strftime("%Y-%m-%d"),
                end_date=(ts + timedelta(days=days_to_friday)).strftime("%Y-%m-%d"),
            ))

    # December JPY repatriation
    if month == 12 and day >= 10:
        signals.append(SeasonalSignal(
            pattern_name="jpn_year_end_repatriation",
            affected_instruments=["USD/JPY"],
            expected_effect="JPY tends to strengthen as corporates repatriate",
            confidence=0.60,
            start_date=f"{ts.year}-12-10",
            end_date=f"{ts.year}-12-31",
        ))

    # Golden Week Japan (late April - early May)
    if (month == 4 and day >= 27) or (month == 5 and day <= 6):
        signals.append(SeasonalSignal(
            pattern_name="golden_week_japan",
            affected_instruments=["USD/JPY"],
            expected_effect="JPY liquidity drops, wider spreads, avoid large positions",
            confidence=0.75,
            start_date=f"{ts.year}-04-27",
            end_date=f"{ts.year}-05-06",
        ))

    # Month-end rebalancing (last 3 trading days)
    next_month = (ts.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day = next_month - timedelta(days=1)
    if (last_day - ts).days <= 3 and (last_day - ts).days >= 0:
        signals.append(SeasonalSignal(
            pattern_name="month_end_rebalancing",
            affected_instruments=["EUR/USD", "GBP/USD", "USD/JPY"],
            expected_effect="Large portfolio rebalancing flows, directional bias from fixing models",
            confidence=0.65,
            start_date=(last_day - timedelta(days=3)).strftime("%Y-%m-%d"),
            end_date=last_day.strftime("%Y-%m-%d"),
        ))

    # Quarter-end amplified rebalancing
    if month in (3, 6, 9, 12) and (last_day - ts).days <= 3 and (last_day - ts).days >= 0:
        signals.append(SeasonalSignal(
            pattern_name="quarter_end_rebalancing",
            affected_instruments=["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"],
            expected_effect="Amplified rebalancing flows, 2-3x month-end magnitude",
            confidence=0.70,
            start_date=(last_day - timedelta(days=3)).strftime("%Y-%m-%d"),
            end_date=last_day.strftime("%Y-%m-%d"),
        ))

    # January effect (USD typically weakens in January)
    if month == 1:
        signals.append(SeasonalSignal(
            pattern_name="january_usd_weakness",
            affected_instruments=["EUR/USD", "GBP/USD", "AUD/USD"],
            expected_effect="USD tends to weaken in January (portfolio rotation into EM/commodities)",
            confidence=0.55,
            start_date=f"{ts.year}-01-01",
            end_date=f"{ts.year}-01-31",
        ))

    return signals


# ── Layer 9: Reflexivity Detection ────────────────────────────

@dataclass
class ReflexivitySignal:
    """Detected self-reinforcing loop."""
    instrument: str
    loop_type: str  # "bubble" or "panic"
    phase: str  # "accelerating", "mature", "exhausting"
    strength: float  # 0-100
    fundamental_support: bool  # Does the fundamental still support the move?
    reasoning: str = ""


def detect_reflexivity(
    instrument: str,
    price_change_30d_pct: float,
    price_change_7d_pct: float,
    price_change_1d_pct: float,
    fundamental_direction: str,  # From Macro Agent: "bullish" or "bearish"
    cot_momentum: str,  # "building" or "unwinding"
) -> ReflexivitySignal | None:
    """
    Detect self-reinforcing loops (Soros reflexivity).

    Bubble: price rising → attracts flow → more rising → disconnects from fundamentals
    Panic: price falling → capital flight → more falling → overshoots fair value

    The reversal comes when:
    1. Fundamentals have shifted but price hasn't caught up
    2. Positioning is exhausting (COT unwinding)
    3. Acceleration is decelerating (still up, but less fast)
    """
    # Detect acceleration pattern
    is_accelerating = (
        abs(price_change_1d_pct) > abs(price_change_7d_pct) / 7 * 1.5 and
        abs(price_change_7d_pct) > abs(price_change_30d_pct) / 30 * 7 * 1.2
    )

    # Is it a strong move?
    strong_move = abs(price_change_30d_pct) > 3  # >3% in 30 days is significant for forex

    if not strong_move:
        return None

    is_up = price_change_30d_pct > 0
    loop_type = "bubble" if is_up else "panic"

    # Phase detection
    if is_accelerating:
        phase = "accelerating"
    elif abs(price_change_7d_pct) > abs(price_change_30d_pct) / 4:
        phase = "mature"
    else:
        phase = "exhausting"

    # Does fundamental support the move?
    fund_supports = (
        (is_up and fundamental_direction == "bullish") or
        (not is_up and fundamental_direction == "bearish")
    )

    # Strength score
    strength = min(abs(price_change_30d_pct) * 10, 100)
    if not fund_supports:
        strength *= 0.7  # Discount if fundamental doesn't support
    if cot_momentum == "unwinding":
        strength *= 0.8  # Discount if positioning is reversing

    if strength < 20:
        return None

    reasoning_parts = [f"30d move: {price_change_30d_pct:+.2f}%"]
    if not fund_supports:
        reasoning_parts.append("fundamental direction has shifted against the move")
    if cot_momentum == "unwinding":
        reasoning_parts.append("institutional positioning is unwinding")
    if phase == "exhausting":
        reasoning_parts.append("acceleration is decelerating")

    return ReflexivitySignal(
        instrument=instrument,
        loop_type=loop_type,
        phase=phase,
        strength=round(strength, 1),
        fundamental_support=fund_supports,
        reasoning="; ".join(reasoning_parts),
    )


# ── Layer 10: Feedback Loop ──────────────────────────────────

@dataclass
class PredictionRecord:
    """A tracked sentiment prediction for feedback scoring."""
    ts: datetime
    instrument: str
    prediction: str  # "bullish" or "bearish"
    source: str      # "news", "cot", "surprise", "divergence"
    confidence: float
    outcome: str | None = None  # "correct", "incorrect", None (pending)
    price_at_prediction: float = 0
    price_at_resolution: float = 0


class FeedbackTracker:
    """
    Tracks prediction accuracy over time.
    Adjusts source weights based on historical hit rates.
    """

    def __init__(self) -> None:
        self._records: list[PredictionRecord] = []
        self._source_weights: dict[str, float] = {
            "news": 1.0,
            "cot": 1.0,
            "surprise": 1.0,
            "divergence": 1.0,
        }
        self._max_records = 1000

    def record_prediction(
        self,
        instrument: str,
        prediction: str,
        source: str,
        confidence: float,
        current_price: float,
    ) -> None:
        """Record a new prediction for later scoring."""
        self._records.append(PredictionRecord(
            ts=datetime.now(timezone.utc),
            instrument=instrument,
            prediction=prediction,
            source=source,
            confidence=confidence,
            price_at_prediction=current_price,
        ))
        # Trim old records
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

    def resolve_predictions(
        self,
        instrument: str,
        current_price: float,
        min_age_hours: float = 4.0,
        threshold_pct: float = 0.1,
    ) -> None:
        """
        Resolve pending predictions that are old enough.
        Mark as correct/incorrect based on whether price moved in predicted direction.
        """
        now = datetime.now(timezone.utc)
        for record in self._records:
            if record.outcome is not None:
                continue
            if record.instrument != instrument:
                continue
            age_hours = (now - record.ts).total_seconds() / 3600
            if age_hours < min_age_hours:
                continue

            record.price_at_resolution = current_price
            change_pct = ((current_price - record.price_at_prediction) /
                          record.price_at_prediction) * 100

            if abs(change_pct) < threshold_pct:
                record.outcome = "neutral"
            elif (record.prediction == "bullish" and change_pct > 0) or \
                 (record.prediction == "bearish" and change_pct < 0):
                record.outcome = "correct"
            else:
                record.outcome = "incorrect"

    def get_hit_rates(self, window_days: int = 30) -> dict[str, dict[str, float]]:
        """Get hit rate per source over the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        rates: dict[str, dict[str, float]] = {}

        for source in self._source_weights:
            relevant = [
                r for r in self._records
                if r.source == source and r.ts > cutoff and r.outcome in ("correct", "incorrect")
            ]
            total = len(relevant)
            correct = sum(1 for r in relevant if r.outcome == "correct")

            rates[source] = {
                "total": total,
                "correct": correct,
                "hit_rate": round(correct / total, 3) if total > 0 else 0.5,
            }

        return rates

    def update_weights(self) -> None:
        """Adjust source weights based on recent hit rates."""
        rates = self.get_hit_rates(window_days=30)
        for source, data in rates.items():
            if data["total"] >= 10:  # Need minimum sample
                # Weight = hit_rate normalized around 1.0
                # 60% hit rate → 1.2 weight, 40% → 0.8 weight
                self._source_weights[source] = round(data["hit_rate"] * 2, 2)
            # Clamp to 0.5-2.0
            self._source_weights[source] = max(0.5, min(2.0, self._source_weights[source]))

    @property
    def weights(self) -> dict[str, float]:
        return self._source_weights.copy()


# ── Fear & Greed Index ────────────────────────────────────────

@dataclass
class FearGreedIndex:
    """Composite fear/greed score."""
    score: float  # 0 (extreme fear) to 100 (extreme greed)
    label: str    # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
    components: dict[str, float] = field(default_factory=dict)
    active_divergences: int = 0
    contrarian_signal: str = "none"  # "contrarian_buy", "contrarian_sell", "none"


def compute_fear_greed(
    news_sentiment_avg: float,       # -5 to +5
    cot_percentile_avg: float,       # 0-100 (avg across instruments)
    surprise_score_avg: float,       # -100 to +100
    vix: float | None,
    divergence_count: int = 0,
    intermarket_risk_score: float = 50,  # 0 (risk-off) to 100 (risk-on)
    source_weights: dict[str, float] | None = None,
) -> FearGreedIndex:
    """
    Compute composite Fear & Greed Index.

    Below 15 = extreme fear → contrarian buy signal
    Above 85 = extreme greed → contrarian sell signal
    """
    weights = source_weights or {}

    # Normalize each component to 0-100
    # News sentiment: -5 to +5 → 0 to 100
    news_score = ((news_sentiment_avg + 5) / 10) * 100
    news_weight = 0.15 * weights.get("news", 1.0)

    # COT positioning: already 0-100
    cot_score = cot_percentile_avg
    cot_weight = 0.20 * weights.get("cot", 1.0)

    # Surprise: -100 to +100 → 0 to 100
    surprise_score = ((surprise_score_avg + 100) / 200) * 100
    surprise_weight = 0.15 * weights.get("surprise", 1.0)

    # VIX: inverted (high VIX = fear), scaled 10-40 → 100-0
    if vix is not None:
        vix_score = max(0, min(100, (40 - vix) / 30 * 100))
    else:
        vix_score = 50
    vix_weight = 0.15

    # Divergences: more divergences = more stress = lower score
    div_score = max(0, 100 - divergence_count * 30)
    div_weight = 0.10

    # Intermarket: already 0-100
    inter_weight = 0.10

    # Unused weight goes to neutral (ensures weights sum properly)
    total_weight = news_weight + cot_weight + surprise_weight + vix_weight + div_weight + inter_weight

    composite = (
        news_score * news_weight +
        cot_score * cot_weight +
        surprise_score * surprise_weight +
        vix_score * vix_weight +
        div_score * div_weight +
        intermarket_risk_score * inter_weight
    ) / total_weight if total_weight > 0 else 50

    composite = max(0, min(100, composite))

    # Label
    if composite <= 15:
        label = "extreme_fear"
    elif composite <= 35:
        label = "fear"
    elif composite <= 65:
        label = "neutral"
    elif composite <= 85:
        label = "greed"
    else:
        label = "extreme_greed"

    # Contrarian signal
    contrarian = "none"
    if composite <= 15:
        contrarian = "contrarian_buy"
    elif composite >= 85:
        contrarian = "contrarian_sell"

    return FearGreedIndex(
        score=round(composite, 1),
        label=label,
        components={
            "news_sentiment": round(news_score, 1),
            "cot_positioning": round(cot_score, 1),
            "economic_surprise": round(surprise_score, 1),
            "vix": round(vix_score, 1),
            "divergence": round(div_score, 1),
            "intermarket": round(intermarket_risk_score, 1),
        },
        active_divergences=divergence_count,
        contrarian_signal=contrarian,
    )


# ── Helpers ───────────────────────────────────────────────────

def _percentile_rank(value: float, distribution: list[float]) -> float:
    """Compute percentile rank of a value in a distribution (0-100)."""
    if not distribution:
        return 50.0
    below = sum(1 for v in distribution if v < value)
    return round((below / len(distribution)) * 100, 1)


def compute_sentiment_decay(
    scores: list[tuple[float, datetime]],
    half_life_hours: float = 4.0,
) -> float:
    """
    Compute time-decayed average of sentiment scores.
    Recent scores weighted more heavily.
    """
    now = datetime.now(timezone.utc)
    weighted_sum = 0.0
    weight_total = 0.0

    for score, ts in scores:
        hours_ago = max((now - ts).total_seconds() / 3600, 0)
        weight = math.exp(-0.693 * hours_ago / half_life_hours)
        weighted_sum += score * weight
        weight_total += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.0
