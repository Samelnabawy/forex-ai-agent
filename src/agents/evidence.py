"""
Evidence Framework for Bull/Bear Debate.

Every claim in the debate must be backed by scored evidence.
This isn't about opinions — it's about weighted probability.

Evidence scoring:
  - Source reliability (how often has this agent/signal been correct?)
  - Recency (fresh data > stale data)
  - Relevance (direct match > loose analogy)
  - Confluence (multiple independent sources confirming > single source)

Expected Value calculation:
  EV = P(win) × avg_win_pips - P(loss) × avg_loss_pips
  A trade is worth taking when EV > 0 AND EV/risk > minimum threshold.
  Even 45% win rate is profitable if avg_win > 2× avg_loss.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ── Evidence Types ────────────────────────────────────────────

class EvidenceSource(StrEnum):
    TECHNICAL = "technical_analyst"
    MACRO = "macro_analyst"
    CORRELATION = "correlation_agent"
    SENTIMENT = "sentiment_agent"
    HISTORICAL_BRAIN = "historical_brain"
    PRICE_ACTION = "price_action"
    STRUCTURE = "market_structure"
    CALENDAR = "economic_calendar"
    POSITIONING = "cot_positioning"
    INTERMARKET = "intermarket"


class EvidenceDirection(StrEnum):
    SUPPORTING = "supporting"    # Supports the trade thesis
    OPPOSING = "opposing"        # Contradicts the trade thesis
    NEUTRAL = "neutral"          # Neither supports nor opposes
    CONDITIONAL = "conditional"  # Supports IF a condition is met


class EvidenceStrength(StrEnum):
    STRONG = "strong"       # >70% historical reliability
    MODERATE = "moderate"   # 50-70% reliability
    WEAK = "weak"           # 30-50% reliability
    SPECULATIVE = "speculative"  # <30% or insufficient data


@dataclass
class Evidence:
    """A single piece of evidence in the debate."""
    id: str                           # Unique ID for reference in rebuttals
    source: EvidenceSource
    direction: EvidenceDirection
    strength: EvidenceStrength
    claim: str                        # What this evidence asserts
    data: dict[str, Any]              # Raw data backing the claim
    reliability_score: float          # 0-1, historical accuracy of this signal type
    recency_hours: float              # How old is this data?
    relevance_score: float            # 0-1, how directly applicable
    weight: float = 0.0              # Computed: reliability × recency × relevance
    rebuttal: str | None = None       # Bear's counter-argument (filled during debate)
    rebuttal_strength: float = 0.0    # How strong is the rebuttal? (0-1)
    survived_rebuttal: bool = True    # Does this evidence hold after challenge?

    def compute_weight(self) -> float:
        """Compute evidence weight from component scores."""
        # Recency decay: halves every 6 hours
        recency_factor = math.exp(-0.693 * self.recency_hours / 6.0)
        self.weight = round(
            self.reliability_score * recency_factor * self.relevance_score, 4
        )
        return self.weight


@dataclass
class HistoricalPrecedent:
    """A historical match from the Brain that supports or opposes the trade."""
    event_description: str
    date: str
    similarity_score: float           # 0-1, how similar to current setup
    outcome: str                      # "win" or "loss"
    pnl_pips: float                   # Actual P&L if this trade was taken
    holding_period_hours: float
    market_context: dict[str, Any] = field(default_factory=dict)
    what_happened: str = ""           # Narrative of how it played out


@dataclass
class TradeProposal:
    """A trade idea to be debated."""
    trade_id: str
    instrument: str
    direction: str                    # "LONG" or "SHORT"
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None = None
    atr: float = 0.0
    timeframe: str = "1h"
    # Source signals
    technical_signal: dict[str, Any] = field(default_factory=dict)
    macro_outlook: dict[str, Any] = field(default_factory=dict)
    correlation_assessment: dict[str, Any] = field(default_factory=dict)
    sentiment_reading: dict[str, Any] = field(default_factory=dict)


# ── Expected Value Calculation ────────────────────────────────

@dataclass
class ExpectedValueCalc:
    """Full expected value breakdown for a proposed trade."""
    # Probabilities
    win_probability: float            # 0-1
    loss_probability: float           # 0-1 (= 1 - win_probability - scratch_probability)
    scratch_probability: float = 0.05 # Small chance of breakeven exit

    # Outcomes in pips
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0        # Negative number
    risk_reward_ratio: float = 0.0

    # Expected value
    expected_value_pips: float = 0.0
    ev_per_risk_unit: float = 0.0     # EV / risk — measures efficiency

    # Verdict
    is_positive_ev: bool = False
    meets_minimum_threshold: bool = False
    reasoning: str = ""


def calculate_expected_value(
    win_probability: float,
    entry: float,
    stop_loss: float,
    take_profit: float,
    pip_size: float,
    min_ev_threshold: float = 0.3,  # Minimum EV per pip risked
) -> ExpectedValueCalc:
    """
    Calculate the expected value of a trade.

    This is the core of institutional decision-making:
    - A 40% win rate is fine if winners are 3× bigger than losers
    - A 70% win rate is bad if losers are 5× bigger than winners
    - Only take trades with positive EV above a minimum threshold
    """
    risk_pips = abs(entry - stop_loss) / pip_size
    reward_pips = abs(take_profit - entry) / pip_size

    if risk_pips == 0:
        return ExpectedValueCalc(
            win_probability=win_probability,
            loss_probability=1 - win_probability,
            reasoning="Invalid: zero risk distance",
        )

    rr_ratio = reward_pips / risk_pips
    loss_probability = 1 - win_probability - 0.05  # 5% scratch rate
    loss_probability = max(loss_probability, 0)

    ev = (win_probability * reward_pips) + (loss_probability * -risk_pips)
    ev_per_risk = ev / risk_pips if risk_pips > 0 else 0

    is_positive = ev > 0
    meets_threshold = ev_per_risk >= min_ev_threshold

    reasoning_parts = []
    if is_positive:
        reasoning_parts.append(f"Positive EV: {ev:.1f} pips")
    else:
        reasoning_parts.append(f"Negative EV: {ev:.1f} pips — SKIP")

    if rr_ratio < 1.5:
        reasoning_parts.append(f"R:R ratio {rr_ratio:.2f} below 1.5 minimum")
    elif rr_ratio >= 2.5:
        reasoning_parts.append(f"Excellent R:R ratio: {rr_ratio:.2f}")

    if win_probability < 0.45:
        reasoning_parts.append(f"Low win rate ({win_probability:.0%}) — need >2.5 R:R to compensate")
    elif win_probability > 0.65:
        reasoning_parts.append(f"High win rate ({win_probability:.0%}) — strong statistical edge")

    return ExpectedValueCalc(
        win_probability=round(win_probability, 3),
        loss_probability=round(loss_probability, 3),
        avg_win_pips=round(reward_pips, 1),
        avg_loss_pips=round(-risk_pips, 1),
        risk_reward_ratio=round(rr_ratio, 2),
        expected_value_pips=round(ev, 1),
        ev_per_risk_unit=round(ev_per_risk, 3),
        is_positive_ev=is_positive,
        meets_minimum_threshold=meets_threshold,
        reasoning=" | ".join(reasoning_parts),
    )


# ── Evidence Aggregation ──────────────────────────────────────

@dataclass
class EvidencePackage:
    """Complete evidence package for one side of the debate."""
    side: str  # "bull" or "bear"
    trade_id: str
    instrument: str
    direction: str
    evidence: list[Evidence] = field(default_factory=list)
    precedents: list[HistoricalPrecedent] = field(default_factory=list)
    expected_value: ExpectedValueCalc | None = None
    total_weight: float = 0.0
    conviction_score: float = 0.0  # 0-100
    invalidation_conditions: list[str] = field(default_factory=list)
    alternative_actions: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    reasoning_summary: str = ""


def aggregate_evidence(evidence_list: list[Evidence]) -> dict[str, Any]:
    """
    Aggregate evidence into weighted scores.
    Returns supporting weight, opposing weight, and net conviction.
    """
    supporting_weight = 0.0
    opposing_weight = 0.0
    neutral_weight = 0.0

    for e in evidence_list:
        e.compute_weight()
        if e.direction == EvidenceDirection.SUPPORTING:
            if e.survived_rebuttal:
                supporting_weight += e.weight
            else:
                supporting_weight += e.weight * 0.3  # Weakened by rebuttal
        elif e.direction == EvidenceDirection.OPPOSING:
            opposing_weight += e.weight
        else:
            neutral_weight += e.weight

    total = supporting_weight + opposing_weight + neutral_weight
    if total == 0:
        return {
            "supporting_weight": 0, "opposing_weight": 0, "neutral_weight": 0,
            "net_weight": 0, "conviction": 0, "evidence_count": len(evidence_list),
            "strong_evidence": 0,
        }

    net = supporting_weight - opposing_weight
    conviction = (net / total) * 100 if total > 0 else 0

    return {
        "supporting_weight": round(supporting_weight, 3),
        "opposing_weight": round(opposing_weight, 3),
        "neutral_weight": round(neutral_weight, 3),
        "net_weight": round(net, 3),
        "conviction": round(max(-100, min(100, conviction)), 1),
        "evidence_count": len(evidence_list),
        "strong_evidence": sum(1 for e in evidence_list if e.strength == EvidenceStrength.STRONG),
    }


def compute_win_probability(
    bull_evidence: EvidencePackage,
    bear_evidence: EvidencePackage,
    historical_win_rate: float = 0.5,
) -> float:
    """
    Compute win probability from evidence packages.

    Combines:
    - Historical base rate (from similar setups)
    - Bull evidence strength (adjusted for rebuttals)
    - Bear evidence strength
    - Confluence factor (more independent supporting sources = higher probability)
    """
    # Base rate from historical precedents
    if bull_evidence.precedents:
        wins = sum(1 for p in bull_evidence.precedents if p.outcome == "win")
        total = len(bull_evidence.precedents)
        hist_rate = wins / total if total > 0 else 0.5
    else:
        hist_rate = historical_win_rate

    # Evidence adjustment
    bull_agg = aggregate_evidence(bull_evidence.evidence)
    bear_agg = aggregate_evidence(bear_evidence.evidence)

    bull_strength = bull_agg["supporting_weight"]
    bear_strength = bear_agg["opposing_weight"]
    total_strength = bull_strength + bear_strength

    if total_strength == 0:
        return hist_rate

    # Bayesian-ish update: evidence shifts the base rate
    evidence_ratio = bull_strength / total_strength  # 0 to 1
    # Blend: 60% evidence, 40% historical base rate
    probability = 0.6 * evidence_ratio + 0.4 * hist_rate

    # Confluence bonus: more independent sources = higher confidence
    unique_bull_sources = len(set(e.source for e in bull_evidence.evidence if e.direction == EvidenceDirection.SUPPORTING))
    if unique_bull_sources >= 4:
        probability = min(probability + 0.05, 0.85)
    elif unique_bull_sources >= 3:
        probability = min(probability + 0.03, 0.80)

    # Cap: never above 85% (markets are uncertain) or below 15%
    probability = max(0.15, min(0.85, probability))

    return round(probability, 3)


# ── Evidence Extraction Helpers ───────────────────────────────

def extract_technical_evidence(
    technical_signal: dict[str, Any],
    direction: str,
) -> list[Evidence]:
    """Extract evidence pieces from Technical Agent output."""
    evidence: list[Evidence] = []
    if not technical_signal:
        return evidence

    indicators = technical_signal.get("indicators", {})
    confluence = technical_signal.get("confluence_score", 0)
    is_long = direction == "LONG"

    # RSI
    rsi = indicators.get("rsi")
    if rsi is not None:
        if (is_long and rsi < 35) or (not is_long and rsi > 65):
            evidence.append(Evidence(
                id="tech_rsi",
                source=EvidenceSource.TECHNICAL,
                direction=EvidenceDirection.SUPPORTING,
                strength=EvidenceStrength.MODERATE,
                claim=f"RSI at {rsi:.1f} — {'oversold' if is_long else 'overbought'} condition",
                data={"rsi": rsi},
                reliability_score=0.60,
                recency_hours=0.1,
                relevance_score=0.8,
            ))

    # MACD cross
    macd_cross = indicators.get("macd_cross", "none")
    if (is_long and macd_cross == "bullish") or (not is_long and macd_cross == "bearish"):
        evidence.append(Evidence(
            id="tech_macd",
            source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.MODERATE,
            claim=f"MACD {macd_cross} cross confirmed",
            data={"macd_cross": macd_cross},
            reliability_score=0.55,
            recency_hours=0.1,
            relevance_score=0.85,
        ))

    # EMA alignment
    ema = indicators.get("ema_alignment", "")
    if (is_long and ema == "bullish") or (not is_long and ema == "bearish"):
        evidence.append(Evidence(
            id="tech_ema",
            source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim=f"EMA alignment {ema} (20>50>200)" if is_long else f"EMA alignment {ema} (20<50<200)",
            data={"ema_alignment": ema},
            reliability_score=0.65,
            recency_hours=0.1,
            relevance_score=0.9,
        ))

    # ADX trending
    adx = indicators.get("adx")
    if adx and adx > 25:
        evidence.append(Evidence(
            id="tech_adx",
            source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.MODERATE,
            claim=f"ADX at {adx:.1f} — strong trend in play",
            data={"adx": adx},
            reliability_score=0.60,
            recency_hours=0.1,
            relevance_score=0.7,
        ))

    # Ichimoku
    ichimoku = indicators.get("ichimoku", "")
    if (is_long and ichimoku == "above_cloud") or (not is_long and ichimoku == "below_cloud"):
        evidence.append(Evidence(
            id="tech_ichimoku",
            source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim=f"Price {ichimoku.replace('_', ' ')} — Ichimoku confirms trend",
            data={"ichimoku": ichimoku},
            reliability_score=0.62,
            recency_hours=0.1,
            relevance_score=0.85,
        ))

    # Squeeze (from volatility)
    if indicators.get("squeeze"):
        evidence.append(Evidence(
            id="tech_squeeze",
            source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.MODERATE,
            claim="Bollinger/Keltner squeeze — volatility expansion imminent",
            data={"squeeze": True},
            reliability_score=0.58,
            recency_hours=0.1,
            relevance_score=0.75,
        ))

    # Candlestick patterns
    patterns = technical_signal.get("patterns", [])
    bullish_patterns = {"hammer", "bullish_engulfing", "morning_star", "three_white_soldiers"}
    bearish_patterns = {"shooting_star", "bearish_engulfing", "evening_star", "three_black_crows"}
    target_patterns = bullish_patterns if is_long else bearish_patterns

    for p in patterns:
        if p in target_patterns:
            evidence.append(Evidence(
                id=f"tech_pattern_{p}",
                source=EvidenceSource.TECHNICAL,
                direction=EvidenceDirection.SUPPORTING,
                strength=EvidenceStrength.WEAK,
                claim=f"Candlestick pattern: {p.replace('_', ' ')}",
                data={"pattern": p},
                reliability_score=0.45,
                recency_hours=0.1,
                relevance_score=0.6,
            ))

    return evidence


def extract_macro_evidence(
    macro_outlook: dict[str, Any],
    instrument: str,
    direction: str,
) -> list[Evidence]:
    """Extract evidence from Macro Agent output."""
    evidence: list[Evidence] = []
    if not macro_outlook:
        return evidence

    is_long = direction == "LONG"
    currency_scores = macro_outlook.get("currency_scores", {})

    # Parse instrument currencies
    parts = instrument.split("/")
    if len(parts) != 2:
        return evidence

    base, quote = parts
    base_score = currency_scores.get(base, 0)
    quote_score = currency_scores.get(quote, 0)
    differential = base_score - quote_score

    # Currency score differential
    supports = (is_long and differential > 0) or (not is_long and differential < 0)
    if abs(differential) >= 2:
        evidence.append(Evidence(
            id="macro_currency_diff",
            source=EvidenceSource.MACRO,
            direction=EvidenceDirection.SUPPORTING if supports else EvidenceDirection.OPPOSING,
            strength=EvidenceStrength.STRONG if abs(differential) >= 4 else EvidenceStrength.MODERATE,
            claim=f"Macro currency differential: {base} ({base_score:+d}) vs {quote} ({quote_score:+d}) = {differential:+d}",
            data={"base_score": base_score, "quote_score": quote_score, "differential": differential},
            reliability_score=0.65,
            recency_hours=1.0,
            relevance_score=0.9,
        ))

    # Macro regime
    regime = macro_outlook.get("macro_regime", "neutral")
    regime_supports = (
        (is_long and regime == "risk_on" and base in ("AUD", "NZD", "EUR", "GBP")) or
        (not is_long and regime == "risk_off" and base in ("AUD", "NZD")) or
        (is_long and regime == "risk_off" and base in ("JPY", "CHF", "XAU"))
    )
    if regime != "neutral":
        evidence.append(Evidence(
            id="macro_regime",
            source=EvidenceSource.MACRO,
            direction=EvidenceDirection.SUPPORTING if regime_supports else EvidenceDirection.NEUTRAL,
            strength=EvidenceStrength.MODERATE,
            claim=f"Macro regime: {regime}",
            data={"regime": regime},
            reliability_score=0.60,
            recency_hours=1.0,
            relevance_score=0.7,
        ))

    # Preferred pairs
    for pair in macro_outlook.get("preferred_pairs", []):
        if pair.get("pair") == instrument:
            pair_dir = pair.get("direction", "")
            pair_supports = (is_long and pair_dir == "LONG") or (not is_long and pair_dir == "SHORT")
            conviction = pair.get("conviction", "low")
            evidence.append(Evidence(
                id="macro_preferred_pair",
                source=EvidenceSource.MACRO,
                direction=EvidenceDirection.SUPPORTING if pair_supports else EvidenceDirection.OPPOSING,
                strength={"high": EvidenceStrength.STRONG, "medium": EvidenceStrength.MODERATE}.get(conviction, EvidenceStrength.WEAK),
                claim=f"Macro Agent prefers {instrument} {pair_dir} ({conviction} conviction)",
                data=pair,
                reliability_score=0.65,
                recency_hours=1.0,
                relevance_score=0.95,
            ))

    return evidence


def extract_correlation_evidence(
    correlation_data: dict[str, Any],
    instrument: str,
    direction: str,
) -> list[Evidence]:
    """Extract evidence from Correlation Agent output."""
    evidence: list[Evidence] = []
    if not correlation_data:
        return evidence

    # Cross-validation verdict
    for cv in correlation_data.get("cross_validation", {}).get("technical_signals", []):
        if cv.get("instrument") == instrument:
            verdict = cv.get("correlation_assessment", "NEUTRAL")
            adj = cv.get("adjusted_confidence", 0.5)
            reasons = cv.get("reasons", [])

            supports = verdict == "CONFIRMED"
            evidence.append(Evidence(
                id="corr_cross_validation",
                source=EvidenceSource.CORRELATION,
                direction=EvidenceDirection.SUPPORTING if supports else (
                    EvidenceDirection.OPPOSING if verdict == "CONTRADICTED" else EvidenceDirection.NEUTRAL
                ),
                strength=EvidenceStrength.STRONG if abs(adj - 0.5) > 0.15 else EvidenceStrength.MODERATE,
                claim=f"Correlation cross-validation: {verdict} (adj confidence {adj:.2f})",
                data={"verdict": verdict, "adjusted_confidence": adj, "reasons": reasons},
                reliability_score=0.70,
                recency_hours=0.02,  # 1 minute ago
                relevance_score=0.95,
            ))

    # Active cascades
    for cascade in correlation_data.get("active_cascades", []):
        for effect in cascade.get("effects", []):
            if effect.get("instrument") == instrument:
                expected_dir = effect.get("direction", "")
                is_long = direction == "LONG"
                supports = (is_long and expected_dir == "up") or (not is_long and expected_dir == "down")

                evidence.append(Evidence(
                    id=f"corr_cascade_{cascade.get('trigger', '')}",
                    source=EvidenceSource.CORRELATION,
                    direction=EvidenceDirection.SUPPORTING if supports else EvidenceDirection.OPPOSING,
                    strength=EvidenceStrength.STRONG,
                    claim=f"Active cascade from {cascade.get('trigger', '?')}: {instrument} expected {expected_dir}",
                    data=effect,
                    reliability_score=effect.get("confidence", 0.5),
                    recency_hours=0.02,
                    relevance_score=0.90,
                ))

    # DXY decomposition
    dxy = correlation_data.get("dxy_decomposition", {})
    if dxy.get("is_broad_usd") is not None:
        evidence.append(Evidence(
            id="corr_dxy",
            source=EvidenceSource.CORRELATION,
            direction=EvidenceDirection.SUPPORTING if dxy.get("is_broad_usd") else EvidenceDirection.NEUTRAL,
            strength=EvidenceStrength.MODERATE,
            claim=dxy.get("interpretation", "DXY decomposition available"),
            data=dxy,
            reliability_score=0.65,
            recency_hours=0.02,
            relevance_score=0.80,
        ))

    return evidence


def extract_sentiment_evidence(
    sentiment_data: dict[str, Any],
    instrument: str,
    direction: str,
) -> list[Evidence]:
    """Extract evidence from Sentiment Agent output."""
    evidence: list[Evidence] = []
    if not sentiment_data:
        return evidence

    # Fear/Greed Index
    fg = sentiment_data.get("fear_greed_index", {})
    if fg:
        score = fg.get("score", 50)
        contrarian = fg.get("contrarian_signal", "none")

        if contrarian != "none":
            is_long = direction == "LONG"
            supports = (is_long and contrarian == "contrarian_buy") or (
                not is_long and contrarian == "contrarian_sell"
            )
            evidence.append(Evidence(
                id="sent_fear_greed_contrarian",
                source=EvidenceSource.SENTIMENT,
                direction=EvidenceDirection.SUPPORTING if supports else EvidenceDirection.OPPOSING,
                strength=EvidenceStrength.STRONG,
                claim=f"Fear/Greed at {score:.0f} ({fg.get('label', '')}) — {contrarian}",
                data=fg,
                reliability_score=0.60,
                recency_hours=0.5,
                relevance_score=0.85,
            ))

    # Divergences
    for div in sentiment_data.get("divergences", []):
        if div.get("instrument") == instrument:
            div_dir = div.get("direction", "")
            is_long = direction == "LONG"
            # Bearish divergence supports SHORT, bullish divergence supports LONG
            supports = (is_long and div_dir == "bullish_divergence") or (
                not is_long and div_dir == "bearish_divergence"
            )
            evidence.append(Evidence(
                id=f"sent_divergence_{div.get('type', '')}",
                source=EvidenceSource.SENTIMENT,
                direction=EvidenceDirection.SUPPORTING if supports else EvidenceDirection.OPPOSING,
                strength=EvidenceStrength.STRONG if div.get("severity", 0) > 70 else EvidenceStrength.MODERATE,
                claim=f"Sentiment-price divergence: {div_dir} (severity {div.get('severity', 0):.0f})",
                data=div,
                reliability_score=0.65,
                recency_hours=0.5,
                relevance_score=0.90,
            ))

    # COT positioning
    cot = sentiment_data.get("cot_analysis", {}).get(instrument, {})
    if cot.get("contrarian_signal", "none") != "none":
        is_long = direction == "LONG"
        supports = (is_long and cot["contrarian_signal"] == "contrarian_buy") or (
            not is_long and cot["contrarian_signal"] == "contrarian_sell"
        )
        evidence.append(Evidence(
            id="sent_cot_contrarian",
            source=EvidenceSource.POSITIONING,
            direction=EvidenceDirection.SUPPORTING if supports else EvidenceDirection.OPPOSING,
            strength=EvidenceStrength.STRONG,
            claim=f"COT extreme: {cot['contrarian_signal']} (reversal prob {cot.get('reversal_probability', 0):.0%})",
            data=cot,
            reliability_score=0.62,
            recency_hours=48.0,  # Weekly data
            relevance_score=0.80,
        ))

    # Reflexivity
    for ref in sentiment_data.get("reflexivity", []):
        if ref.get("instrument") == instrument:
            # Reflexivity signal: if fundamental_support is False and phase is exhausting,
            # it means the current move is about to reverse
            if not ref.get("fundamental_support") or ref.get("phase") == "exhausting":
                is_long = direction == "LONG"
                loop_is_up = ref.get("loop_type") == "bubble"
                # If it's a bubble exhausting and we want to go long → bad
                opposes = (is_long and loop_is_up) or (not is_long and not loop_is_up)
                evidence.append(Evidence(
                    id="sent_reflexivity",
                    source=EvidenceSource.SENTIMENT,
                    direction=EvidenceDirection.OPPOSING if opposes else EvidenceDirection.SUPPORTING,
                    strength=EvidenceStrength.STRONG,
                    claim=f"Reflexivity: {ref.get('loop_type', '')} in {ref.get('phase', '')} phase, fundamental support: {ref.get('fundamental_support')}",
                    data=ref,
                    reliability_score=0.55,
                    recency_hours=0.5,
                    relevance_score=0.85,
                ))

    return evidence
