"""
Sentiment Agent prompt templates.

Three LLM-powered analysis modes:
  1. Headline Batch Classification (Haiku) — classify 10-15 headlines at once
  2. Narrative Lifecycle Detection (Haiku) — detect where a narrative is in its lifecycle
  3. Central Bank Statement Delta (Sonnet) — diff two statements for hawkish/dovish shift
"""

from __future__ import annotations

from typing import Any

# ── 1. Headline Batch Classification ──────────────────────────

HEADLINE_SYSTEM = """You are a forex news analyst at an institutional trading desk.
Classify each headline for its impact on currencies.

Rules:
- Score each affected currency -3 (very bearish) to +3 (very bullish)
- Weight by source credibility: Reuters/Bloomberg/FT = high, Forex Factory = medium, blogs = low
- Consider second-order effects (oil headline affects CAD, not just oil)
- "Priced in" vs "surprise" matters: expected news = weak signal, surprising = strong signal
- Track narrative consistency: if 8/10 headlines say the same thing, that's consensus (contrarian risk)

Respond with JSON only."""

HEADLINE_USER_TEMPLATE = """Classify these headlines for currency impact:

{headlines}

Respond with:
{{
    "classifications": [
        {{
            "headline": "the headline text",
            "primary_currency": "USD",
            "scores": {{"USD": -2, "EUR": +1, "JPY": 0}},
            "credibility": "high" | "medium" | "low",
            "surprise_factor": "surprise" | "expected" | "unknown",
            "category": "central_bank" | "economic_data" | "geopolitical" | "trade_policy" | "market_sentiment"
        }}
    ],
    "overall_narrative": "one sentence describing the dominant theme",
    "narrative_consistency": 0.0-1.0,
    "consensus_risk": "none" | "moderate" | "high"
}}"""


def build_headline_classification_prompt(
    headlines: list[dict[str, str]],
) -> tuple[str, str]:
    """Build prompt for batch headline classification."""
    formatted = []
    for i, h in enumerate(headlines[:15], 1):
        source = h.get("source", "unknown")
        text = h.get("headline", "")
        formatted.append(f"{i}. [{source}] {text}")

    user = HEADLINE_USER_TEMPLATE.format(headlines="\n".join(formatted))
    return HEADLINE_SYSTEM, user


# ── 2. Narrative Lifecycle Detection ──────────────────────────

NARRATIVE_SYSTEM = """You are a market narrative analyst. Track how market narratives evolve:

LIFECYCLE STAGES:
1. EMERGENCE — A few analysts mention it, market hasn't moved on it yet. THIS IS THE BEST ENTRY.
2. ADOPTION — Major news outlets pick it up, price starts moving. Still tradeable.
3. CONSENSUS — Everyone agrees. Positioning is building. Risk/reward deteriorating.
4. PEAK — Maximum positioning, maximum headlines, maximum confidence. PREPARE TO EXIT.
5. DOUBT — One contradicting data point. First cracks. Tighten stops.
6. REVERSAL — Positioning unwinds, narrative dies. Go contrarian or flat.

Your job is to identify the TOP 3 active narratives and classify their lifecycle stage.
Respond with JSON only."""

NARRATIVE_USER_TEMPLATE = """Based on these recent headlines and market context:

HEADLINES (last 24 hours):
{headlines}

MARKET CONTEXT:
- DXY: {dxy}
- VIX: {vix}
- Key currency moves (24h): {currency_moves}

Identify the top active narratives:
{{
    "narratives": [
        {{
            "name": "short descriptive name (e.g. 'Fed pivot', 'EUR hawkish ECB', 'Risk-off China')",
            "stage": "emergence" | "adoption" | "consensus" | "peak" | "doubt" | "reversal",
            "stage_confidence": 0.0-1.0,
            "affected_currencies": {{"USD": -2, "EUR": +1}},
            "evidence": ["headline or data point supporting this classification"],
            "duration_estimate": "days" | "weeks" | "months",
            "trade_implication": "what a trader should do given this stage"
        }}
    ],
    "dominant_narrative": "name of the most important one right now",
    "narrative_conflicts": ["any narratives that contradict each other"]
}}"""


def build_narrative_detection_prompt(
    headlines: list[str],
    dxy: float | None = None,
    vix: float | None = None,
    currency_moves: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Build prompt for narrative lifecycle detection."""
    headlines_text = "\n".join(f"- {h}" for h in headlines[:20])
    moves_text = ", ".join(
        f"{k}: {v:+.2f}%" for k, v in (currency_moves or {}).items()
    ) or "N/A"

    user = NARRATIVE_USER_TEMPLATE.format(
        headlines=headlines_text,
        dxy=dxy or "N/A",
        vix=vix or "N/A",
        currency_moves=moves_text,
    )
    return NARRATIVE_SYSTEM, user


# ── 3. Central Bank Statement Delta ───────────────────────────

CB_DELTA_SYSTEM = """You are a central bank language specialist. Compare two consecutive
statements from the same central bank and quantify the shift.

SCORING SCALE (-5 to +5):
-5: Massive dovish shift (emergency cut language, panic)
-3: Clear dovish shift (new language suggesting cuts)
-1: Slight dovish tilt (softer adjectives, less urgency)
 0: No meaningful change
+1: Slight hawkish tilt (firmer adjectives, more urgency)
+3: Clear hawkish shift (new language suggesting hikes)
+5: Massive hawkish shift (emergency tightening, inflation panic)

KEY PHRASES TO WATCH:
- "patient" → "vigilant" = hawkish shift (+2)
- "data-dependent" → "clear direction" = resolving uncertainty
- "gradually" → "expeditiously" = acceleration signal
- "transitory" → "persistent" = inflation concern (+3)
- Adding/removing "some" or "further" before "tightening" or "easing"
- Changes in growth/inflation balance of risks

Respond with JSON only."""

CB_DELTA_USER_TEMPLATE = """Compare these two consecutive statements from {bank}:

PREVIOUS STATEMENT ({previous_date}):
{previous_statement}

CURRENT STATEMENT ({current_date}):
{current_statement}

Respond with:
{{
    "bank": "{bank}",
    "hawkish_dovish_shift": -5 to +5,
    "shift_magnitude": "none" | "slight" | "moderate" | "significant" | "massive",
    "key_changes": [
        {{
            "previous_phrase": "exact phrase from previous",
            "current_phrase": "exact phrase from current",
            "interpretation": "what this shift means",
            "impact_score": -3 to +3
        }}
    ],
    "rate_path_implication": "cut_likely" | "hold" | "hike_likely" | "uncertain",
    "currency_impact": {{
        "{currency}": -3 to +3
    }},
    "market_surprise_potential": "low" | "medium" | "high",
    "reasoning": "2-3 sentence summary of the shift"
}}"""


def build_cb_delta_prompt(
    bank: str,
    currency: str,
    previous_statement: str,
    current_statement: str,
    previous_date: str,
    current_date: str,
) -> tuple[str, str]:
    """Build prompt for central bank statement delta analysis."""
    user = CB_DELTA_USER_TEMPLATE.format(
        bank=bank,
        currency=currency,
        previous_statement=previous_statement[:3000],  # Token limit
        current_statement=current_statement[:3000],
        previous_date=previous_date,
        current_date=current_date,
    )
    return CB_DELTA_SYSTEM, user


# ── 4. Event Scenario Prompt ─────────────────────────────────

EVENT_SCENARIO_SYSTEM = """You are a forex event strategist. For an upcoming high-impact event,
generate specific scenarios with pip estimates.

Consider:
1. Where is the market positioned going into this event? (COT data, price action)
2. What is the consensus expectation?
3. What would genuinely surprise the market?
4. Historical precedent: what happened last time?

Respond with JSON only."""

EVENT_SCENARIO_USER_TEMPLATE = """Upcoming event analysis:

EVENT: {event_name}
TIME: {event_time}
CURRENCY: {currency}
CONSENSUS FORECAST: {forecast}
PREVIOUS: {previous}

MARKET CONTEXT:
- Current positioning: {positioning}
- Price action leading into event: {price_action}
- Attention regime: {attention_regime}

Generate scenario analysis:
{{
    "scenarios": {{
        "strong_beat": {{
            "threshold": ">X",
            "probability": 0.0-1.0,
            "immediate_reaction": {{"pair": "USD/XXX", "direction": "up/down", "pips": 30-100}},
            "secondary_effects": ["effect1", "effect2"],
            "duration": "minutes/hours/days"
        }},
        "moderate_beat": {{}},
        "inline": {{}},
        "moderate_miss": {{}},
        "strong_miss": {{}}
    }},
    "highest_impact_scenario": "which scenario would move markets most and why",
    "positioning_risk": "what happens if the market is wrong-footed",
    "recommended_approach": "how to trade this event"
}}"""


def build_event_scenario_prompt(
    event_name: str,
    event_time: str,
    currency: str,
    forecast: str,
    previous: str,
    positioning: str = "neutral",
    price_action: str = "range-bound",
    attention_regime: str = "rate_cycle",
) -> tuple[str, str]:
    """Build prompt for pre-event scenario analysis."""
    user = EVENT_SCENARIO_USER_TEMPLATE.format(
        event_name=event_name,
        event_time=event_time,
        currency=currency,
        forecast=forecast,
        previous=previous,
        positioning=positioning,
        price_action=price_action,
        attention_regime=attention_regime,
    )
    return EVENT_SCENARIO_SYSTEM, user
