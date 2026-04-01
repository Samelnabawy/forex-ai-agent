"""
Macro Analyst prompt templates.

Versioned for A/B testing during paper trading.
Each prompt version is a complete system+user prompt pair.

The system prompt defines the analyst's personality and methodology.
The user prompt provides current data and requests structured analysis.
"""

from __future__ import annotations

from typing import Any

# ── Prompt Version Registry ───────────────────────────────────

CURRENT_VERSION = "v2"

# ── System Prompts ────────────────────────────────────────────

SYSTEM_PROMPTS: dict[str, str] = {
    "v1": """You are an elite institutional macro analyst at a top-tier forex trading desk.
You analyze global macroeconomic conditions to determine the fundamental direction 
and strength of major currencies.

Your methodology:
1. Analyze central bank policy stances (hawkish/dovish) and rate differentials
2. Evaluate economic data relative to expectations and trend
3. Assess yield curve shape and bond market signals
4. Factor in geopolitical risk and safe-haven flows
5. Consider commodity price impacts on commodity-linked currencies
6. Determine the macro regime (risk-on, risk-off, transitioning)

You think in terms of MONETARY POLICY DIVERGENCE — what matters is not whether 
a central bank is hawkish, but whether it is MORE hawkish than the OTHER central 
bank in a currency pair.

You score each currency from -5 (extremely bearish) to +5 (extremely bullish).
You always respond with valid JSON only.""",

    "v2": """You are an elite institutional macro analyst at a top-tier forex trading desk 
managing a portfolio across 9 instruments: EUR/USD, GBP/USD, USD/JPY, USD/CHF, 
AUD/USD, USD/CAD, NZD/USD, XAU/USD (Gold), and WTI Crude Oil.

## Your Analytical Framework

TIER 1 — MONETARY POLICY DIVERGENCE (highest weight):
- What matters is the RATE DIFFERENTIAL between two central banks, not absolute levels
- A 25bp cut by the Fed when ECB is on hold = EUR bullish, even if ECB rates are lower
- Forward guidance matters more than the current rate — markets price expectations
- Use yield curve shape as a market-derived signal for rate expectations

TIER 2 — ECONOMIC MOMENTUM:
- Track data RELATIVE TO EXPECTATIONS, not absolute values
- A strong NFP that was expected is neutral; a weak NFP that was expected is also neutral
- Build an "economic surprise" mental model per economy
- PMI trends (3-month direction) matter more than single prints

TIER 3 — RISK SENTIMENT & FLOWS:
- VIX > 25 = risk-off regime (JPY, CHF, Gold strengthen; AUD, NZD weaken)
- VIX < 15 = complacent (carry trades work; JPY weakest)
- Yield curve inversion = recession signal = medium-term risk-off
- Month-end / quarter-end rebalancing flows can dominate for 2-3 days

TIER 4 — COMMODITY LINKAGES:
- Oil → USD/CAD (inverse: oil up → CAD strong → USD/CAD down)
- Iron ore → AUD/USD (positive: iron up → AUD up)
- Real rates → XAU/USD (inverse: real rates up → gold down)
- Copper → global growth proxy → risk sentiment

TIER 5 — GEOPOLITICAL:
- War/conflict escalation → safe havens (JPY, CHF, Gold)
- Sanctions → commodity supply disruption → oil/gold
- Election uncertainty → currency volatility premium

## Output Rules
- Score each currency -5 to +5 with clear reasoning
- Identify macro regime with confidence level
- Flag any upcoming events that could shift the regime
- When uncertain, say so — false confidence is worse than admitting uncertainty
- Always respond with valid JSON only, no markdown""",
}


# ── User Prompt Builder ───────────────────────────────────────

def build_macro_analysis_prompt(
    yield_curve: dict[str, Any],
    rate_differentials: dict[str, Any],
    real_rates: dict[str, Any],
    economic_data: dict[str, Any],
    commodity_data: dict[str, Any],
    session_context: dict[str, Any],
    recent_news: list[str],
    dxy: float | None = None,
    vix: float | None = None,
    prices: dict[str, Any] | None = None,
    version: str = CURRENT_VERSION,
) -> tuple[str, str]:
    """
    Build the complete system + user prompt for macro analysis.
    Returns (system_prompt, user_prompt).
    """
    system = SYSTEM_PROMPTS.get(version, SYSTEM_PROMPTS[CURRENT_VERSION])

    # Build context sections
    sections: list[str] = []

    # 1. Bond yields and curve
    sections.append("## BOND YIELDS & YIELD CURVE")
    if yield_curve:
        yields = yield_curve.get("yields", {})
        for name, value in yields.items():
            if value is not None:
                sections.append(f"  {name}: {value}%")
        curve_shape = yield_curve.get("curve_shape", "unknown")
        spread = yield_curve.get("spread_2y_10y")
        sections.append(f"  2Y-10Y Spread: {spread}% ({curve_shape})")
        if yield_curve.get("inversion"):
            sections.append("  ⚠️ YIELD CURVE INVERTED — recession signal")

    # 2. Rate differentials
    sections.append("\n## RATE DIFFERENTIALS")
    if rate_differentials:
        for pair, data in rate_differentials.items():
            diff = data.get("differential", "N/A")
            carry = data.get("carry_direction", "N/A")
            sections.append(f"  {pair}: {diff}% differential → carry favors {carry}")

    # 3. Real rates (gold driver)
    sections.append("\n## REAL INTEREST RATES")
    if real_rates:
        for name, value in real_rates.items():
            if value is not None:
                sections.append(f"  {name}: {value}%")

    # 4. Market indices
    sections.append("\n## MARKET INDICES")
    if dxy is not None:
        sections.append(f"  DXY (Dollar Index): {dxy}")
    if vix is not None:
        sections.append(f"  VIX (Volatility): {vix}")
        if vix > 25:
            sections.append("  ⚠️ VIX ELEVATED — risk-off conditions")
        elif vix < 15:
            sections.append("  VIX low — complacent market, carry trades favorable")

    # 5. Commodity context
    sections.append("\n## COMMODITY PRICES")
    if commodity_data:
        oil = commodity_data.get("oil", {})
        if oil.get("wti_price"):
            sections.append(f"  WTI Crude: ${oil['wti_price']}")
        if oil.get("brent_price"):
            sections.append(f"  Brent Crude: ${oil['brent_price']}")
        commodities = commodity_data.get("commodities", {})
        for name, value in commodities.items():
            if value is not None:
                sections.append(f"  {name}: {value}")
        interpretation = commodity_data.get("interpretation", {})
        for pair, note in interpretation.items():
            sections.append(f"  → {pair}: {note}")

    # 6. Economic data
    sections.append("\n## ECONOMIC DATA (LATEST)")
    if economic_data:
        for country, data in economic_data.items():
            if isinstance(data, dict):
                sections.append(f"  {country}:")
                for indicator, value in data.items():
                    if value is not None:
                        sections.append(f"    {indicator}: {value}")
            elif data is not None:
                sections.append(f"  {country}: {data}")

    # 7. Current prices
    if prices:
        sections.append("\n## CURRENT PRICES")
        for symbol, price_data in prices.items():
            mid = price_data.get("mid", "N/A")
            sections.append(f"  {symbol}: {mid}")

    # 8. Session context
    sections.append("\n## SESSION CONTEXT")
    if session_context:
        sections.append(f"  Session: {session_context.get('active_session', 'unknown')}")
        sections.append(f"  Day type: {session_context.get('day_type', 'normal')}")
        for note in session_context.get("special_notes", []):
            sections.append(f"  ⚠️ {note}")
        for meeting in session_context.get("upcoming_cb_meetings", []):
            sections.append(f"  📅 Upcoming: {meeting.get('bank', '')} on {meeting.get('date', '')}")

    # 9. Recent news
    if recent_news:
        sections.append("\n## RECENT NEWS HEADLINES (last 6 hours)")
        for headline in recent_news[:15]:  # Limit to 15 headlines
            sections.append(f"  • {headline}")

    context = "\n".join(sections)

    user_prompt = f"""Analyze the current macro environment and provide your assessment.

{context}

Respond with this exact JSON structure:
{{
    "macro_regime": "risk_on" | "risk_off" | "neutral" | "transitioning",
    "regime_confidence": 0.0-1.0,
    "usd_outlook": "bullish" | "bearish" | "neutral" + "_short_term" or "_medium_term",
    "reasoning": "2-3 sentence summary of your macro thesis",
    "currency_scores": {{
        "USD": -5 to +5,
        "EUR": -5 to +5,
        "GBP": -5 to +5,
        "JPY": -5 to +5,
        "CHF": -5 to +5,
        "AUD": -5 to +5,
        "CAD": -5 to +5,
        "NZD": -5 to +5
    }},
    "score_reasoning": {{
        "USD": "brief reason",
        "EUR": "brief reason",
        "GBP": "brief reason",
        "JPY": "brief reason",
        "CHF": "brief reason",
        "AUD": "brief reason",
        "CAD": "brief reason",
        "NZD": "brief reason"
    }},
    "preferred_pairs": [
        {{"pair": "EUR/USD", "direction": "LONG"|"SHORT", "conviction": "high"|"medium"|"low", "reason": "..."}}
    ],
    "gold_outlook": "bullish|bearish|neutral + brief reason",
    "oil_outlook": "bullish|bearish|neutral + brief reason",
    "risk_events": [
        {{"event": "name", "time": "when", "impact": "HIGH|MEDIUM", "scenario": "what could happen"}}
    ],
    "regime_change_risk": "low" | "medium" | "high",
    "regime_change_trigger": "what would cause a regime shift"
}}"""

    return system, user_prompt


# ── Event Impact Prompt ───────────────────────────────────────

EVENT_IMPACT_SYSTEM = """You are a forex event analyst. Given an economic release,
assess its immediate impact on currencies. Consider:
1. Actual vs Expected (surprise factor)
2. Revision to previous data
3. Trend context (is this confirming or contradicting the trend?)
4. Market positioning (was the market already positioned for this?)

Respond with JSON only."""


def build_event_impact_prompt(
    event_name: str,
    actual: str,
    forecast: str,
    previous: str,
    currency: str,
) -> tuple[str, str]:
    """Build prompt for quick event impact assessment (uses Haiku)."""
    user_prompt = f"""Economic release just occurred:

Event: {event_name}
Actual: {actual}
Forecast: {forecast}
Previous: {previous}
Primary currency: {currency}

Respond with:
{{
    "surprise": "positive" | "negative" | "inline",
    "magnitude": "large" | "moderate" | "small",
    "currency_impact": {{
        "{currency}": "bullish" | "bearish" | "neutral"
    }},
    "affected_pairs": ["pair1", "pair2"],
    "expected_move_pips": 0-100,
    "duration": "minutes" | "hours" | "days",
    "reasoning": "brief explanation"
}}"""

    return EVENT_IMPACT_SYSTEM, user_prompt
