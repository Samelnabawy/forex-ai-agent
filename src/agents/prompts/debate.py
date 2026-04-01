"""
Bull/Bear Debate Prompt Templates.

Three-phase adversarial debate:
  Phase 1: Bull builds the case (with all supporting evidence)
  Phase 2: Bear attacks the case (receives bull's full argument)
  Phase 3: Verdict synthesis (weigh both sides, produce final recommendation)

The LLM's job is NOT to decide — it's to ARGUE well.
The decision comes from the evidence weights and expected value math.
"""

from __future__ import annotations

from typing import Any


# ── Phase 1: Bull Case ────────────────────────────────────────

BULL_SYSTEM = """You are a senior forex trader building the STRONGEST POSSIBLE CASE 
for taking a trade. You are the advocate — find every reason this trade works.

Rules:
1. Use ONLY the evidence provided — no hallucinated data
2. Weight evidence by its reliability score and recency
3. Connect evidence across agents (technical + macro + correlation confirming = stronger)
4. Identify the SINGLE most compelling reason this trade works
5. Be specific about entry timing and expected duration
6. State what would INVALIDATE your thesis (intellectual honesty)
7. Calculate expected value explicitly

You are arguing FOR the trade, but you must be honest about weaknesses.
A good advocate acknowledges the 2-3 biggest risks and explains why they're manageable.

Respond with JSON only."""


BULL_USER_TEMPLATE = """Build the bull case for this trade:

TRADE PROPOSAL:
  Instrument: {instrument}
  Direction: {direction}
  Entry: {entry}
  Stop Loss: {stop_loss}
  Take Profit 1: {take_profit_1}
  Take Profit 2: {take_profit_2}
  Risk/Reward: {risk_reward}

SUPPORTING EVIDENCE:
{evidence_text}

HISTORICAL PRECEDENTS (similar setups that WORKED):
{precedents_text}

CURRENT MARKET CONTEXT:
  Macro Regime: {macro_regime}
  VIX: {vix}
  Session: {session}
  Day Type: {day_type}
  Fear/Greed: {fear_greed}

Respond with:
{{
    "case": "BULLISH",
    "instrument": "{instrument}",
    "direction": "{direction}",
    "thesis": "2-3 sentence core thesis",
    "single_best_reason": "the ONE most compelling evidence point",
    "evidence_synthesis": [
        {{
            "evidence_id": "tech_rsi",
            "importance": "critical" | "supporting" | "minor",
            "how_it_fits": "how this evidence supports the thesis"
        }}
    ],
    "confluence_narrative": "how multiple independent signals align to create edge",
    "entry_timing": {{
        "optimal_entry": "exact level and why",
        "worst_entry": "don't enter above/below this",
        "expected_duration": "hours/days for the trade to play out"
    }},
    "expected_value": {{
        "win_probability": 0.0-1.0,
        "reasoning": "why this probability"
    }},
    "risks_acknowledged": [
        {{
            "risk": "description",
            "mitigation": "how to manage it",
            "probability": 0.0-1.0
        }}
    ],
    "invalidation_conditions": [
        "If X happens, this thesis is DEAD — exit immediately"
    ],
    "confidence": 0.0-1.0,
    "recommended_position_size": "full (1%) | standard (0.75%) | reduced (0.5%)",
    "reasoning_summary": "complete argument in 4-5 sentences"
}}"""


# ── Phase 2: Bear Case ───────────────────────────────────────

BEAR_SYSTEM = """You are a senior risk analyst whose job is to DESTROY the bull case.
You must find every flaw, every risk, every reason this trade could fail.

Rules:
1. Attack SPECIFIC evidence from the bull case — don't make generic objections
2. For each bull evidence point, explain why it might be wrong or misleading
3. Find risks the bull didn't mention
4. Use the counter-evidence and failure precedents provided
5. If the trade is genuinely good, say so — but even good trades have risks
6. Propose alternatives: "Don't trade now, but if X happens, THEN trade"
7. Identify the "kill zone" — the price level where the trade is definitively wrong

You are NOT anti-trade by default. Your job is truth-seeking through adversarial testing.
If the bull case is genuinely strong, your counter-case should be specific and limited.

Respond with JSON only."""


BEAR_USER_TEMPLATE = """Attack this bull case:

BULL'S ARGUMENT:
{bull_case_text}

COUNTER-EVIDENCE (reasons AGAINST the trade):
{counter_evidence_text}

HISTORICAL PRECEDENTS (similar setups that FAILED):
{failure_precedents_text}

ADDITIONAL RISKS:
  Correlation warnings: {correlation_warnings}
  Upcoming events: {upcoming_events}
  Seasonal factors: {seasonal_factors}
  Positioning risk: {positioning_risk}
  Attention regime: {attention_regime}

Respond with:
{{
    "case": "BEARISH",
    "instrument": "{instrument}",
    "evidence_attacks": [
        {{
            "target_evidence_id": "tech_rsi",
            "attack": "why this evidence is weak or misleading",
            "attack_strength": "devastating" | "significant" | "minor"
        }}
    ],
    "missed_risks": [
        {{
            "risk": "something the bull didn't consider",
            "severity": "critical" | "moderate" | "low",
            "probability": 0.0-1.0
        }}
    ],
    "failure_scenarios": [
        {{
            "scenario": "what goes wrong",
            "trigger": "what causes it",
            "expected_loss_pips": 0,
            "probability": 0.0-1.0
        }}
    ],
    "kill_zone": {{
        "price_level": "the price where the trade is definitively wrong",
        "reasoning": "why this level invalidates the thesis"
    }},
    "alternative_action": {{
        "recommendation": "wait" | "trade_opposite" | "modify" | "proceed_with_caution",
        "condition": "if X happens, then Y",
        "reasoning": "why this is better than the bull's plan"
    }},
    "confidence_against": 0.0-1.0,
    "honest_assessment": "Is the bull case genuinely strong despite my objections? 1-2 sentences.",
    "reasoning_summary": "complete counter-argument in 4-5 sentences"
}}"""


# ── Phase 3: Verdict Synthesis ────────────────────────────────

VERDICT_SYSTEM = """You are the chief strategist synthesizing a bull/bear debate into a final verdict.
You are OBJECTIVE — not biased toward either side.

Your job:
1. Identify points where both sides AGREE (consensus)
2. Identify points where they DISAGREE (contested)
3. Weigh the evidence quality on each side (reliability scores, recency, confluence)
4. Determine if the bull adequately addressed the bear's concerns
5. Make a DECISIVE recommendation — no fence-sitting

The expected value calculation has already been done. Your job is to assess
whether the qualitative arguments change the quantitative conclusion.

Respond with JSON only."""


VERDICT_USER_TEMPLATE = """Synthesize this debate:

BULL CASE:
{bull_summary}

BEAR CASE:
{bear_summary}

QUANTITATIVE ANALYSIS:
  Win Probability: {win_probability}
  Expected Value: {expected_value} pips
  Risk/Reward: {risk_reward}
  EV per risk unit: {ev_per_risk}

EVIDENCE WEIGHTS:
  Bull total weight: {bull_weight}
  Bear total weight: {bear_weight}
  Evidence survived rebuttal: {survived_count}/{total_count}

Respond with:
{{
    "verdict": "EXECUTE" | "SKIP" | "WAIT" | "MODIFY",
    "consensus_points": ["points both sides agree on"],
    "contested_points": [
        {{
            "point": "the disagreement",
            "bull_argument": "...",
            "bear_argument": "...",
            "who_is_stronger": "bull" | "bear" | "unclear"
        }}
    ],
    "unresolved_risks": ["risks that neither side could fully address"],
    "conviction_score": 0-100,
    "position_size_recommendation": "full (1%) | standard (0.75%) | reduced (0.5%) | skip (0%)",
    "modifications": {{
        "adjust_stop": "wider/tighter stop and why",
        "adjust_entry": "different entry level and why",
        "adjust_timing": "wait for X before entering"
    }},
    "reasoning": "3-4 sentence definitive statement on why this verdict"
}}"""


# ── Prompt Builders ───────────────────────────────────────────

def build_bull_prompt(
    proposal: dict[str, Any],
    evidence: list[dict[str, Any]],
    precedents: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[str, str]:
    """Build the bull case prompt."""
    evidence_text = "\n".join(
        f"  [{e.get('id', '?')}] ({e.get('strength', '?')}, reliability={e.get('reliability_score', 0):.2f}) "
        f"{e.get('claim', '')}"
        for e in evidence
    ) or "  No direct evidence available"

    precedents_text = "\n".join(
        f"  [{p.get('date', '?')}] (similarity={p.get('similarity_score', 0):.2f}) "
        f"{p.get('event_description', '')} → {p.get('outcome', '?')} ({p.get('pnl_pips', 0):+.0f} pips)"
        for p in precedents
    ) or "  No historical precedents available"

    risk_pips = abs(proposal.get("entry", 0) - proposal.get("stop_loss", 0))
    reward_pips = abs(proposal.get("take_profit_1", 0) - proposal.get("entry", 0))
    rr = round(reward_pips / risk_pips, 2) if risk_pips > 0 else 0

    user = BULL_USER_TEMPLATE.format(
        instrument=proposal.get("instrument", "?"),
        direction=proposal.get("direction", "?"),
        entry=proposal.get("entry", 0),
        stop_loss=proposal.get("stop_loss", 0),
        take_profit_1=proposal.get("take_profit_1", 0),
        take_profit_2=proposal.get("take_profit_2", "N/A"),
        risk_reward=rr,
        evidence_text=evidence_text,
        precedents_text=precedents_text,
        macro_regime=context.get("macro_regime", "neutral"),
        vix=context.get("vix", "N/A"),
        session=context.get("session", "unknown"),
        day_type=context.get("day_type", "normal"),
        fear_greed=context.get("fear_greed", "N/A"),
    )

    return BULL_SYSTEM, user


def build_bear_prompt(
    instrument: str,
    bull_case: dict[str, Any],
    counter_evidence: list[dict[str, Any]],
    failure_precedents: list[dict[str, Any]],
    risk_context: dict[str, Any],
) -> tuple[str, str]:
    """Build the bear counter-case prompt."""
    bull_text = bull_case.get("reasoning_summary", str(bull_case)[:2000])

    counter_text = "\n".join(
        f"  [{e.get('id', '?')}] ({e.get('strength', '?')}) {e.get('claim', '')}"
        for e in counter_evidence
    ) or "  No direct counter-evidence"

    failure_text = "\n".join(
        f"  [{p.get('date', '?')}] (similarity={p.get('similarity_score', 0):.2f}) "
        f"{p.get('event_description', '')} → LOSS ({p.get('pnl_pips', 0):+.0f} pips)"
        for p in failure_precedents
    ) or "  No failure precedents found"

    user = BEAR_USER_TEMPLATE.format(
        instrument=instrument,
        bull_case_text=bull_text,
        counter_evidence_text=counter_text,
        failure_precedents_text=failure_text,
        correlation_warnings=risk_context.get("correlation_warnings", "None"),
        upcoming_events=risk_context.get("upcoming_events", "None"),
        seasonal_factors=risk_context.get("seasonal_factors", "None"),
        positioning_risk=risk_context.get("positioning_risk", "None"),
        attention_regime=risk_context.get("attention_regime", "mixed"),
    )

    return BEAR_SYSTEM, user


def build_verdict_prompt(
    bull_case: dict[str, Any],
    bear_case: dict[str, Any],
    ev_calc: dict[str, Any],
    evidence_stats: dict[str, Any],
) -> tuple[str, str]:
    """Build the verdict synthesis prompt."""
    user = VERDICT_USER_TEMPLATE.format(
        bull_summary=bull_case.get("reasoning_summary", ""),
        bear_summary=bear_case.get("reasoning_summary", ""),
        win_probability=ev_calc.get("win_probability", 0.5),
        expected_value=ev_calc.get("expected_value_pips", 0),
        risk_reward=ev_calc.get("risk_reward_ratio", 0),
        ev_per_risk=ev_calc.get("ev_per_risk_unit", 0),
        bull_weight=evidence_stats.get("bull_weight", 0),
        bear_weight=evidence_stats.get("bear_weight", 0),
        survived_count=evidence_stats.get("survived_count", 0),
        total_count=evidence_stats.get("total_count", 0),
    )

    return VERDICT_SYSTEM, user
