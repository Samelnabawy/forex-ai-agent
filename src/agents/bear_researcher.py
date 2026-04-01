"""
Bear Researcher — Agent 06

"The Skeptic" — finds every flaw, every risk, every historical failure case.
Not anti-trade by default — truth-seeking through adversarial testing.

Process:
  1. Receives the SAME trade proposal AND the Bull's complete case
  2. Attacks EACH specific piece of bull evidence
  3. Queries Historical Brain for precedents that FAILED
  4. Identifies risks the Bull missed (correlation, event, positioning, seasonal)
  5. LLM builds the counter-case targeting Bull's weakest points
  6. Proposes alternatives: "wait for X" or "modify entry/stop"
  7. Identifies the "kill zone" — price where trade is definitively wrong

The Bear's job is to stress-test, not to veto.
If the bull case is genuinely strong, the bear should say so.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar

from src.agents.base_agent import BaseAgent
from src.agents.evidence import (
    Evidence,
    EvidenceDirection,
    EvidencePackage,
    EvidenceSource,
    EvidenceStrength,
    HistoricalPrecedent,
    TradeProposal,
    aggregate_evidence,
)
from src.agents.llm_client import get_llm_client
from src.agents.prompts.debate import build_bear_prompt
from src.data.storage.cache import get_agent_state
from src.models import MarketState

logger = logging.getLogger(__name__)


class BearResearcherAgent(BaseAgent):
    """
    Agent 06: Bear Researcher

    Builds the strongest possible case AGAINST a trade.
    Receives the Bull's case and attacks it specifically.
    """

    name: ClassVar[str] = "bear_researcher"
    description: ClassVar[str] = "Builds strongest case AGAINST a trade, stress-tests thesis"
    execution_frequency: ClassVar[str] = "on_demand"

    def __init__(self) -> None:
        super().__init__()
        self._llm = get_llm_client()

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """Not used directly — Bear is triggered by debate engine."""
        return {"status": "awaiting_bull_case"}

    async def build_counter_case(
        self,
        proposal: TradeProposal,
        bull_package: EvidencePackage,
        market_state: MarketState,
    ) -> EvidencePackage:
        """
        Build the bear counter-case.
        Receives the bull's complete evidence package and attacks it.
        """
        self.logger.info(
            "Building bear counter-case",
            extra={
                "instrument": proposal.instrument,
                "direction": proposal.direction,
                "bull_conviction": bull_package.conviction_score,
            },
        )

        # 1. Gather counter-evidence (evidence AGAINST the trade)
        counter_evidence = await self._gather_counter_evidence(proposal, bull_package)

        # 2. Query Historical Brain for FAILURE precedents
        failure_precedents = await self._query_failure_precedents(proposal)

        # 3. Identify risks Bull missed
        risk_context = await self._build_risk_context(proposal, bull_package)

        # 4. Compute counter-evidence weights
        for e in counter_evidence:
            e.compute_weight()

        # 5. LLM: Attack the bull case
        bull_llm_case = getattr(bull_package, "llm_case", {}) if hasattr(bull_package, "__dict__") else {}
        if not bull_llm_case:
            bull_llm_case = {"reasoning_summary": bull_package.reasoning_summary}

        counter_dicts = [
            {"id": e.id, "claim": e.claim, "strength": e.strength.value, "source": e.source.value}
            for e in counter_evidence
        ]
        failure_dicts = [
            {"date": p.date, "event_description": p.event_description,
             "similarity_score": p.similarity_score, "pnl_pips": p.pnl_pips}
            for p in failure_precedents
        ]

        system, user = build_bear_prompt(
            instrument=proposal.instrument,
            bull_case=bull_llm_case,
            counter_evidence=counter_dicts,
            failure_precedents=failure_dicts,
            risk_context=risk_context,
        )

        llm_counter = await self._llm.analyze(
            system_prompt=system,
            user_prompt=user,
            agent_name=self.name,
            prompt_version="v1",
        )

        # 6. Apply rebuttals to bull evidence
        if not llm_counter.get("_fallback"):
            self._apply_rebuttals(bull_package, llm_counter)

        # 7. Build bear evidence package
        agg = aggregate_evidence(counter_evidence)
        confidence_against = llm_counter.get("confidence_against", 0.5) if not llm_counter.get("_fallback") else 0.3

        package = EvidencePackage(
            side="bear",
            trade_id=proposal.trade_id,
            instrument=proposal.instrument,
            direction="SHORT" if proposal.direction == "LONG" else "LONG",
            evidence=counter_evidence,
            precedents=failure_precedents,
            total_weight=agg.get("opposing_weight", 0),
            conviction_score=round(confidence_against * 100, 1),
            key_risks=[r.get("risk", "") for r in llm_counter.get("missed_risks", [])],
            alternative_actions=[llm_counter.get("alternative_action", {}).get("recommendation", "")],
            reasoning_summary=llm_counter.get("reasoning_summary", ""),
        )

        # Store LLM output
        package.__dict__["llm_case"] = llm_counter

        return package

    async def _gather_counter_evidence(
        self, proposal: TradeProposal, bull_package: EvidencePackage
    ) -> list[Evidence]:
        """
        Gather evidence that OPPOSES the trade.
        Invert the direction and re-extract from agents.
        """
        counter_evidence: list[Evidence] = []
        opposite_direction = "SHORT" if proposal.direction == "LONG" else "LONG"

        # Re-extract from agents with opposite direction perspective
        from src.agents.evidence import (
            extract_technical_evidence,
            extract_macro_evidence,
            extract_correlation_evidence,
            extract_sentiment_evidence,
        )

        # Technical: look for signals AGAINST the direction
        tech_state = await get_agent_state("technical_analyst")
        if tech_state:
            for sig in tech_state.get("signals", []):
                if sig.get("instrument") == proposal.instrument:
                    opposing = extract_technical_evidence(sig, opposite_direction)
                    for e in opposing:
                        e.direction = EvidenceDirection.OPPOSING
                    counter_evidence.extend(opposing)

        # Macro: any currency scores that oppose
        macro_state = await get_agent_state("macro_analyst")
        if macro_state:
            opposing = extract_macro_evidence(macro_state, proposal.instrument, opposite_direction)
            for e in opposing:
                if e.direction == EvidenceDirection.SUPPORTING:
                    e.direction = EvidenceDirection.OPPOSING  # Flip: supporting opposite = opposing original
            counter_evidence.extend(opposing)

        # Correlation: contradictions
        corr_state = await get_agent_state("correlation_agent")
        if corr_state:
            # Check for correlation warnings
            cv = corr_state.get("cross_validation", {})
            for signal_cv in cv.get("technical_signals", []):
                if signal_cv.get("instrument") == proposal.instrument:
                    if signal_cv.get("correlation_assessment") == "CONTRADICTED":
                        counter_evidence.append(Evidence(
                            id="bear_corr_contradiction",
                            source=EvidenceSource.CORRELATION,
                            direction=EvidenceDirection.OPPOSING,
                            strength=EvidenceStrength.STRONG,
                            claim=f"Correlation Agent CONTRADICTS this trade",
                            data=signal_cv,
                            reliability_score=0.70,
                            recency_hours=0.02,
                            relevance_score=0.95,
                        ))

            # Crowding risk
            crowding = corr_state.get("crowding_risk", {})
            for warning in crowding.get("crowding_warnings", []):
                if "CROWDED" in warning and proposal.instrument in warning:
                    counter_evidence.append(Evidence(
                        id="bear_crowding",
                        source=EvidenceSource.CORRELATION,
                        direction=EvidenceDirection.OPPOSING,
                        strength=EvidenceStrength.MODERATE,
                        claim=f"Crowding risk: {warning}",
                        data={"warning": warning},
                        reliability_score=0.65,
                        recency_hours=0.02,
                        relevance_score=0.85,
                    ))

            # Anomalies
            for anomaly in corr_state.get("anomalies", []):
                if anomaly.get("severity") in ("high", "extreme"):
                    pairs = [anomaly.get("pair_a", ""), anomaly.get("pair_b", "")]
                    if proposal.instrument in pairs:
                        counter_evidence.append(Evidence(
                            id=f"bear_anomaly_{anomaly.get('pair_a', '')}_{anomaly.get('pair_b', '')}",
                            source=EvidenceSource.CORRELATION,
                            direction=EvidenceDirection.OPPOSING,
                            strength=EvidenceStrength.MODERATE,
                            claim=f"Correlation anomaly: {anomaly.get('status', '')} (z={anomaly.get('z_score', 0)})",
                            data=anomaly,
                            reliability_score=0.60,
                            recency_hours=0.02,
                            relevance_score=0.75,
                        ))

        # Sentiment: opposing signals
        sent_state = await get_agent_state("sentiment_agent")
        if sent_state:
            opposing_sent = extract_sentiment_evidence(sent_state, proposal.instrument, opposite_direction)
            for e in opposing_sent:
                if e.direction == EvidenceDirection.SUPPORTING:
                    e.direction = EvidenceDirection.OPPOSING
            counter_evidence.extend(opposing_sent)

            # Upcoming events (calendar risk)
            attention = sent_state.get("attention", {})
            if attention.get("regime") == "crisis":
                counter_evidence.append(Evidence(
                    id="bear_crisis_regime",
                    source=EvidenceSource.SENTIMENT,
                    direction=EvidenceDirection.OPPOSING,
                    strength=EvidenceStrength.STRONG,
                    claim="Market in CRISIS attention regime — elevated uncertainty",
                    data=attention,
                    reliability_score=0.70,
                    recency_hours=0.5,
                    relevance_score=0.80,
                ))

        # News blackout risk
        try:
            from src.data.ingestion.calendar_feed import EconomicCalendarFeed
            cal = EconomicCalendarFeed()
            if await cal.is_news_blackout():
                counter_evidence.append(Evidence(
                    id="bear_news_blackout",
                    source=EvidenceSource.CALENDAR,
                    direction=EvidenceDirection.OPPOSING,
                    strength=EvidenceStrength.STRONG,
                    claim="NEWS BLACKOUT: High-impact event within 15 minutes",
                    data={"blackout": True},
                    reliability_score=0.90,
                    recency_hours=0,
                    relevance_score=1.0,
                ))
            await cal.close()
        except Exception:
            pass

        return counter_evidence

    def _apply_rebuttals(
        self, bull_package: EvidencePackage, bear_llm: dict[str, Any]
    ) -> None:
        """Apply bear's attacks to bull's evidence pieces."""
        attacks = bear_llm.get("evidence_attacks", [])
        attack_map = {a.get("target_evidence_id", ""): a for a in attacks}

        for evidence in bull_package.evidence:
            attack = attack_map.get(evidence.id)
            if attack:
                evidence.rebuttal = attack.get("attack", "")
                strength = attack.get("attack_strength", "minor")
                evidence.rebuttal_strength = {
                    "devastating": 0.9,
                    "significant": 0.6,
                    "minor": 0.3,
                }.get(strength, 0.3)
                evidence.survived_rebuttal = evidence.rebuttal_strength < 0.5

    async def _query_failure_precedents(
        self, proposal: TradeProposal
    ) -> list[HistoricalPrecedent]:
        """Query Historical Brain for similar setups that FAILED. Stub."""
        # TODO: Connect to brain/query_engine.py with outcome_filter="loss"
        return []

    async def _build_risk_context(
        self, proposal: TradeProposal, bull_package: EvidencePackage
    ) -> dict[str, Any]:
        """Build additional risk context for the bear prompt."""
        context: dict[str, Any] = {
            "correlation_warnings": "None",
            "upcoming_events": "None",
            "seasonal_factors": "None",
            "positioning_risk": "None",
            "attention_regime": "mixed",
        }

        corr = await get_agent_state("correlation_agent")
        if corr:
            warnings = corr.get("crowding_risk", {}).get("crowding_warnings", [])
            if warnings:
                context["correlation_warnings"] = "; ".join(warnings[:3])

        sent = await get_agent_state("sentiment_agent")
        if sent:
            attention = sent.get("attention", {})
            context["attention_regime"] = attention.get("regime", "mixed")

            seasonals = sent.get("seasonal_patterns", [])
            if seasonals:
                context["seasonal_factors"] = "; ".join(
                    s.get("effect", "") for s in seasonals[:3]
                )

            cot = sent.get("cot_analysis", {}).get(proposal.instrument, {})
            if cot.get("crowded_trade"):
                context["positioning_risk"] = (
                    f"COT extreme: {cot.get('positioning_extreme', '')} "
                    f"(reversal prob {cot.get('reversal_probability', 0):.0%})"
                )

        return context
