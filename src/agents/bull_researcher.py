"""
Bull Researcher — Agent 05

"The Optimistic Advocate" — finds every reason why this trade works.
Not blindly bullish — acknowledges risks but argues they're manageable.

Process:
  1. Receives trade proposal from Technical Agent signal
  2. Gathers ALL supporting evidence from Agents 1-4
  3. Queries Historical Brain for precedents that WORKED
  4. Scores each evidence piece (reliability, recency, relevance)
  5. LLM synthesizes the strongest possible case
  6. Calculates expected value
  7. Identifies invalidation conditions

Does NOT decide to trade. Presents the case to the debate engine.
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
    EvidenceStrength,
    HistoricalPrecedent,
    TradeProposal,
    aggregate_evidence,
    calculate_expected_value,
    extract_correlation_evidence,
    extract_macro_evidence,
    extract_sentiment_evidence,
    extract_technical_evidence,
)
from src.agents.llm_client import get_llm_client
from src.agents.prompts.debate import build_bull_prompt
from src.config.instruments import INSTRUMENTS
from src.data.storage.cache import get_agent_state
from src.models import MarketState

logger = logging.getLogger(__name__)


class BullResearcherAgent(BaseAgent):
    """
    Agent 05: Bull Researcher

    Builds the strongest possible case FOR a trade.
    Evidence-driven, not opinion-driven.
    """

    name: ClassVar[str] = "bull_researcher"
    description: ClassVar[str] = "Builds strongest case FOR a trade using historical evidence"
    execution_frequency: ClassVar[str] = "on_demand"

    def __init__(self) -> None:
        super().__init__()
        self._llm = get_llm_client()

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """Not used directly — Bull is triggered by debate engine."""
        return {"status": "awaiting_trade_proposal"}

    async def build_case(
        self,
        proposal: TradeProposal,
        market_state: MarketState,
    ) -> EvidencePackage:
        """
        Build the bull case for a specific trade proposal.
        Called by the debate engine.
        """
        self.logger.info(
            "Building bull case",
            extra={"instrument": proposal.instrument, "direction": proposal.direction},
        )

        # 1. Gather all supporting evidence from agents
        evidence = await self._gather_evidence(proposal)

        # 2. Query Historical Brain for winning precedents
        precedents = await self._query_precedents(proposal, "win")

        # 3. Compute evidence weights
        for e in evidence:
            e.compute_weight()

        # 4. Calculate expected value
        inst = INSTRUMENTS.get(proposal.instrument)
        pip_size = float(inst.pip_size) if inst else 0.0001
        ev = calculate_expected_value(
            win_probability=0.55,  # Will be refined by debate verdict
            entry=proposal.entry,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit_1,
            pip_size=pip_size,
        )

        # 5. Build context for LLM
        context = await self._build_context(market_state)

        # 6. LLM synthesis
        supporting = [e for e in evidence if e.direction == EvidenceDirection.SUPPORTING]
        evidence_dicts = [
            {"id": e.id, "claim": e.claim, "strength": e.strength.value,
             "reliability_score": e.reliability_score, "source": e.source.value}
            for e in supporting
        ]
        precedent_dicts = [
            {"date": p.date, "event_description": p.event_description,
             "similarity_score": p.similarity_score, "outcome": p.outcome,
             "pnl_pips": p.pnl_pips}
            for p in precedents
        ]
        proposal_dict = {
            "instrument": proposal.instrument,
            "direction": proposal.direction,
            "entry": proposal.entry,
            "stop_loss": proposal.stop_loss,
            "take_profit_1": proposal.take_profit_1,
            "take_profit_2": proposal.take_profit_2,
        }

        system, user = build_bull_prompt(proposal_dict, evidence_dicts, precedent_dicts, context)
        llm_case = await self._llm.analyze(
            system_prompt=system,
            user_prompt=user,
            agent_name=self.name,
            prompt_version="v1",
        )

        # 7. Build evidence package
        agg = aggregate_evidence(evidence)
        conviction = llm_case.get("confidence", 0.5) * 100 if not llm_case.get("_fallback") else agg.get("conviction", 0)

        package = EvidencePackage(
            side="bull",
            trade_id=proposal.trade_id,
            instrument=proposal.instrument,
            direction=proposal.direction,
            evidence=evidence,
            precedents=precedents,
            expected_value=ev,
            total_weight=agg.get("supporting_weight", 0),
            conviction_score=round(conviction, 1),
            invalidation_conditions=llm_case.get("invalidation_conditions", []),
            key_risks=[r.get("risk", "") for r in llm_case.get("risks_acknowledged", [])],
            reasoning_summary=llm_case.get("reasoning_summary", ""),
        )

        # Store full LLM output for the debate engine
        package.__dict__["llm_case"] = llm_case

        return package

    async def _gather_evidence(self, proposal: TradeProposal) -> list[Evidence]:
        """Gather evidence from all 4 analysis agents."""
        evidence: list[Evidence] = []

        # Technical Agent evidence
        tech_state = await get_agent_state("technical_analyst")
        if tech_state:
            tech_signal = None
            for sig in tech_state.get("signals", []):
                if sig.get("instrument") == proposal.instrument:
                    tech_signal = sig
                    break
            if tech_signal:
                evidence.extend(extract_technical_evidence(tech_signal, proposal.direction))

        # Macro Agent evidence
        macro_state = await get_agent_state("macro_analyst")
        if macro_state:
            evidence.extend(extract_macro_evidence(macro_state, proposal.instrument, proposal.direction))

        # Correlation Agent evidence
        corr_state = await get_agent_state("correlation_agent")
        if corr_state:
            evidence.extend(extract_correlation_evidence(corr_state, proposal.instrument, proposal.direction))

        # Sentiment Agent evidence
        sent_state = await get_agent_state("sentiment_agent")
        if sent_state:
            evidence.extend(extract_sentiment_evidence(sent_state, proposal.instrument, proposal.direction))

        return evidence

    async def _query_precedents(
        self, proposal: TradeProposal, outcome_filter: str
    ) -> list[HistoricalPrecedent]:
        """Query Historical Brain for similar setups. Stub until Brain is built."""
        # TODO: Connect to brain/query_engine.py
        return []

    async def _build_context(self, market_state: MarketState) -> dict[str, Any]:
        """Build context dict for the LLM prompt."""
        macro = await get_agent_state("macro_analyst")
        sent = await get_agent_state("sentiment_agent")
        from src.config.sessions import get_session_context
        session = get_session_context()

        return {
            "macro_regime": macro.get("macro_regime", "neutral") if macro else "neutral",
            "vix": "N/A",
            "session": session.active_session.value,
            "day_type": session.day_type.value,
            "fear_greed": sent.get("fear_greed_index", {}).get("score", "N/A") if sent else "N/A",
        }
