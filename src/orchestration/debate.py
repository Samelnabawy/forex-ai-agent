"""
Debate Engine — Orchestrates the Bull/Bear adversarial process.

Three-phase debate:
  Phase 1: Bull builds the case (all supporting evidence)
  Phase 2: Bear attacks the case (specific rebuttals + counter-evidence)
  Phase 3: Verdict synthesis (evidence weights + LLM judgment)

The debate is triggered when a trade proposal arrives from the signal queue.
The verdict goes to the Risk Manager and then Portfolio Manager.

This is NOT two independent opinions averaged together.
It's a structured adversarial process where:
  - The Bear sees the Bull's FULL case before responding
  - The Bear attacks SPECIFIC evidence, not just general concerns
  - Bull evidence is marked as "survived" or "rebutted"
  - The verdict is computed from evidence weights, not a vote
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.agents.bull_researcher import BullResearcherAgent
from src.agents.bear_researcher import BearResearcherAgent
from src.agents.evidence import (
    EvidencePackage,
    TradeProposal,
    aggregate_evidence,
    calculate_expected_value,
    compute_win_probability,
)
from src.agents.llm_client import get_llm_client
from src.agents.prompts.debate import build_verdict_prompt
from src.config.instruments import INSTRUMENTS
from src.models import MarketState

logger = logging.getLogger(__name__)


class DebateVerdict:
    """The final output of the debate process."""

    def __init__(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        verdict: str,
        conviction: float,
        position_size: str,
        bull_package: EvidencePackage,
        bear_package: EvidencePackage,
        expected_value: dict[str, Any],
        consensus_points: list[str],
        contested_points: list[dict[str, Any]],
        unresolved_risks: list[str],
        modifications: dict[str, Any],
        reasoning: str,
    ) -> None:
        self.trade_id = trade_id
        self.instrument = instrument
        self.direction = direction
        self.verdict = verdict
        self.conviction = conviction
        self.position_size = position_size
        self.bull_package = bull_package
        self.bear_package = bear_package
        self.expected_value = expected_value
        self.consensus_points = consensus_points
        self.contested_points = contested_points
        self.unresolved_risks = unresolved_risks
        self.modifications = modifications
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, Any]:
        bull_agg = aggregate_evidence(self.bull_package.evidence)
        bear_agg = aggregate_evidence(self.bear_package.evidence)

        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "verdict": self.verdict,
            "conviction": self.conviction,
            "position_size": self.position_size,
            "expected_value": self.expected_value,
            "bull_summary": {
                "conviction": self.bull_package.conviction_score,
                "evidence_count": len(self.bull_package.evidence),
                "weight": bull_agg.get("supporting_weight", 0),
                "strong_evidence": bull_agg.get("strong_evidence", 0),
                "key_points": self.bull_package.reasoning_summary,
                "invalidation": self.bull_package.invalidation_conditions,
            },
            "bear_summary": {
                "conviction": self.bear_package.conviction_score,
                "evidence_count": len(self.bear_package.evidence),
                "weight": bear_agg.get("opposing_weight", 0),
                "key_risks": self.bear_package.key_risks,
                "alternative": self.bear_package.alternative_actions,
                "key_points": self.bear_package.reasoning_summary,
            },
            "consensus_points": self.consensus_points,
            "contested_points": self.contested_points,
            "unresolved_risks": self.unresolved_risks,
            "modifications": self.modifications,
            "reasoning": self.reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class DebateEngine:
    """
    Orchestrates the Bull/Bear debate for each trade proposal.
    Produces a DebateVerdict that the Risk Manager and Portfolio Manager consume.
    """

    def __init__(self) -> None:
        self._bull = BullResearcherAgent()
        self._bear = BearResearcherAgent()
        self._llm = get_llm_client()
        self._debate_count = 0

    async def run_debate(
        self,
        proposal: TradeProposal,
        market_state: MarketState,
    ) -> DebateVerdict:
        """
        Execute the full three-phase debate.

        Phase 1: Bull builds case
        Phase 2: Bear attacks (with full visibility of Bull's case)
        Phase 3: Verdict synthesis
        """
        self._debate_count += 1
        logger.info(
            "Debate %d starting",
            self._debate_count,
            extra={"instrument": proposal.instrument, "direction": proposal.direction},
        )

        # ── Phase 1: Bull Case ────────────────────────────────
        bull_package = await self._bull.build_case(proposal, market_state)
        logger.info(
            "Bull case built",
            extra={
                "conviction": bull_package.conviction_score,
                "evidence_count": len(bull_package.evidence),
            },
        )

        # ── Phase 2: Bear Counter-Case ────────────────────────
        # Bear receives Bull's FULL package — can see all evidence
        bear_package = await self._bear.build_counter_case(
            proposal, bull_package, market_state
        )
        logger.info(
            "Bear counter-case built",
            extra={
                "conviction": bear_package.conviction_score,
                "evidence_count": len(bear_package.evidence),
                "rebuttals_applied": sum(
                    1 for e in bull_package.evidence if e.rebuttal is not None
                ),
            },
        )

        # ── Phase 3: Verdict ──────────────────────────────────
        verdict = await self._synthesize_verdict(
            proposal, bull_package, bear_package
        )

        logger.info(
            "Debate %d concluded: %s (conviction: %.1f)",
            self._debate_count,
            verdict.verdict,
            verdict.conviction,
            extra={
                "instrument": proposal.instrument,
                "direction": proposal.direction,
                "position_size": verdict.position_size,
            },
        )

        return verdict

    async def _synthesize_verdict(
        self,
        proposal: TradeProposal,
        bull_package: EvidencePackage,
        bear_package: EvidencePackage,
    ) -> DebateVerdict:
        """Phase 3: Synthesize both cases into a final verdict."""

        # Quantitative analysis
        win_prob = compute_win_probability(bull_package, bear_package)

        inst = INSTRUMENTS.get(proposal.instrument)
        pip_size = float(inst.pip_size) if inst else 0.0001

        ev = calculate_expected_value(
            win_probability=win_prob,
            entry=proposal.entry,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit_1,
            pip_size=pip_size,
        )

        # Evidence statistics
        bull_agg = aggregate_evidence(bull_package.evidence)
        bear_agg = aggregate_evidence(bear_package.evidence)

        survived = sum(1 for e in bull_package.evidence if e.survived_rebuttal)
        total = len(bull_package.evidence)

        evidence_stats = {
            "bull_weight": bull_agg.get("supporting_weight", 0),
            "bear_weight": bear_agg.get("opposing_weight", 0),
            "survived_count": survived,
            "total_count": total,
        }

        # LLM verdict synthesis
        bull_llm = getattr(bull_package, "llm_case", {}) if hasattr(bull_package, "__dict__") else {}
        bear_llm = getattr(bear_package, "llm_case", {}) if hasattr(bear_package, "__dict__") else {}

        if not bull_llm:
            bull_llm = {"reasoning_summary": bull_package.reasoning_summary}
        if not bear_llm:
            bear_llm = {"reasoning_summary": bear_package.reasoning_summary}

        ev_dict = {
            "win_probability": ev.win_probability,
            "expected_value_pips": ev.expected_value_pips,
            "risk_reward_ratio": ev.risk_reward_ratio,
            "ev_per_risk_unit": ev.ev_per_risk_unit,
        }

        system, user = build_verdict_prompt(bull_llm, bear_llm, ev_dict, evidence_stats)

        llm_verdict = await self._llm.analyze(
            system_prompt=system,
            user_prompt=user,
            agent_name="debate_verdict",
            prompt_version="v1",
        )

        # Extract verdict components
        if llm_verdict.get("_fallback"):
            # Fallback: use quantitative analysis only
            verdict_str = "EXECUTE" if ev.is_positive_ev and ev.meets_minimum_threshold else "SKIP"
            conviction = bull_package.conviction_score * 0.5  # Reduced without LLM
            position_size = "reduced (0.5%)"
            consensus = []
            contested = []
            unresolved = ["LLM unavailable — verdict based on quantitative analysis only"]
            modifications = {}
            reasoning = f"Quantitative verdict: EV={ev.expected_value_pips:.1f} pips, R:R={ev.risk_reward_ratio:.2f}"
        else:
            verdict_str = llm_verdict.get("verdict", "SKIP")
            conviction = llm_verdict.get("conviction_score", 50)
            position_size = llm_verdict.get("position_size_recommendation", "reduced (0.5%)")
            consensus = llm_verdict.get("consensus_points", [])
            contested = llm_verdict.get("contested_points", [])
            unresolved = llm_verdict.get("unresolved_risks", [])
            modifications = llm_verdict.get("modifications", {})
            reasoning = llm_verdict.get("reasoning", "")

        return DebateVerdict(
            trade_id=proposal.trade_id,
            instrument=proposal.instrument,
            direction=proposal.direction,
            verdict=verdict_str,
            conviction=conviction,
            position_size=position_size,
            bull_package=bull_package,
            bear_package=bear_package,
            expected_value=ev_dict,
            consensus_points=consensus,
            contested_points=contested,
            unresolved_risks=unresolved,
            modifications=modifications,
            reasoning=reasoning,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "debates_conducted": self._debate_count,
            "bull_stats": self._bull.stats,
            "bear_stats": self._bear.stats,
        }
