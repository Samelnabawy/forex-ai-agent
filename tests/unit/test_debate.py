"""
Unit tests for Bull/Bear debate system.
Tests evidence scoring, expected value, debate flow, and verdict logic.
"""

from datetime import datetime, timezone

import pytest

from src.agents.evidence import (
    Evidence,
    EvidenceDirection,
    EvidencePackage,
    EvidenceSource,
    EvidenceStrength,
    ExpectedValueCalc,
    HistoricalPrecedent,
    TradeProposal,
    aggregate_evidence,
    calculate_expected_value,
    compute_win_probability,
    extract_technical_evidence,
)


# ── Evidence Scoring Tests ────────────────────────────────────

class TestEvidenceWeight:
    def test_weight_computation(self) -> None:
        e = Evidence(
            id="test", source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim="Test claim", data={},
            reliability_score=0.7, recency_hours=0.1, relevance_score=0.9,
        )
        weight = e.compute_weight()
        assert weight > 0
        assert weight <= 1.0

    def test_stale_evidence_lower_weight(self) -> None:
        fresh = Evidence(
            id="fresh", source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim="Fresh", data={},
            reliability_score=0.7, recency_hours=0.1, relevance_score=0.9,
        )
        stale = Evidence(
            id="stale", source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim="Stale", data={},
            reliability_score=0.7, recency_hours=24.0, relevance_score=0.9,
        )
        assert fresh.compute_weight() > stale.compute_weight()

    def test_irrelevant_evidence_lower_weight(self) -> None:
        relevant = Evidence(
            id="rel", source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim="Relevant", data={},
            reliability_score=0.7, recency_hours=0.1, relevance_score=0.95,
        )
        irrelevant = Evidence(
            id="irrel", source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim="Irrelevant", data={},
            reliability_score=0.7, recency_hours=0.1, relevance_score=0.2,
        )
        assert relevant.compute_weight() > irrelevant.compute_weight()


class TestEvidenceAggregation:
    def test_supporting_vs_opposing(self) -> None:
        evidence = [
            Evidence(
                id="s1", source=EvidenceSource.TECHNICAL,
                direction=EvidenceDirection.SUPPORTING,
                strength=EvidenceStrength.STRONG,
                claim="Support 1", data={},
                reliability_score=0.7, recency_hours=0.1, relevance_score=0.9,
            ),
            Evidence(
                id="s2", source=EvidenceSource.MACRO,
                direction=EvidenceDirection.SUPPORTING,
                strength=EvidenceStrength.MODERATE,
                claim="Support 2", data={},
                reliability_score=0.6, recency_hours=1.0, relevance_score=0.8,
            ),
            Evidence(
                id="o1", source=EvidenceSource.SENTIMENT,
                direction=EvidenceDirection.OPPOSING,
                strength=EvidenceStrength.WEAK,
                claim="Oppose 1", data={},
                reliability_score=0.4, recency_hours=2.0, relevance_score=0.5,
            ),
        ]
        agg = aggregate_evidence(evidence)
        assert agg["supporting_weight"] > agg["opposing_weight"]
        assert agg["conviction"] > 0

    def test_rebutted_evidence_loses_weight(self) -> None:
        e = Evidence(
            id="rebutted", source=EvidenceSource.TECHNICAL,
            direction=EvidenceDirection.SUPPORTING,
            strength=EvidenceStrength.STRONG,
            claim="Was strong", data={},
            reliability_score=0.8, recency_hours=0.1, relevance_score=0.9,
        )
        # Before rebuttal
        e.compute_weight()
        original_agg = aggregate_evidence([e])

        # After devastating rebuttal
        e.survived_rebuttal = False
        rebutted_agg = aggregate_evidence([e])

        assert rebutted_agg["supporting_weight"] < original_agg["supporting_weight"]

    def test_empty_evidence(self) -> None:
        agg = aggregate_evidence([])
        assert agg["conviction"] == 0
        assert agg["evidence_count"] == 0


# ── Expected Value Tests ──────────────────────────────────────

class TestExpectedValue:
    def test_positive_ev_trade(self) -> None:
        ev = calculate_expected_value(
            win_probability=0.60,
            entry=1.0845,
            stop_loss=1.0810,
            take_profit=1.0910,
            pip_size=0.0001,
        )
        assert ev.is_positive_ev is True
        assert ev.expected_value_pips > 0
        assert ev.risk_reward_ratio > 1.5

    def test_negative_ev_trade(self) -> None:
        ev = calculate_expected_value(
            win_probability=0.30,
            entry=1.0845,
            stop_loss=1.0810,
            take_profit=1.0870,  # Poor R:R
            pip_size=0.0001,
        )
        assert ev.is_positive_ev is False

    def test_high_rr_compensates_low_winrate(self) -> None:
        ev = calculate_expected_value(
            win_probability=0.40,
            entry=1.0845,
            stop_loss=1.0830,  # 15 pip risk
            take_profit=1.0920,  # 75 pip reward (5:1 R:R)
            pip_size=0.0001,
        )
        assert ev.is_positive_ev is True
        assert ev.risk_reward_ratio >= 4.0

    def test_zero_risk_returns_invalid(self) -> None:
        ev = calculate_expected_value(
            win_probability=0.5,
            entry=1.0845,
            stop_loss=1.0845,  # Zero risk
            take_profit=1.0900,
            pip_size=0.0001,
        )
        assert "Invalid" in ev.reasoning


# ── Win Probability Tests ─────────────────────────────────────

class TestWinProbability:
    def test_strong_bull_higher_probability(self) -> None:
        bull = EvidencePackage(
            side="bull", trade_id="T1", instrument="EUR/USD", direction="LONG",
            evidence=[
                Evidence(id="s1", source=EvidenceSource.TECHNICAL,
                         direction=EvidenceDirection.SUPPORTING,
                         strength=EvidenceStrength.STRONG, claim="", data={},
                         reliability_score=0.8, recency_hours=0.1, relevance_score=0.9),
                Evidence(id="s2", source=EvidenceSource.MACRO,
                         direction=EvidenceDirection.SUPPORTING,
                         strength=EvidenceStrength.STRONG, claim="", data={},
                         reliability_score=0.7, recency_hours=1.0, relevance_score=0.85),
            ],
        )
        bear = EvidencePackage(
            side="bear", trade_id="T1", instrument="EUR/USD", direction="SHORT",
            evidence=[
                Evidence(id="o1", source=EvidenceSource.SENTIMENT,
                         direction=EvidenceDirection.OPPOSING,
                         strength=EvidenceStrength.WEAK, claim="", data={},
                         reliability_score=0.3, recency_hours=2.0, relevance_score=0.5),
            ],
        )

        prob = compute_win_probability(bull, bear)
        assert prob > 0.55  # Bull is stronger

    def test_probability_capped(self) -> None:
        """Win probability should never exceed 85% or go below 15%."""
        bull = EvidencePackage(
            side="bull", trade_id="T1", instrument="EUR/USD", direction="LONG",
            evidence=[
                Evidence(id=f"s{i}", source=EvidenceSource.TECHNICAL,
                         direction=EvidenceDirection.SUPPORTING,
                         strength=EvidenceStrength.STRONG, claim="", data={},
                         reliability_score=0.9, recency_hours=0.1, relevance_score=0.95)
                for i in range(10)
            ],
        )
        bear = EvidencePackage(
            side="bear", trade_id="T1", instrument="EUR/USD", direction="SHORT",
            evidence=[],
        )

        prob = compute_win_probability(bull, bear)
        assert prob <= 0.85

    def test_historical_precedent_influence(self) -> None:
        """Historical win rate should influence probability."""
        bull = EvidencePackage(
            side="bull", trade_id="T1", instrument="EUR/USD", direction="LONG",
            evidence=[],
            precedents=[
                HistoricalPrecedent(
                    event_description="Similar setup", date="2025-01-01",
                    similarity_score=0.8, outcome="win", pnl_pips=45,
                    holding_period_hours=4,
                )
                for _ in range(8)  # 8 out of 10 won
            ] + [
                HistoricalPrecedent(
                    event_description="Similar setup", date="2025-02-01",
                    similarity_score=0.8, outcome="loss", pnl_pips=-25,
                    holding_period_hours=2,
                )
                for _ in range(2)
            ],
        )
        bear = EvidencePackage(
            side="bear", trade_id="T1", instrument="EUR/USD", direction="SHORT",
            evidence=[],
        )

        prob = compute_win_probability(bull, bear)
        assert prob > 0.6  # 80% historical win rate should push up


# ── Evidence Extraction Tests ─────────────────────────────────

class TestEvidenceExtraction:
    def test_technical_rsi_oversold(self) -> None:
        signal = {
            "instrument": "EUR/USD",
            "indicators": {"rsi": 28.5, "macd_cross": "bullish", "ema_alignment": "bullish"},
            "patterns": ["hammer"],
        }
        evidence = extract_technical_evidence(signal, "LONG")
        rsi_evidence = [e for e in evidence if "rsi" in e.id.lower()]
        assert len(rsi_evidence) > 0
        assert rsi_evidence[0].direction == EvidenceDirection.SUPPORTING

    def test_technical_no_evidence_when_neutral(self) -> None:
        signal = {
            "instrument": "EUR/USD",
            "indicators": {"rsi": 50, "macd_cross": "none", "ema_alignment": "mixed"},
            "patterns": [],
        }
        evidence = extract_technical_evidence(signal, "LONG")
        # Should have fewer or no supporting evidence with neutral readings
        supporting = [e for e in evidence if e.direction == EvidenceDirection.SUPPORTING]
        assert len(supporting) <= 1  # Maybe ADX if present


# ── Trade Proposal Tests ──────────────────────────────────────

class TestTradeProposal:
    def test_create_proposal(self) -> None:
        proposal = TradeProposal(
            trade_id="T-20260401-001",
            instrument="EUR/USD",
            direction="LONG",
            entry=1.0845,
            stop_loss=1.0810,
            take_profit_1=1.0910,
            take_profit_2=1.0950,
        )
        assert proposal.instrument == "EUR/USD"
        assert proposal.direction == "LONG"
        assert proposal.take_profit_1 > proposal.entry
        assert proposal.stop_loss < proposal.entry
