"""
Unit tests for Portfolio Manager — triage, decision matrix,
signal management, performance tracking.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.portfolio_manager import (
    AgentVoteResult,
    PerformanceTracker,
    SignalManager,
    TriageLevel,
    compute_decision_matrix,
    triage_signal,
)


# ── Signal Triage Tests ───────────────────────────────────────

class TestTriage:
    def test_noise(self) -> None:
        assert triage_signal({"confluence_score": 30}) == TriageLevel.NOISE

    def test_weak(self) -> None:
        assert triage_signal({"confluence_score": 55}) == TriageLevel.WEAK

    def test_standard(self) -> None:
        assert triage_signal({"confluence_score": 68}) == TriageLevel.STANDARD

    def test_strong(self) -> None:
        assert triage_signal({"confluence_score": 82}) == TriageLevel.STRONG

    def test_exceptional(self) -> None:
        assert triage_signal({"confluence_score": 95}) == TriageLevel.EXCEPTIONAL

    def test_confidence_based(self) -> None:
        """Should work with confidence (0-1) as well as confluence (0-100)."""
        assert triage_signal({"confidence": 0.75}) == TriageLevel.STRONG


# ── Decision Matrix Tests ─────────────────────────────────────

class TestDecisionMatrix:
    def test_all_agree_full_position(self) -> None:
        votes = [
            AgentVoteResult("technical", "BUY", 0.8),
            AgentVoteResult("macro", "BUY", 0.75),
            AgentVoteResult("correlation", "BUY", 0.7),
            AgentVoteResult("sentiment", "BUY", 0.65),
        ]
        result = compute_decision_matrix(
            votes, debate_conviction=75, expected_value_pips=15,
            ev_per_risk=0.5, correlation_confirmed=True,
            sentiment_extreme=False, fear_greed_score=55,
        )
        assert "EXECUTE" in result["action"]
        assert result["base_size_pct"] > 0.5

    def test_split_vote_skip(self) -> None:
        votes = [
            AgentVoteResult("technical", "BUY", 0.7),
            AgentVoteResult("macro", "BUY", 0.6),
            AgentVoteResult("correlation", "SELL", 0.7),
            AgentVoteResult("sentiment", "SELL", 0.6),
        ]
        result = compute_decision_matrix(
            votes, debate_conviction=50, expected_value_pips=2,
            ev_per_risk=0.1, correlation_confirmed=False,
            sentiment_extreme=False, fear_greed_score=50,
        )
        assert result["action"] == "SKIP"

    def test_three_agree_one_neutral(self) -> None:
        votes = [
            AgentVoteResult("technical", "BUY", 0.8),
            AgentVoteResult("macro", "BUY", 0.7),
            AgentVoteResult("correlation", "BUY", 0.65),
            AgentVoteResult("sentiment", "NEUTRAL", 0.5),
        ]
        result = compute_decision_matrix(
            votes, debate_conviction=70, expected_value_pips=10,
            ev_per_risk=0.4, correlation_confirmed=True,
            sentiment_extreme=False, fear_greed_score=55,
        )
        assert "EXECUTE" in result["action"]

    def test_sentiment_extreme_reduces_size(self) -> None:
        votes = [
            AgentVoteResult("technical", "BUY", 0.8),
            AgentVoteResult("macro", "BUY", 0.75),
            AgentVoteResult("correlation", "BUY", 0.7),
            AgentVoteResult("sentiment", "BUY", 0.6),
        ]

        normal = compute_decision_matrix(
            votes, debate_conviction=75, expected_value_pips=15,
            ev_per_risk=0.5, correlation_confirmed=True,
            sentiment_extreme=False, fear_greed_score=55,
        )

        extreme = compute_decision_matrix(
            votes, debate_conviction=75, expected_value_pips=15,
            ev_per_risk=0.5, correlation_confirmed=True,
            sentiment_extreme=True, fear_greed_score=90,
        )

        assert extreme["base_size_pct"] < normal["base_size_pct"]

    def test_high_ev_boosts_size(self) -> None:
        votes = [
            AgentVoteResult("technical", "BUY", 0.7),
            AgentVoteResult("macro", "BUY", 0.65),
            AgentVoteResult("correlation", "BUY", 0.6),
            AgentVoteResult("sentiment", "NEUTRAL", 0.5),
        ]

        low_ev = compute_decision_matrix(
            votes, debate_conviction=70, expected_value_pips=5,
            ev_per_risk=0.1, correlation_confirmed=False,
            sentiment_extreme=False, fear_greed_score=50,
        )

        high_ev = compute_decision_matrix(
            votes, debate_conviction=70, expected_value_pips=25,
            ev_per_risk=0.8, correlation_confirmed=False,
            sentiment_extreme=False, fear_greed_score=50,
        )

        assert high_ev["base_size_pct"] >= low_ev["base_size_pct"]

    def test_votes_tracked(self) -> None:
        votes = [
            AgentVoteResult("technical", "BUY", 0.8),
            AgentVoteResult("macro", "SELL", 0.6),
        ]
        result = compute_decision_matrix(
            votes, 60, 10, 0.3, False, False, 50,
        )
        assert "technical" in result["votes"]
        assert "macro" in result["votes"]
        assert result["buy_votes"] == 1
        assert result["sell_votes"] == 1


# ── Signal Manager Tests ──────────────────────────────────────

class TestSignalManager:
    def test_add_and_get(self) -> None:
        mgr = SignalManager()
        signal = {"instrument": "EUR/USD", "direction": "LONG",
                  "confluence_score": 75, "timestamp": datetime.now(timezone.utc).isoformat()}
        assert mgr.add_signal(signal) is True
        assert mgr.queue_depth == 1

        retrieved = mgr.get_next()
        assert retrieved is not None
        assert retrieved["instrument"] == "EUR/USD"
        assert mgr.queue_depth == 0

    def test_duplicate_rejected(self) -> None:
        mgr = SignalManager()
        signal = {"instrument": "EUR/USD", "direction": "LONG",
                  "confluence_score": 75, "timestamp": datetime.now(timezone.utc).isoformat()}
        mgr.add_signal(signal)
        assert mgr.add_signal(signal) is False  # Duplicate
        assert mgr.queue_depth == 1

    def test_stronger_signal_replaces(self) -> None:
        mgr = SignalManager()
        weak = {"instrument": "EUR/USD", "direction": "LONG",
                "confluence_score": 65, "timestamp": datetime.now(timezone.utc).isoformat()}
        strong = {"instrument": "EUR/USD", "direction": "LONG",
                  "confluence_score": 85, "timestamp": datetime.now(timezone.utc).isoformat()}

        mgr.add_signal(weak)
        mgr.add_signal(strong)
        assert mgr.queue_depth == 1

        retrieved = mgr.get_next()
        assert retrieved["confluence_score"] == 85

    def test_priority_ordering(self) -> None:
        mgr = SignalManager()
        mgr.add_signal({"instrument": "EUR/USD", "direction": "LONG",
                        "confluence_score": 65, "timestamp": datetime.now(timezone.utc).isoformat()})
        mgr.add_signal({"instrument": "GBP/USD", "direction": "LONG",
                        "confluence_score": 85, "timestamp": datetime.now(timezone.utc).isoformat()})
        mgr.add_signal({"instrument": "USD/JPY", "direction": "SHORT",
                        "confluence_score": 75, "timestamp": datetime.now(timezone.utc).isoformat()})

        first = mgr.get_next()
        assert first["instrument"] == "GBP/USD"  # Highest confluence first

    def test_queue_cap(self) -> None:
        mgr = SignalManager(max_pending=3)
        for i in range(5):
            mgr.add_signal({
                "instrument": f"PAIR{i}", "direction": "LONG",
                "confluence_score": 60 + i * 5,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        assert mgr.queue_depth == 3  # Capped

    def test_empty_queue(self) -> None:
        mgr = SignalManager()
        assert mgr.get_next() is None


# ── Performance Tracker Tests ─────────────────────────────────

class TestPerformanceTracker:
    def test_record_and_stats(self) -> None:
        tracker = PerformanceTracker()
        for i in range(10):
            tracker.record_closed_trade({
                "outcome": "win" if i < 6 else "loss",
                "pnl_pips": 30 if i < 6 else -20,
                "direction": "LONG",
                "agent_votes": {"technical": "BUY", "macro": "BUY"},
            })

        stats = tracker.get_stats(10)
        assert stats["trades"] == 10
        assert stats["wins"] == 6
        assert stats["win_rate"] == 0.6

    def test_strategy_adjustment_conservative(self) -> None:
        tracker = PerformanceTracker()
        for i in range(20):
            tracker.record_closed_trade({
                "outcome": "loss" if i < 15 else "win",
                "pnl_pips": -20 if i < 15 else 30,
                "direction": "LONG",
                "agent_votes": {},
            })

        adj = tracker.get_strategy_adjustments()
        assert adj["mode"] == "conservative"

    def test_strategy_adjustment_aggressive(self) -> None:
        tracker = PerformanceTracker()
        for i in range(20):
            tracker.record_closed_trade({
                "outcome": "win" if i < 15 else "loss",
                "pnl_pips": 30 if i < 15 else -20,
                "direction": "LONG",
                "agent_votes": {},
            })

        adj = tracker.get_strategy_adjustments()
        assert adj["mode"] == "aggressive"

    def test_not_enough_data(self) -> None:
        tracker = PerformanceTracker()
        tracker.record_closed_trade({"outcome": "win", "pnl_pips": 30, "direction": "LONG", "agent_votes": {}})
        adj = tracker.get_strategy_adjustments()
        assert adj["mode"] == "normal"  # Not enough trades to judge

    def test_agent_accuracy(self) -> None:
        tracker = PerformanceTracker()
        # Technical says BUY, trade goes LONG and wins
        tracker.record_closed_trade({
            "outcome": "win", "pnl_pips": 30, "direction": "LONG",
            "agent_votes": {"technical": "BUY"},
        })
        stats = tracker.get_stats()
        assert stats["agent_accuracy"]["technical"] > 0


# ── Telegram Format Tests ────────────────────────────────────

class TestTelegramFormat:
    def test_signal_format(self) -> None:
        from src.agents.evidence import TradeProposal
        from src.execution.telegram_bot import format_trade_signal

        proposal = TradeProposal(
            trade_id="T-20260401-001",
            instrument="EUR/USD",
            direction="LONG",
            entry=1.08450,
            stop_loss=1.08100,
            take_profit_1=1.09000,
            take_profit_2=1.09400,
        )

        decision = {
            "final_position_size": 0.5,
            "matrix": {"agreement_ratio": 0.8},
            "debate": {
                "expected_value": {"expected_value_pips": 12.4},
                "reasoning": "Test reasoning",
                "bull_summary": {"key_points": "", "invalidation": []},
                "bear_summary": {"key_risks": ["Test risk"]},
            },
        }

        msg = format_trade_signal(decision, proposal)
        assert "LONG EUR/USD" in msg
        assert "1.08450" in msg
        assert "Risk:" in msg

    def test_alert_format(self) -> None:
        from src.execution.telegram_bot import format_alert
        msg = format_alert("circuit_breaker", "3 consecutive losses", "high")
        assert "🔴" in msg
        assert "⚠️" in msg
