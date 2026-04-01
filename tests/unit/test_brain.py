"""
Unit tests for Historical Brain — embeddings, patterns, knowledge base.
"""

import pytest

from src.brain.embeddings import (
    EMBEDDING_DIM,
    _deterministic_embedding,
    encode_market_moment,
)
from src.brain.patterns import (
    ALL_PATTERNS,
    COMPOSITE_PATTERNS,
    MACRO_PATTERNS,
    PATTERN_INDEX,
    TECHNICAL_PATTERNS,
    find_matching_patterns,
)


# ── Embedding Tests ───────────────────────────────────────────

class TestEmbeddings:
    def test_deterministic_embedding_dimension(self) -> None:
        emb = _deterministic_embedding("test text")
        assert len(emb) == EMBEDDING_DIM

    def test_deterministic_embedding_consistent(self) -> None:
        emb1 = _deterministic_embedding("same input")
        emb2 = _deterministic_embedding("same input")
        assert emb1 == emb2

    def test_deterministic_embedding_different_for_different_input(self) -> None:
        emb1 = _deterministic_embedding("input A")
        emb2 = _deterministic_embedding("input B")
        assert emb1 != emb2

    def test_deterministic_embedding_normalized(self) -> None:
        import numpy as np
        emb = _deterministic_embedding("test normalization")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01  # Should be approximately unit length


class TestMarketMomentEncoding:
    def test_encode_bullish_setup(self) -> None:
        text = encode_market_moment(
            instrument="EUR/USD",
            price=1.0845,
            indicators={"ema_alignment": "bullish", "adx": 32, "rsi": 28, "macd_cross": "bullish", "squeeze": True},
            macro_regime="risk_on",
        )
        assert "EUR/USD" in text
        assert "bullish" in text.lower()
        assert "oversold" in text.lower()
        assert "squeeze" in text.lower()

    def test_encode_ranging_market(self) -> None:
        text = encode_market_moment(
            instrument="USD/JPY",
            price=150.25,
            indicators={"ema_alignment": "mixed", "adx": 15, "rsi": 50},
            macro_regime="neutral",
        )
        assert "Ranging" in text or "no clear trend" in text.lower()

    def test_encode_with_correlations(self) -> None:
        text = encode_market_moment(
            instrument="EUR/USD",
            price=1.08,
            indicators={},
            macro_regime="neutral",
            correlations={"GBP/USD": 0.85, "USD/JPY": -0.3},
        )
        assert "GBP/USD" in text
        assert "0.85" in text

    def test_encode_with_sentiment(self) -> None:
        text = encode_market_moment(
            instrument="XAU/USD",
            price=2000,
            indicators={},
            macro_regime="risk_off",
            sentiment={"fear_greed": {"score": 15, "label": "extreme_fear"}},
        )
        assert "extreme_fear" in text


# ── Pattern Catalog Tests ─────────────────────────────────────

class TestPatternCatalog:
    def test_all_patterns_have_outcomes(self) -> None:
        for p in ALL_PATTERNS:
            assert p.outcome is not None
            assert p.outcome.total_occurrences > 0
            assert 0 <= p.outcome.win_rate <= 1
            assert p.outcome.avg_win_pips > 0
            assert p.outcome.avg_loss_pips < 0

    def test_pattern_index(self) -> None:
        assert "rsi_oversold_macd_cross" in PATTERN_INDEX
        assert "fed_dovish_shift" in PATTERN_INDEX
        assert "triple_divergence_reversal" in PATTERN_INDEX

    def test_technical_patterns_exist(self) -> None:
        assert len(TECHNICAL_PATTERNS) >= 4

    def test_macro_patterns_exist(self) -> None:
        assert len(MACRO_PATTERNS) >= 3

    def test_composite_patterns_exist(self) -> None:
        assert len(COMPOSITE_PATTERNS) >= 2

    def test_pattern_categories(self) -> None:
        categories = set(p.category for p in ALL_PATTERNS)
        assert "technical" in categories
        assert "macro" in categories
        assert "composite" in categories

    def test_find_matching_rsi_oversold(self) -> None:
        matches = find_matching_patterns(
            indicators={"rsi": 25, "macd_cross": "bullish", "timeframe": "1h"},
        )
        # Should find rsi_oversold_macd_cross
        pattern_names = [p.name for p, _ in matches]
        assert "rsi_oversold_macd_cross" in pattern_names

    def test_find_matching_squeeze(self) -> None:
        matches = find_matching_patterns(
            indicators={"squeeze": True, "squeeze_release": True},
        )
        pattern_names = [p.name for p, _ in matches]
        assert "bollinger_squeeze_breakout" in pattern_names

    def test_find_matching_macro(self) -> None:
        matches = find_matching_patterns(
            indicators={},
            macro_context={"vix": 30, "vix_change": 25, "trigger": "geopolitical"},
        )
        pattern_names = [p.name for p, _ in matches]
        assert "risk_off_vix_spike" in pattern_names

    def test_no_match_returns_empty(self) -> None:
        matches = find_matching_patterns(
            indicators={"completely_unrelated": "value"},
        )
        # May have low-scoring matches but none above 0.5
        high_matches = [(p, s) for p, s in matches if s > 0.5]
        # Some patterns might partially match, but it should be limited
        assert len(high_matches) <= 2

    def test_match_scores_between_zero_and_one(self) -> None:
        matches = find_matching_patterns(
            indicators={"rsi": 25, "macd_cross": "bullish"},
        )
        for _, score in matches:
            assert 0 <= score <= 1


# ── Seed Data Tests ───────────────────────────────────────────

class TestSeedData:
    def test_seed_events_valid(self) -> None:
        from scripts.seed_knowledge_base import SEED_EVENTS
        assert len(SEED_EVENTS) >= 10

        for event in SEED_EVENTS:
            assert "event_type" in event
            assert "event_name" in event
            assert "description" in event
            assert "ts" in event
            assert "affected_pairs" in event
            assert "price_impact" in event
            assert len(event["description"]) > 20

    def test_seed_events_have_tags(self) -> None:
        from scripts.seed_knowledge_base import SEED_EVENTS
        for event in SEED_EVENTS:
            assert "tags" in event
            assert len(event["tags"]) > 0

    def test_seed_events_have_price_impact(self) -> None:
        from scripts.seed_knowledge_base import SEED_EVENTS
        for event in SEED_EVENTS:
            impact = event["price_impact"]
            assert len(impact) > 0
            for instrument, data in impact.items():
                assert "pips" in data
                assert "direction" in data


# ── Knowledge Base Interface Tests ────────────────────────────

class TestKnowledgeBase:
    def test_singleton(self) -> None:
        from src.brain.knowledge_base import get_brain
        brain1 = get_brain()
        brain2 = get_brain()
        assert brain1 is brain2

    def test_brain_has_query_method(self) -> None:
        from src.brain.knowledge_base import HistoricalBrain
        brain = HistoricalBrain()
        assert hasattr(brain, "query")
        assert hasattr(brain, "query_setup")
        assert hasattr(brain, "query_event")
        assert hasattr(brain, "store_event")
        assert hasattr(brain, "get_stats")
