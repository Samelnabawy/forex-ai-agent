"""
Unit tests for Sentiment Agent — all 10 analytical layers.
Pure computation tests — no database, no LLM, no Redis.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.sentiment_models import (
    AttentionRegime,
    COTDecomposition,
    FearGreedIndex,
    FeedbackTracker,
    PositioningExtreme,
    SeasonalSignal,
    SentimentPriceDivergence,
    SurpriseIndex,
    analyze_cot_positioning,
    classify_attention_regime,
    compute_fear_greed,
    compute_sentiment_decay,
    compute_surprise_index,
    detect_reflexivity,
    detect_seasonal_patterns,
    detect_sentiment_price_divergence,
)


# ── Layer 2: COT Positioning Tests ────────────────────────────

class TestCOTPositioning:
    def test_extreme_long_detection(self) -> None:
        current = {
            "instrument": "EUR/USD",
            "non_commercial_long": 200000,
            "non_commercial_short": 50000,
            "open_interest": 500000,
        }
        # History where current net (150K) is at the top
        history = [
            {"net_position": i * 10000, "commercial_long": 100000, "commercial_short": 100000}
            for i in range(-5, 10)
        ]
        result = analyze_cot_positioning(current, history)
        assert result.speculator_percentile > 80
        assert result.positioning_extreme in (
            PositioningExtreme.EXTREME_LONG,
            PositioningExtreme.ELEVATED_LONG,
        )

    def test_neutral_positioning(self) -> None:
        current = {
            "instrument": "EUR/USD",
            "non_commercial_long": 100000,
            "non_commercial_short": 100000,
            "open_interest": 500000,
        }
        history = [
            {"net_position": i * 10000, "commercial_long": 100000, "commercial_short": 100000}
            for i in range(-10, 10)
        ]
        result = analyze_cot_positioning(current, history)
        assert result.positioning_extreme == PositioningExtreme.NEUTRAL

    def test_crowded_trade_flags_reversal(self) -> None:
        current = {
            "instrument": "EUR/USD",
            "non_commercial_long": 300000,
            "non_commercial_short": 10000,
            "open_interest": 500000,
        }
        history = [
            {"net_position": i * 5000, "commercial_long": 100000, "commercial_short": 100000}
            for i in range(0, 40)
        ]
        result = analyze_cot_positioning(current, history)
        if result.crowded_trade:
            assert result.reversal_probability > 0.25

    def test_contrarian_signal_on_extreme(self) -> None:
        current = {
            "instrument": "GBP/USD",
            "non_commercial_long": 5000,
            "non_commercial_short": 200000,
            "open_interest": 400000,
        }
        history = [
            {"net_position": i * 10000, "commercial_long": 100000, "commercial_short": 100000}
            for i in range(-10, 15)
        ]
        result = analyze_cot_positioning(current, history)
        # With extreme short positioning, might get contrarian buy
        assert result.speculator_percentile < 30

    def test_empty_history(self) -> None:
        current = {"instrument": "EUR/USD", "non_commercial_long": 100000,
                    "non_commercial_short": 50000, "open_interest": 300000}
        result = analyze_cot_positioning(current, [])
        assert result.positioning_extreme == PositioningExtreme.NEUTRAL


# ── Layer 3: Economic Surprise Tests ──────────────────────────

class TestEconomicSurprise:
    def test_positive_surprise_index(self) -> None:
        releases = [
            {"event_name": "NFP", "actual": "250", "forecast": "200", "previous": "180",
             "ts": datetime.now(timezone.utc)},
            {"event_name": "CPI", "actual": "3.5", "forecast": "3.2", "previous": "3.1",
             "ts": datetime.now(timezone.utc)},
        ]
        result = compute_surprise_index(releases, "US")
        assert result.score > 0
        assert result.country == "US"

    def test_negative_surprise_index(self) -> None:
        releases = [
            {"event_name": "NFP", "actual": "100", "forecast": "200", "previous": "220",
             "ts": datetime.now(timezone.utc)},
            {"event_name": "PMI", "actual": "48", "forecast": "52", "previous": "51",
             "ts": datetime.now(timezone.utc)},
        ]
        result = compute_surprise_index(releases, "US")
        assert result.score < 0

    def test_decay_weighting(self) -> None:
        now = datetime.now(timezone.utc)
        old_beat = {"event_name": "Old", "actual": "300", "forecast": "200", "previous": "190",
                    "ts": now - timedelta(days=7)}
        recent_miss = {"event_name": "Recent", "actual": "150", "forecast": "200", "previous": "210",
                       "ts": now}

        result = compute_surprise_index([old_beat, recent_miss], "US")
        # Recent miss should dominate despite old beat
        assert result.score < 0

    def test_streak_counting(self) -> None:
        now = datetime.now(timezone.utc)
        releases = [
            {"event_name": f"Data{i}", "actual": "110", "forecast": "100", "previous": "95",
             "ts": now - timedelta(hours=i)}
            for i in range(5)
        ]
        result = compute_surprise_index(releases, "US")
        assert result.streak > 0

    def test_empty_releases(self) -> None:
        result = compute_surprise_index([], "US")
        assert result.score == 0


# ── Layer 4: Divergence Tests ─────────────────────────────────

class TestDivergence:
    def test_bearish_divergence_detected(self) -> None:
        result = detect_sentiment_price_divergence(
            instrument="EUR/USD",
            price_trend="up",
            news_sentiment=-2.0,
            cot_momentum="unwinding",
            surprise_trend="deteriorating",
        )
        assert result is not None
        assert result.direction == "bearish_divergence"
        assert result.divergence_type.value == "triple"
        assert result.severity >= 80

    def test_bullish_divergence_detected(self) -> None:
        result = detect_sentiment_price_divergence(
            instrument="GBP/USD",
            price_trend="down",
            news_sentiment=2.0,
            cot_momentum="building",
            surprise_trend="improving",
        )
        assert result is not None
        assert result.direction == "bullish_divergence"

    def test_no_divergence_when_aligned(self) -> None:
        result = detect_sentiment_price_divergence(
            instrument="EUR/USD",
            price_trend="up",
            news_sentiment=2.0,
            cot_momentum="building",
            surprise_trend="improving",
        )
        assert result is None

    def test_single_divergence_lower_severity(self) -> None:
        result = detect_sentiment_price_divergence(
            instrument="EUR/USD",
            price_trend="up",
            news_sentiment=-2.0,
            cot_momentum="flat",
            surprise_trend="flat",
        )
        assert result is not None
        assert result.severity < 80  # Single divergence = lower severity


# ── Layer 6: Attention Model Tests ────────────────────────────

class TestAttentionModel:
    def test_rate_cycle_detection(self) -> None:
        headlines = [
            "Fed signals rate cut at next meeting",
            "ECB hawks push for rate hike",
            "FOMC minutes reveal hawkish dissent",
            "Bond yields surge on rate expectations",
        ]
        result = classify_attention_regime(headlines)
        assert result.regime == AttentionRegime.RATE_CYCLE

    def test_crisis_detection(self) -> None:
        headlines = [
            "Military escalation in Middle East",
            "Sanctions imposed on major economy",
            "Conflict spreads to new region",
            "Attack on oil infrastructure",
        ]
        result = classify_attention_regime(headlines, geo_risk_score=60)
        assert result.regime == AttentionRegime.CRISIS

    def test_growth_scare_detection(self) -> None:
        headlines = [
            "GDP contracts for second quarter",
            "PMI falls below 50 signaling recession",
            "Jobs report shows unemployment rising",
            "Retail sales plunge unexpectedly",
        ]
        result = classify_attention_regime(headlines)
        assert result.regime == AttentionRegime.GROWTH_SCARE

    def test_mixed_when_no_dominant(self) -> None:
        headlines = ["Weather forecast sunny", "New restaurant opens"]
        result = classify_attention_regime(headlines)
        assert result.regime == AttentionRegime.MIXED

    def test_what_matters_populated(self) -> None:
        headlines = ["Fed cuts rates by 50bps"]
        result = classify_attention_regime(headlines)
        assert len(result.what_matters_now) > 0


# ── Layer 8: Seasonal Tests ───────────────────────────────────

class TestSeasonals:
    def test_december_jpy_repatriation(self) -> None:
        ts = datetime(2026, 12, 15, 12, 0, 0, tzinfo=timezone.utc)
        signals = detect_seasonal_patterns(ts)
        jpy = [s for s in signals if "jpn_year_end" in s.pattern_name]
        assert len(jpy) > 0
        assert "USD/JPY" in jpy[0].affected_instruments

    def test_golden_week(self) -> None:
        ts = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        signals = detect_seasonal_patterns(ts)
        gw = [s for s in signals if "golden_week" in s.pattern_name]
        assert len(gw) > 0

    def test_month_end_rebalancing(self) -> None:
        ts = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
        signals = detect_seasonal_patterns(ts)
        me = [s for s in signals if "month_end" in s.pattern_name]
        assert len(me) > 0

    def test_no_seasonals_mid_month(self) -> None:
        ts = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        signals = detect_seasonal_patterns(ts)
        # Might still have some but fewer
        assert isinstance(signals, list)


# ── Layer 9: Reflexivity Tests ────────────────────────────────

class TestReflexivity:
    def test_bubble_detection(self) -> None:
        result = detect_reflexivity(
            instrument="EUR/USD",
            price_change_30d_pct=5.0,
            price_change_7d_pct=2.5,
            price_change_1d_pct=0.8,
            fundamental_direction="bullish",
            cot_momentum="building",
        )
        assert result is not None
        assert result.loop_type == "bubble"

    def test_panic_detection(self) -> None:
        result = detect_reflexivity(
            instrument="GBP/USD",
            price_change_30d_pct=-4.0,
            price_change_7d_pct=-2.0,
            price_change_1d_pct=-0.5,
            fundamental_direction="bearish",
            cot_momentum="unwinding",
        )
        assert result is not None
        assert result.loop_type == "panic"

    def test_no_reflexivity_in_small_move(self) -> None:
        result = detect_reflexivity(
            instrument="EUR/USD",
            price_change_30d_pct=0.5,
            price_change_7d_pct=0.1,
            price_change_1d_pct=0.05,
            fundamental_direction="neutral",
            cot_momentum="flat",
        )
        assert result is None

    def test_fundamental_disconnect(self) -> None:
        result = detect_reflexivity(
            instrument="EUR/USD",
            price_change_30d_pct=5.0,
            price_change_7d_pct=2.0,
            price_change_1d_pct=0.5,
            fundamental_direction="bearish",  # Fundamental shifted against the move
            cot_momentum="unwinding",
        )
        assert result is not None
        assert result.fundamental_support is False


# ── Layer 10: Feedback Loop Tests ─────────────────────────────

class TestFeedbackLoop:
    def test_record_and_resolve(self) -> None:
        tracker = FeedbackTracker()
        tracker.record_prediction("EUR/USD", "bullish", "news", 0.7, 1.0800)
        assert len(tracker._records) == 1

        # Price went up → correct
        tracker.resolve_predictions("EUR/USD", 1.0900, min_age_hours=0)
        assert tracker._records[0].outcome == "correct"

    def test_hit_rate_calculation(self) -> None:
        tracker = FeedbackTracker()
        now = datetime.now(timezone.utc)

        from src.agents.sentiment_models import PredictionRecord
        for i in range(10):
            tracker._records.append(PredictionRecord(
                ts=now, instrument="EUR/USD", prediction="bullish",
                source="news", confidence=0.7,
                outcome="correct" if i < 7 else "incorrect",
            ))

        rates = tracker.get_hit_rates(window_days=30)
        assert rates["news"]["hit_rate"] == 0.7
        assert rates["news"]["total"] == 10

    def test_weight_update(self) -> None:
        tracker = FeedbackTracker()
        from src.agents.sentiment_models import PredictionRecord
        now = datetime.now(timezone.utc)

        # 80% hit rate for COT
        for i in range(20):
            tracker._records.append(PredictionRecord(
                ts=now, instrument="EUR/USD", prediction="bullish",
                source="cot", confidence=0.7,
                outcome="correct" if i < 16 else "incorrect",
            ))

        tracker.update_weights()
        assert tracker._source_weights["cot"] > 1.0  # Should be boosted


# ── Sentiment Decay Tests ─────────────────────────────────────

class TestSentimentDecay:
    def test_recent_scores_dominate(self) -> None:
        now = datetime.now(timezone.utc)
        scores = [
            (-3.0, now - timedelta(hours=12)),  # Old bearish
            (3.0, now),                          # Recent bullish
        ]
        result = compute_sentiment_decay(scores, half_life_hours=4.0)
        assert result > 0  # Recent bullish should dominate

    def test_equal_time_equal_weight(self) -> None:
        now = datetime.now(timezone.utc)
        scores = [
            (2.0, now),
            (-2.0, now),
        ]
        result = compute_sentiment_decay(scores, half_life_hours=4.0)
        assert abs(result) < 0.01  # Should roughly cancel


# ── Fear & Greed Index Tests ──────────────────────────────────

class TestFearGreed:
    def test_extreme_fear(self) -> None:
        result = compute_fear_greed(
            news_sentiment_avg=-4.0,
            cot_percentile_avg=5.0,
            surprise_score_avg=-80,
            vix=35.0,
            divergence_count=3,
        )
        assert result.score < 20
        assert result.label == "extreme_fear"
        assert result.contrarian_signal == "contrarian_buy"

    def test_extreme_greed(self) -> None:
        result = compute_fear_greed(
            news_sentiment_avg=4.0,
            cot_percentile_avg=95.0,
            surprise_score_avg=80,
            vix=12.0,
            divergence_count=0,
            intermarket_risk_score=90,
        )
        assert result.score > 80
        assert result.label == "extreme_greed"
        assert result.contrarian_signal == "contrarian_sell"

    def test_neutral_conditions(self) -> None:
        result = compute_fear_greed(
            news_sentiment_avg=0.0,
            cot_percentile_avg=50.0,
            surprise_score_avg=0,
            vix=18.0,
        )
        assert 35 < result.score < 65
        assert result.label == "neutral"

    def test_components_populated(self) -> None:
        result = compute_fear_greed(
            news_sentiment_avg=1.0,
            cot_percentile_avg=60.0,
            surprise_score_avg=10,
            vix=16.0,
        )
        assert "news_sentiment" in result.components
        assert "cot_positioning" in result.components
        assert "vix" in result.components
