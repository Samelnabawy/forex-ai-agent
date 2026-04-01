"""
Unit tests for Macro Analyst Agent and supporting components.
Tests prompt construction, response validation, session detection, and data feeds.
No actual LLM calls — mocked for deterministic testing.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.config.sessions import (
    DayType,
    TradingSession,
    get_active_session,
    get_session_context,
    is_central_bank_day,
    is_london_fix_window,
    is_month_end,
    is_nfp_day,
    is_quarter_end,
    is_rollover_window,
)
from src.agents.prompts.macro import (
    CURRENT_VERSION,
    SYSTEM_PROMPTS,
    build_event_impact_prompt,
    build_macro_analysis_prompt,
)
from src.agents.llm_client import LLMClient, UsageStats


# ── Session Detection Tests ───────────────────────────────────

class TestSessionDetection:
    def test_asian_session(self) -> None:
        ts = datetime(2026, 4, 1, 3, 0, 0, tzinfo=timezone.utc)
        assert get_active_session(ts) == TradingSession.ASIAN

    def test_london_session(self) -> None:
        ts = datetime(2026, 4, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert get_active_session(ts) == TradingSession.LONDON

    def test_ny_session(self) -> None:
        ts = datetime(2026, 4, 1, 18, 0, 0, tzinfo=timezone.utc)
        assert get_active_session(ts) == TradingSession.NEW_YORK

    def test_overlap_session(self) -> None:
        ts = datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc)
        assert get_active_session(ts) == TradingSession.OVERLAP

    def test_off_hours(self) -> None:
        ts = datetime(2026, 4, 1, 22, 0, 0, tzinfo=timezone.utc)
        assert get_active_session(ts) == TradingSession.OFF_HOURS


class TestNFPDay:
    def test_first_friday(self) -> None:
        # April 3, 2026 is a Friday and day 3 (first Friday)
        ts = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
        assert is_nfp_day(ts) is True

    def test_second_friday(self) -> None:
        # April 10, 2026 is a Friday but day 10 (second Friday)
        ts = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
        assert is_nfp_day(ts) is False

    def test_not_friday(self) -> None:
        ts = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)  # Wednesday
        assert is_nfp_day(ts) is False


class TestCentralBankDay:
    def test_fed_meeting_day(self) -> None:
        ts = datetime(2026, 1, 28, 12, 0, 0, tzinfo=timezone.utc)
        is_cb, meetings = is_central_bank_day(ts)
        assert is_cb is True
        assert any(m["bank"] == "Fed" for m in meetings)

    def test_normal_day(self) -> None:
        ts = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        is_cb, meetings = is_central_bank_day(ts)
        assert is_cb is False
        assert len(meetings) == 0


class TestMonthEnd:
    def test_last_day(self) -> None:
        ts = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
        assert is_month_end(ts) is True

    def test_mid_month(self) -> None:
        ts = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert is_month_end(ts) is False

    def test_three_days_before(self) -> None:
        ts = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
        assert is_month_end(ts, days_before=3) is True


class TestQuarterEnd:
    def test_march_end(self) -> None:
        ts = datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc)
        assert is_quarter_end(ts) is True

    def test_april(self) -> None:
        ts = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
        assert is_quarter_end(ts) is False


class TestLondonFix:
    def test_at_fix_time(self) -> None:
        ts = datetime(2026, 4, 1, 16, 0, 0, tzinfo=timezone.utc)
        assert is_london_fix_window(ts) is True

    def test_away_from_fix(self) -> None:
        ts = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert is_london_fix_window(ts) is False


class TestRollover:
    def test_at_rollover(self) -> None:
        ts = datetime(2026, 4, 1, 22, 0, 0, tzinfo=timezone.utc)
        assert is_rollover_window(ts) is True

    def test_away_from_rollover(self) -> None:
        ts = datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc)
        assert is_rollover_window(ts) is False


class TestSessionContext:
    def test_full_context(self) -> None:
        ts = datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc)
        ctx = get_session_context(ts)
        assert ctx.active_session == TradingSession.OVERLAP
        assert ctx.is_tradeable is True
        assert ctx.day_type in DayType

    def test_nfp_day_context(self) -> None:
        ts = datetime(2026, 4, 3, 14, 0, 0, tzinfo=timezone.utc)
        ctx = get_session_context(ts)
        assert ctx.day_type == DayType.NFP_DAY
        assert any("NFP" in note for note in ctx.special_notes)


# ── Prompt Construction Tests ─────────────────────────────────

class TestMacroPrompts:
    def test_system_prompt_versions_exist(self) -> None:
        assert "v1" in SYSTEM_PROMPTS
        assert "v2" in SYSTEM_PROMPTS
        assert CURRENT_VERSION in SYSTEM_PROMPTS

    def test_build_prompt_returns_both(self) -> None:
        system, user = build_macro_analysis_prompt(
            yield_curve={"yields": {"US_10Y": 4.25}, "curve_shape": "normal"},
            rate_differentials={"EUR/USD": {"differential": 1.5}},
            real_rates={"US_REAL_10Y": 1.8},
            economic_data={"US": {"CPI": 3.2}},
            commodity_data={"oil": {"wti_price": 75.0}},
            session_context={"active_session": "overlap", "day_type": "normal"},
            recent_news=["Fed signals rate hold", "ECB hawkish rhetoric"],
        )
        assert len(system) > 100
        assert len(user) > 100
        assert "currency_scores" in user  # JSON template
        assert "macro_regime" in user

    def test_prompt_includes_yields(self) -> None:
        system, user = build_macro_analysis_prompt(
            yield_curve={"yields": {"US_10Y": 4.25, "US_2Y": 4.50}, "spread_2y_10y": -0.25, "curve_shape": "inverted", "inversion": True},
            rate_differentials={},
            real_rates={},
            economic_data={},
            commodity_data={},
            session_context={},
            recent_news=[],
        )
        assert "INVERTED" in user
        assert "4.25" in user

    def test_prompt_includes_vix_warning(self) -> None:
        system, user = build_macro_analysis_prompt(
            yield_curve={},
            rate_differentials={},
            real_rates={},
            economic_data={},
            commodity_data={},
            session_context={},
            recent_news=[],
            vix=30.0,
        )
        assert "VIX ELEVATED" in user

    def test_event_impact_prompt(self) -> None:
        system, user = build_event_impact_prompt(
            event_name="US Non-Farm Payrolls",
            actual="150K",
            forecast="200K",
            previous="225K",
            currency="USD",
        )
        assert "150K" in user
        assert "200K" in user
        assert "USD" in user


# ── LLM Client Tests ─────────────────────────────────────────

class TestUsageStats:
    def test_cost_tracking(self) -> None:
        stats = UsageStats()
        cost = stats.record(
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            agent_name="macro_analyst",
        )
        assert cost > 0
        assert stats.total_calls == 1
        assert stats.total_input_tokens == 1000
        assert stats.total_output_tokens == 500

    def test_daily_cap_detection(self) -> None:
        stats = UsageStats()
        # Simulate many calls
        for _ in range(1000):
            stats.record(
                model="claude-sonnet-4-20250514",
                input_tokens=100000,
                output_tokens=50000,
                agent_name="test",
            )
        assert stats.is_over_daily_cap is True

    def test_agent_tracking(self) -> None:
        stats = UsageStats()
        stats.record("claude-sonnet-4-20250514", 100, 50, "macro_analyst")
        stats.record("claude-sonnet-4-20250514", 100, 50, "macro_analyst")
        stats.record("claude-haiku-4-5-20251001", 100, 50, "sentiment_agent")
        assert stats.calls_by_agent["macro_analyst"] == 2
        assert stats.calls_by_agent["sentiment_agent"] == 1


class TestLLMClientParsing:
    def test_parse_clean_json(self) -> None:
        client = LLMClient()
        result = client._parse_json_response('{"key": "value", "num": 42}')
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_parse_json_with_fences(self) -> None:
        client = LLMClient()
        result = client._parse_json_response('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"

    def test_parse_invalid_json(self) -> None:
        client = LLMClient()
        result = client._parse_json_response("this is not json")
        assert "parse_error" in result
        assert "raw_response" in result


# ── Macro Agent Validation Tests ──────────────────────────────

class TestMacroValidation:
    def test_derive_preferred_pairs(self) -> None:
        from src.agents.macro import MacroAnalystAgent
        agent = MacroAnalystAgent()
        scores = {
            "USD": -3, "EUR": +4, "GBP": +1, "JPY": -2,
            "CHF": 0, "AUD": +2, "CAD": +1, "NZD": +1,
        }
        pairs = agent._derive_preferred_pairs(scores)
        assert len(pairs) > 0
        # EUR/USD should be top pair (EUR +4, USD -3 = diff of 7)
        assert pairs[0]["pair"] == "EUR/USD"
        assert pairs[0]["direction"] == "LONG"
        assert pairs[0]["conviction"] == "high"

    def test_derive_pairs_with_low_scores(self) -> None:
        from src.agents.macro import MacroAnalystAgent
        agent = MacroAnalystAgent()
        scores = {c: 0 for c in ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]}
        pairs = agent._derive_preferred_pairs(scores)
        assert len(pairs) == 0  # No conviction

    def test_validate_clamps_scores(self) -> None:
        from src.agents.macro import MacroAnalystAgent
        agent = MacroAnalystAgent()
        raw = {
            "macro_regime": "risk_on",
            "currency_scores": {"USD": 10, "EUR": -10},  # Out of range
        }
        context = {"yield_curve": {}, "rate_differentials": {}, "real_rates": {},
                    "commodity_data": {}, "prices": {}, "recent_news": [],
                    "session_context": {}, "dxy": None, "vix": None, "geopolitical": {}}
        result = agent._validate_and_enrich(raw, context)
        assert result["currency_scores"]["USD"] == 0  # Clamped invalid
        assert result["currency_scores"]["EUR"] == 0  # Clamped invalid

    def test_validate_fills_missing_currencies(self) -> None:
        from src.agents.macro import MacroAnalystAgent
        agent = MacroAnalystAgent()
        raw = {
            "macro_regime": "neutral",
            "currency_scores": {"USD": 2},  # Only USD
        }
        context = {"yield_curve": {}, "rate_differentials": {}, "real_rates": {},
                    "commodity_data": {}, "prices": {}, "recent_news": [],
                    "session_context": {}, "dxy": None, "vix": None, "geopolitical": {}}
        result = agent._validate_and_enrich(raw, context)
        assert "EUR" in result["currency_scores"]
        assert "JPY" in result["currency_scores"]

    def test_validate_invalid_regime(self) -> None:
        from src.agents.macro import MacroAnalystAgent
        agent = MacroAnalystAgent()
        raw = {
            "macro_regime": "invalid_value",
            "currency_scores": {},
        }
        context = {"yield_curve": {}, "rate_differentials": {}, "real_rates": {},
                    "commodity_data": {}, "prices": {}, "recent_news": [],
                    "session_context": {}, "dxy": None, "vix": None, "geopolitical": {}}
        result = agent._validate_and_enrich(raw, context)
        assert result["macro_regime"] == "neutral"  # Defaults to neutral

    def test_data_quality_assessment(self) -> None:
        from src.agents.macro import MacroAnalystAgent
        agent = MacroAnalystAgent()

        full_context = {
            "yield_curve": {"yields": {"US_10Y": 4.25}},
            "rate_differentials": {"EUR/USD": {}},
            "real_rates": {"US_REAL_10Y": 1.8},
            "commodity_data": {"oil": {"wti_price": 75}},
            "dxy": 104.5,
            "vix": 18.0,
            "prices": {"EUR/USD": {}},
            "recent_news": ["headline"],
        }
        quality = agent._assess_data_quality(full_context)
        assert quality["yields"] == "good"
        assert quality["dxy"] == "good"
        assert quality["vix"] == "good"

        empty_context = {
            "yield_curve": {},
            "rate_differentials": {},
            "real_rates": {},
            "commodity_data": {},
            "dxy": None,
            "vix": None,
            "prices": {},
            "recent_news": [],
        }
        quality = agent._assess_data_quality(empty_context)
        assert quality["yields"] == "missing"
        assert quality["dxy"] == "missing"
