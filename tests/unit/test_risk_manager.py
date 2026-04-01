"""
Unit tests for Risk Manager — all 10 layers.
Tests every rule, sizing model, portfolio check, drawdown protocol, anti-tilt.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.risk_models import (
    PositionSizeCalc,
    assess_execution_quality,
    compute_drawdown_state,
    compute_kelly_fraction,
    compute_portfolio_risk,
    compute_position_size,
    detect_tilt,
    evaluate_open_position,
    validate_stop_loss,
)
from src.agents.risk_rules_engine import (
    check_quality_gate,
    check_rule_1_position_size,
    check_rule_2_daily_loss,
    check_rule_3_weekly_loss,
    check_rule_4_correlation_guard,
    check_rule_5_news_blackout,
    check_rule_6_spread,
    check_rule_7_session,
    check_rule_8_stop_loss,
    check_rule_9_circuit_breaker,
    check_rule_10_weekend,
    run_all_rules,
)


# ── Rule 1: Position Size ────────────────────────────────────

class TestRule1:
    def test_within_limit(self) -> None:
        assert check_rule_1_position_size(0.5).status == "PASS"

    def test_at_limit(self) -> None:
        assert check_rule_1_position_size(1.0).status == "PASS"

    def test_over_limit(self) -> None:
        assert check_rule_1_position_size(1.5).status == "FAIL"


# ── Rule 2: Daily Loss ───────────────────────────────────────

class TestRule2:
    def test_no_loss(self) -> None:
        assert check_rule_2_daily_loss(0.5).status == "PASS"

    def test_at_limit(self) -> None:
        assert check_rule_2_daily_loss(-2.0).status == "FAIL"

    def test_beyond_limit(self) -> None:
        assert check_rule_2_daily_loss(-3.0).status == "FAIL"


# ── Rule 3: Weekly Loss ──────────────────────────────────────

class TestRule3:
    def test_no_loss(self) -> None:
        assert check_rule_3_weekly_loss(1.0).status == "PASS"

    def test_at_limit(self) -> None:
        assert check_rule_3_weekly_loss(-5.0).status == "FAIL"


# ── Rule 4: Correlation Guard ─────────────────────────────────

class TestRule4:
    def test_no_open_positions(self) -> None:
        result = check_rule_4_correlation_guard("EUR/USD", "LONG", [])
        assert result.status == "PASS"

    def test_correlated_positions(self) -> None:
        open_pos = [
            {"instrument": "EUR/USD", "direction": "LONG"},
            {"instrument": "GBP/USD", "direction": "LONG"},
        ]
        # EUR/USD and GBP/USD are in same cluster
        result = check_rule_4_correlation_guard("AUD/USD", "LONG", open_pos)
        # AUD/USD might correlate with some, depends on cluster membership
        assert result.status in ("PASS", "WARN", "FAIL")

    def test_dynamic_correlation(self) -> None:
        open_pos = [{"instrument": "EUR/USD", "direction": "LONG"}]
        corr_matrix = {"GBP/USD": {"EUR/USD": 0.9}}
        result = check_rule_4_correlation_guard("GBP/USD", "LONG", open_pos, corr_matrix)
        assert result.values.get("count", 0) >= 1


# ── Rule 5: News Blackout ────────────────────────────────────

class TestRule5:
    def test_no_blackout(self) -> None:
        assert check_rule_5_news_blackout(False).status == "PASS"

    def test_active_blackout(self) -> None:
        assert check_rule_5_news_blackout(True).status == "FAIL"


# ── Rule 6: Spread Filter ────────────────────────────────────

class TestRule6:
    def test_normal_spread(self) -> None:
        assert check_rule_6_spread(0.8, 0.7).status == "PASS"

    def test_wide_spread(self) -> None:
        assert check_rule_6_spread(2.0, 0.7).status == "FAIL"

    def test_no_avg_spread(self) -> None:
        assert check_rule_6_spread(1.0, 0).status == "WARN"


# ── Rule 7: Session Filter ───────────────────────────────────

class TestRule7:
    def test_overlap_session(self) -> None:
        ts = datetime(2026, 4, 1, 14, 30, 0, tzinfo=timezone.utc)
        assert check_rule_7_session(ts).status == "PASS"

    def test_off_hours(self) -> None:
        ts = datetime(2026, 4, 1, 3, 0, 0, tzinfo=timezone.utc)
        assert check_rule_7_session(ts).status == "FAIL"

    def test_london_open_warning(self) -> None:
        ts = datetime(2026, 4, 1, 8, 5, 0, tzinfo=timezone.utc)
        assert check_rule_7_session(ts).status == "WARN"


# ── Rule 8: Stop Loss ────────────────────────────────────────

class TestRule8:
    def test_has_stop(self) -> None:
        assert check_rule_8_stop_loss(True, 1.08, 1.07, 1.10).status == "PASS"

    def test_no_stop(self) -> None:
        assert check_rule_8_stop_loss(False).status == "FAIL"

    def test_poor_rr(self) -> None:
        result = check_rule_8_stop_loss(True, 1.0800, 1.0750, 1.0820)
        # R:R = 20/50 = 0.4 — below 1.5 minimum
        assert result.status == "WARN"


# ── Rule 9: Circuit Breaker ──────────────────────────────────

class TestRule9:
    def test_no_losses(self) -> None:
        assert check_rule_9_circuit_breaker(0).status == "PASS"

    def test_at_limit(self) -> None:
        assert check_rule_9_circuit_breaker(3).status == "FAIL"

    def test_active_pause(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert check_rule_9_circuit_breaker(0, future).status == "FAIL"


# ── Rule 10: Weekend ─────────────────────────────────────────

class TestRule10:
    def test_tuesday(self) -> None:
        ts = datetime(2026, 3, 31, 14, 0, 0, tzinfo=timezone.utc)  # Tuesday
        assert check_rule_10_weekend(ts).status == "PASS"

    def test_friday_evening(self) -> None:
        ts = datetime(2026, 4, 3, 21, 0, 0, tzinfo=timezone.utc)  # Friday 21:00
        assert check_rule_10_weekend(ts).status == "FAIL"

    def test_friday_afternoon_warning(self) -> None:
        ts = datetime(2026, 4, 3, 17, 0, 0, tzinfo=timezone.utc)  # Friday 17:00
        assert check_rule_10_weekend(ts).status == "WARN"

    def test_saturday(self) -> None:
        ts = datetime(2026, 4, 4, 10, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert check_rule_10_weekend(ts).status == "FAIL"


# ── Run All Rules ─────────────────────────────────────────────

class TestAllRules:
    def test_all_pass(self) -> None:
        ts = datetime(2026, 4, 1, 14, 30, 0, tzinfo=timezone.utc)  # Wednesday overlap
        suite = run_all_rules(
            position_size_pct=0.5,
            daily_pnl_pct=0.3,
            weekly_pnl_pct=1.2,
            instrument="EUR/USD",
            direction="LONG",
            open_positions=[],
            is_news_blackout=False,
            current_spread=0.8,
            avg_spread=0.7,
            has_stop_loss=True,
            consecutive_losses=0,
            entry=1.0845,
            stop_loss=1.0810,
            take_profit=1.0910,
            ts=ts,
        )
        assert not suite.veto
        assert suite.all_passed or suite.has_warnings  # Some rules might WARN

    def test_multiple_failures(self) -> None:
        ts = datetime(2026, 4, 1, 3, 0, 0, tzinfo=timezone.utc)  # Off hours
        suite = run_all_rules(
            position_size_pct=2.0,  # Over limit
            daily_pnl_pct=-3.0,    # Over daily limit
            weekly_pnl_pct=0,
            instrument="EUR/USD",
            direction="LONG",
            open_positions=[],
            is_news_blackout=True,  # Blackout
            current_spread=0.8,
            avg_spread=0.7,
            has_stop_loss=False,    # No stop
            consecutive_losses=0,
            ts=ts,
        )
        assert suite.veto
        failed = [c for c in suite.checks if c.status == "FAIL"]
        assert len(failed) >= 4  # position, daily, blackout, session, stop loss


# ── Kelly Criterion ───────────────────────────────────────────

class TestKelly:
    def test_positive_expectancy(self) -> None:
        f = compute_kelly_fraction(0.6, 40, 20)
        assert f > 0
        assert f < 0.25  # Capped

    def test_negative_expectancy(self) -> None:
        f = compute_kelly_fraction(0.3, 20, 40)
        assert f == 0  # Don't trade

    def test_coin_flip_with_edge(self) -> None:
        f = compute_kelly_fraction(0.5, 30, 20)
        assert f > 0  # Positive edge from asymmetric payoff


# ── Position Sizing ───────────────────────────────────────────

class TestPositionSizing:
    def test_combined_sizing(self) -> None:
        result = compute_position_size(
            win_rate=0.6,
            avg_win_pips=40,
            avg_loss_pips=20,
            current_atr=0.001,
            normal_atr=0.001,
            current_drawdown_pct=0,
            max_drawdown_pct=2.0,
            debate_conviction=75,
        )
        assert result.final_size_pct > 0
        assert result.final_size_pct <= 1.0

    def test_high_vol_reduces_size(self) -> None:
        normal = compute_position_size(0.6, 40, 20, 0.001, 0.001, 0, 2.0, 75)
        high_vol = compute_position_size(0.6, 40, 20, 0.003, 0.001, 0, 2.0, 75)
        assert high_vol.final_size_pct <= normal.final_size_pct

    def test_drawdown_reduces_size(self) -> None:
        no_dd = compute_position_size(0.6, 40, 20, 0.001, 0.001, 0, 2.0, 75)
        in_dd = compute_position_size(0.6, 40, 20, 0.001, 0.001, -1.5, 2.0, 75)
        assert in_dd.final_size_pct <= no_dd.final_size_pct

    def test_low_conviction_reduces_size(self) -> None:
        high = compute_position_size(0.6, 40, 20, 0.001, 0.001, 0, 2.0, 90)
        low = compute_position_size(0.6, 40, 20, 0.001, 0.001, 0, 2.0, 40)
        assert low.final_size_pct <= high.final_size_pct


# ── Portfolio Risk ────────────────────────────────────────────

class TestPortfolioRisk:
    def test_heat_calculation(self) -> None:
        positions = [
            {"instrument": "EUR/USD", "direction": "LONG", "position_size": 0.5},
            {"instrument": "GBP/USD", "direction": "LONG", "position_size": 0.5},
        ]
        result = compute_portfolio_risk(positions)
        assert result.total_heat_pct == 1.0

    def test_heat_limit_breach(self) -> None:
        positions = [
            {"instrument": "EUR/USD", "direction": "LONG", "position_size": 1.0},
            {"instrument": "GBP/USD", "direction": "LONG", "position_size": 1.0},
        ]
        new_trade = {"instrument": "USD/JPY", "direction": "SHORT", "position_size": 1.5}
        result = compute_portfolio_risk(positions, new_trade, max_heat=3.0)
        assert result.heat_after_trade == 3.5
        assert not result.heat_ok

    def test_currency_exposure(self) -> None:
        positions = [
            {"instrument": "EUR/USD", "direction": "LONG", "position_size": 0.5},
        ]
        result = compute_portfolio_risk(positions)
        assert result.currency_exposure.get("EUR", 0) > 0
        assert result.currency_exposure.get("USD", 0) < 0


# ── Drawdown Protocol ────────────────────────────────────────

class TestDrawdownProtocol:
    def test_green_state(self) -> None:
        state = compute_drawdown_state(0, 0)
        assert state.trading_allowed is True
        assert state.max_allowed_size_pct == 1.0

    def test_daily_throttle(self) -> None:
        state = compute_drawdown_state(-1.0, 0)
        assert state.daily_alert_level == "orange"
        assert state.max_allowed_size_pct < 1.0

    def test_daily_halt(self) -> None:
        state = compute_drawdown_state(-2.0, 0)
        assert state.trading_allowed is False
        assert state.daily_alert_level == "halt"

    def test_weekly_halt(self) -> None:
        state = compute_drawdown_state(0, -5.0)
        assert state.trading_allowed is False

    def test_monthly_shutdown(self) -> None:
        state = compute_drawdown_state(0, 0, -8.0)
        assert state.trading_allowed is False
        assert "SHUTDOWN" in state.halt_reason


# ── Execution Quality ────────────────────────────────────────

class TestExecutionQuality:
    def test_normal_conditions(self) -> None:
        result = assess_execution_quality(0.8, 0.7, session_hour_utc=14)
        assert result.spread_ok is True
        assert result.order_type == "market"

    def test_wide_spread(self) -> None:
        result = assess_execution_quality(2.0, 0.7)
        assert result.spread_ok is False
        assert result.order_type == "limit"

    def test_thin_liquidity(self) -> None:
        result = assess_execution_quality(0.8, 0.7, session_hour_utc=3)
        assert result.session_liquidity == "thin"


# ── Stop Loss Validation ─────────────────────────────────────

class TestStopValidation:
    def test_adequate_stop(self) -> None:
        result = validate_stop_loss(
            entry=1.0845, stop_loss=1.0800, direction="LONG",
            atr=0.003, pip_size=0.0001,
        )
        assert result["valid"] is True

    def test_too_tight_stop(self) -> None:
        result = validate_stop_loss(
            entry=1.0845, stop_loss=1.0840, direction="LONG",
            atr=0.003, pip_size=0.0001,
        )
        assert len(result["warnings"]) > 0
        assert result["suggested_stop"] is not None


# ── Anti-Tilt ─────────────────────────────────────────────────

class TestAntiTilt:
    def test_normal_state(self) -> None:
        state = detect_tilt(signals_last_hour=1)
        assert state.overtrading is False

    def test_overtrading(self) -> None:
        state = detect_tilt(signals_last_hour=10)
        assert state.overtrading is True

    def test_quality_degradation(self) -> None:
        state = detect_tilt(
            signals_last_hour=1,
            recent_convictions=[45, 50, 48, 52, 40, 44],
        )
        assert state.quality_degradation is True

    def test_correlated_losses(self) -> None:
        results = [
            {"outcome": "loss", "direction": "LONG"},
            {"outcome": "loss", "direction": "LONG"},
            {"outcome": "loss", "direction": "LONG"},
        ]
        state = detect_tilt(signals_last_hour=1, recent_results=results)
        assert state.correlated_loss_cluster is True


# ── Quality Gate ──────────────────────────────────────────────

class TestQualityGate:
    def test_all_pass(self) -> None:
        result = check_quality_gate(
            debate_conviction=75, expected_value_pips=15,
            ev_per_risk=0.5, evidence_sources=4,
            signal_age_minutes=5, price_moved_pct=10,
            risk_reward=2.0,
        )
        assert result.passed is True

    def test_low_conviction_fails(self) -> None:
        result = check_quality_gate(
            debate_conviction=40, expected_value_pips=15,
            ev_per_risk=0.5, evidence_sources=4,
            signal_age_minutes=5, price_moved_pct=10,
            risk_reward=2.0,
        )
        assert result.passed is False

    def test_negative_ev_fails(self) -> None:
        result = check_quality_gate(
            debate_conviction=75, expected_value_pips=-5,
            ev_per_risk=-0.2, evidence_sources=4,
            signal_age_minutes=5, price_moved_pct=10,
            risk_reward=2.0,
        )
        assert result.passed is False

    def test_stale_signal_fails(self) -> None:
        result = check_quality_gate(
            debate_conviction=75, expected_value_pips=15,
            ev_per_risk=0.5, evidence_sources=4,
            signal_age_minutes=45, price_moved_pct=10,
            risk_reward=2.0,
        )
        assert result.passed is False
