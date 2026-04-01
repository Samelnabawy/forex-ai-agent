"""
Unit tests for configuration layer.
Validates instruments, risk rules, and settings.
"""

from decimal import Decimal

import pytest

from src.config.instruments import (
    ALL_SYMBOLS,
    ALL_TIMEFRAMES,
    CLUSTER_MEMBERS,
    EXECUTION_TIMEFRAMES,
    INSTRUMENTS,
    CorrelationCluster,
    InstrumentCategory,
)
from src.config.risk_rules import RISK_RULES, validate_rules_integrity


class TestInstruments:
    """Validate the instrument universe matches PROJECT_SPEC.md Section 1.2."""

    def test_exactly_nine_instruments(self) -> None:
        assert len(INSTRUMENTS) == 9

    def test_all_symbols_list(self) -> None:
        assert len(ALL_SYMBOLS) == 9
        assert "EUR/USD" in ALL_SYMBOLS
        assert "XAU/USD" in ALL_SYMBOLS
        assert "WTI" in ALL_SYMBOLS

    def test_instrument_categories(self) -> None:
        forex_count = sum(
            1 for i in INSTRUMENTS.values()
            if i.category == InstrumentCategory.MAJOR_FOREX
        )
        assert forex_count == 7

        gold = INSTRUMENTS["XAU/USD"]
        assert gold.category == InstrumentCategory.GOLD

        oil = INSTRUMENTS["WTI"]
        assert oil.category == InstrumentCategory.OIL

    def test_pip_sizes(self) -> None:
        # Standard forex pairs: 0.0001
        assert INSTRUMENTS["EUR/USD"].pip_size == Decimal("0.0001")
        assert INSTRUMENTS["GBP/USD"].pip_size == Decimal("0.0001")

        # JPY pairs: 0.01
        assert INSTRUMENTS["USD/JPY"].pip_size == Decimal("0.01")

        # Gold and Oil: 0.01
        assert INSTRUMENTS["XAU/USD"].pip_size == Decimal("0.01")
        assert INSTRUMENTS["WTI"].pip_size == Decimal("0.01")

    def test_correlation_clusters(self) -> None:
        # EUR/USD and GBP/USD in USD strength cluster
        assert CorrelationCluster.USD_STRENGTH in INSTRUMENTS["EUR/USD"].correlation_clusters
        assert CorrelationCluster.USD_STRENGTH in INSTRUMENTS["GBP/USD"].correlation_clusters

        # JPY and CHF in safe haven cluster
        assert CorrelationCluster.SAFE_HAVEN in INSTRUMENTS["USD/JPY"].correlation_clusters
        assert CorrelationCluster.SAFE_HAVEN in INSTRUMENTS["USD/CHF"].correlation_clusters

        # AUD and NZD in antipodean cluster
        assert CorrelationCluster.ANTIPODEAN in INSTRUMENTS["AUD/USD"].correlation_clusters
        assert CorrelationCluster.ANTIPODEAN in INSTRUMENTS["NZD/USD"].correlation_clusters

        # Oil in commodity cluster
        assert CorrelationCluster.COMMODITY in INSTRUMENTS["WTI"].correlation_clusters
        assert CorrelationCluster.COMMODITY in INSTRUMENTS["USD/CAD"].correlation_clusters

    def test_cluster_members_populated(self) -> None:
        assert len(CLUSTER_MEMBERS) > 0
        assert "EUR/USD" in CLUSTER_MEMBERS[CorrelationCluster.USD_STRENGTH]
        assert "GBP/USD" in CLUSTER_MEMBERS[CorrelationCluster.USD_STRENGTH]

    def test_every_instrument_has_polygon_symbol(self) -> None:
        for sym, inst in INSTRUMENTS.items():
            assert inst.polygon_symbol, f"{sym} missing polygon_symbol"

    def test_every_instrument_has_twelve_data_symbol(self) -> None:
        for sym, inst in INSTRUMENTS.items():
            assert inst.twelve_data_symbol, f"{sym} missing twelve_data_symbol"

    def test_timeframes(self) -> None:
        assert ALL_TIMEFRAMES == ["1m", "5m", "15m", "1h", "4h", "1d"]
        assert EXECUTION_TIMEFRAMES == ["5m", "15m", "1h"]


class TestRiskRules:
    """
    Validate Section 1.5 non-negotiable risk rules.
    These tests are a safety net — if they fail, something critically wrong happened.
    """

    def test_rule_1_position_sizing(self) -> None:
        assert RISK_RULES.max_risk_per_trade_pct == Decimal("1.0")

    def test_rule_2_daily_loss_limit(self) -> None:
        assert RISK_RULES.daily_loss_limit_pct == Decimal("-2.0")

    def test_rule_3_weekly_loss_limit(self) -> None:
        assert RISK_RULES.weekly_loss_limit_pct == Decimal("-5.0")

    def test_rule_4_correlation_guard(self) -> None:
        assert RISK_RULES.max_correlated_trades == 2

    def test_rule_5_news_blackout(self) -> None:
        assert RISK_RULES.news_blackout_minutes == 15

    def test_rule_6_spread_filter(self) -> None:
        assert RISK_RULES.max_spread_multiplier == Decimal("2.0")

    def test_rule_7_session_filter(self) -> None:
        assert RISK_RULES.session_start_utc == 8
        assert RISK_RULES.session_end_utc == 21

    def test_rule_8_stop_loss_required(self) -> None:
        assert RISK_RULES.require_stop_loss is True

    def test_rule_9_circuit_breaker(self) -> None:
        assert RISK_RULES.circuit_breaker_losses == 3
        assert RISK_RULES.circuit_breaker_pause_hours == 2

    def test_rule_10_weekend_rule(self) -> None:
        assert RISK_RULES.weekend_close_day == 4  # Friday
        assert RISK_RULES.weekend_close_hour_utc == 20

    def test_max_concurrent_trades(self) -> None:
        assert RISK_RULES.max_concurrent_trades == 3

    def test_min_risk_reward(self) -> None:
        assert RISK_RULES.min_risk_reward_ratio == Decimal("1.5")

    def test_rules_are_frozen(self) -> None:
        """Risk rules dataclass must be immutable."""
        with pytest.raises(AttributeError):
            RISK_RULES.max_risk_per_trade_pct = Decimal("5.0")  # type: ignore[misc]

    def test_integrity_validation(self) -> None:
        """validate_rules_integrity() should pass with untampered rules."""
        validate_rules_integrity()  # Should not raise


class TestModels:
    """Validate Pydantic models."""

    def test_price_tick_spread_computation(self) -> None:
        from src.models import PriceTick
        from datetime import datetime, timezone

        tick = PriceTick(
            instrument="EUR/USD",
            bid=Decimal("1.0840"),
            ask=Decimal("1.0842"),
            ts=datetime.now(timezone.utc),
        )
        assert tick.spread == Decimal("0.0002")

    def test_trade_decision_validation(self) -> None:
        from src.models import Direction, TradeDecision
        from datetime import datetime, timezone

        decision = TradeDecision(
            trade_id="T-20260401-001",
            final_decision="EXECUTE",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry=Decimal("1.0845"),
            stop_loss=Decimal("1.0810"),
            take_profit_1=Decimal("1.0900"),
            position_size=Decimal("0.5"),
            confidence=Decimal("0.74"),
            agent_votes={"technical": "BUY"},
        )
        assert decision.direction == Direction.LONG
        assert decision.confidence == Decimal("0.74")

    def test_confidence_bounds(self) -> None:
        from src.models import TechnicalSignal, SignalStrength
        from datetime import datetime, timezone

        with pytest.raises(Exception):
            TechnicalSignal(
                instrument="EUR/USD",
                ts=datetime.now(timezone.utc),
                signal=SignalStrength.BUY,
                confidence=Decimal("1.5"),  # > 1.0, should fail
                timeframe="1h",
                entry=Decimal("1.0845"),
                stop_loss=Decimal("1.0810"),
                take_profit=Decimal("1.0900"),
            )
