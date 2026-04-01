"""
Risk Management Rules — Section 1.5 of PROJECT_SPEC.md

████████████████████████████████████████████████████████████████
██  THESE RULES ARE HARDCODED AND CANNOT BE OVERRIDDEN.      ██
██  No agent, no config, no user input can change them.       ██
██  Modifying this file requires explicit approval from Sam.  ██
████████████████████████████████████████████████████████████████
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final


@dataclass(frozen=True, slots=True)
class RiskRules:
    """
    Immutable risk parameters. Frozen dataclass = cannot be mutated at runtime.
    Every value here maps 1:1 to a rule in Section 1.5.
    """

    # Rule 1: Position Sizing — max % of capital risked per trade
    max_risk_per_trade_pct: Decimal = Decimal("1.0")

    # Rule 2: Daily Loss Limit — halt all trading if breached
    daily_loss_limit_pct: Decimal = Decimal("-2.0")

    # Rule 3: Weekly Loss Limit — halt all trading until Monday
    weekly_loss_limit_pct: Decimal = Decimal("-5.0")

    # Rule 4: Correlation Guard — max correlated trades open simultaneously
    max_correlated_trades: int = 2

    # Rule 5: News Blackout — minutes before/after high-impact events
    news_blackout_minutes: int = 15

    # Rule 6: Spread Filter — skip if spread > multiplier × average
    max_spread_multiplier: Decimal = Decimal("2.0")

    # Rule 7: Session Filter — only trade during these UTC windows
    # London: 08:00-16:00 UTC, NY: 13:00-21:00 UTC
    # Combined active window: 08:00-21:00 UTC
    session_start_utc: int = 8    # hour
    session_end_utc: int = 21     # hour

    # Rule 8: Stop Loss — every trade must have one. Boolean flag
    # enforced in the Risk Manager logic, not a numeric value.
    require_stop_loss: bool = True

    # Rule 9: Circuit Breaker — consecutive losses before forced pause
    circuit_breaker_losses: int = 3
    circuit_breaker_pause_hours: int = 2

    # Rule 10: Weekend Rule — all positions closed by this time Friday
    weekend_close_day: int = 4      # Friday = 4 (Monday=0)
    weekend_close_hour_utc: int = 20  # 20:00 UTC

    # ── Derived constraints (not in spec but implied) ─────────
    max_concurrent_trades: int = 3
    min_risk_reward_ratio: Decimal = Decimal("1.5")


# Singleton — import this everywhere
RISK_RULES: Final[RiskRules] = RiskRules()


def validate_rules_integrity() -> None:
    """
    Runtime assertion that rules haven't been tampered with.
    Call this on startup and periodically during operation.
    """
    r = RISK_RULES
    assert r.max_risk_per_trade_pct == Decimal("1.0"), "Rule 1 tampered"
    assert r.daily_loss_limit_pct == Decimal("-2.0"), "Rule 2 tampered"
    assert r.weekly_loss_limit_pct == Decimal("-5.0"), "Rule 3 tampered"
    assert r.max_correlated_trades == 2, "Rule 4 tampered"
    assert r.news_blackout_minutes == 15, "Rule 5 tampered"
    assert r.max_spread_multiplier == Decimal("2.0"), "Rule 6 tampered"
    assert r.session_start_utc == 8, "Rule 7 tampered"
    assert r.session_end_utc == 21, "Rule 7 tampered"
    assert r.require_stop_loss is True, "Rule 8 tampered"
    assert r.circuit_breaker_losses == 3, "Rule 9 tampered"
    assert r.circuit_breaker_pause_hours == 2, "Rule 9 tampered"
    assert r.weekend_close_day == 4, "Rule 10 tampered"
    assert r.weekend_close_hour_utc == 20, "Rule 10 tampered"
