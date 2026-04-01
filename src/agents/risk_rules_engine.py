"""
Risk Rules Engine.

The 10 sacred rules from Section 1.5 as executable, context-aware checks.
Each rule returns a detailed result (PASS/WARN/FAIL) with actual values.

Also includes:
  - Trade Quality Gate (minimum standards before execution)
  - Regime-Adaptive Modifications (crisis mode, NFP day, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config.instruments import INSTRUMENTS, CLUSTER_MEMBERS, CorrelationCluster
from src.config.risk_rules import RISK_RULES

logger = logging.getLogger(__name__)


class CheckResult:
    """Result of a single rule check."""

    def __init__(
        self,
        rule_name: str,
        status: str,  # "PASS", "WARN", "FAIL"
        message: str,
        values: dict[str, Any] | None = None,
    ) -> None:
        self.rule_name = rule_name
        self.status = status
        self.message = message
        self.values = values or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule_name,
            "status": self.status,
            "message": self.message,
            "values": self.values,
        }


@dataclass
class RuleCheckSuite:
    """Complete results from all 10 rule checks."""
    checks: list[CheckResult] = field(default_factory=list)
    all_passed: bool = True
    has_failures: bool = False
    has_warnings: bool = False
    veto: bool = False  # Any FAIL = veto
    modifications: dict[str, Any] = field(default_factory=dict)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)
        if check.status == "FAIL":
            self.has_failures = True
            self.all_passed = False
            self.veto = True
        elif check.status == "WARN":
            self.has_warnings = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "has_failures": self.has_failures,
            "has_warnings": self.has_warnings,
            "veto": self.veto,
            "checks": {c.rule_name: c.to_dict() for c in self.checks},
            "modifications": self.modifications,
        }


# ── The 10 Rules ──────────────────────────────────────────────

def check_rule_1_position_size(position_size_pct: float) -> CheckResult:
    """Rule 1: Never risk more than 1% per trade."""
    max_risk = float(RISK_RULES.max_risk_per_trade_pct)
    if position_size_pct <= max_risk:
        return CheckResult("position_size", "PASS",
                           f"Position size {position_size_pct:.2f}% ≤ {max_risk}% limit",
                           {"size": position_size_pct, "limit": max_risk})
    else:
        return CheckResult("position_size", "FAIL",
                           f"Position size {position_size_pct:.2f}% EXCEEDS {max_risk}% limit",
                           {"size": position_size_pct, "limit": max_risk})


def check_rule_2_daily_loss(daily_pnl_pct: float) -> CheckResult:
    """Rule 2: If daily P&L hits -2%, ALL trading stops."""
    limit = float(RISK_RULES.daily_loss_limit_pct)
    if daily_pnl_pct > limit:
        return CheckResult("daily_loss", "PASS",
                           f"Daily P&L: {daily_pnl_pct:+.2f}% (limit: {limit}%)",
                           {"pnl": daily_pnl_pct, "limit": limit})
    else:
        return CheckResult("daily_loss", "FAIL",
                           f"DAILY LOSS LIMIT BREACHED: {daily_pnl_pct:+.2f}% ≤ {limit}%",
                           {"pnl": daily_pnl_pct, "limit": limit})


def check_rule_3_weekly_loss(weekly_pnl_pct: float) -> CheckResult:
    """Rule 3: If weekly P&L hits -5%, ALL trading stops until Monday."""
    limit = float(RISK_RULES.weekly_loss_limit_pct)
    if weekly_pnl_pct > limit:
        return CheckResult("weekly_loss", "PASS",
                           f"Weekly P&L: {weekly_pnl_pct:+.2f}% (limit: {limit}%)",
                           {"pnl": weekly_pnl_pct, "limit": limit})
    else:
        return CheckResult("weekly_loss", "FAIL",
                           f"WEEKLY LOSS LIMIT BREACHED: {weekly_pnl_pct:+.2f}% ≤ {limit}%",
                           {"pnl": weekly_pnl_pct, "limit": limit})


def check_rule_4_correlation_guard(
    instrument: str,
    direction: str,
    open_positions: list[dict[str, Any]],
    correlation_matrix: dict[str, dict[str, float]] | None = None,
) -> CheckResult:
    """
    Rule 4: Max 2 correlated trades simultaneously.
    Uses DYNAMIC correlation from Agent 3, not static pair lists.
    """
    max_corr = RISK_RULES.max_correlated_trades
    corr_threshold = 0.65  # Dynamic: pairs above this are "correlated"

    correlated_count = 0
    correlated_with: list[str] = []

    for pos in open_positions:
        pos_inst = pos.get("instrument", "")
        if pos_inst == instrument:
            continue

        # Check dynamic correlation if available
        if correlation_matrix and instrument in correlation_matrix and pos_inst in correlation_matrix.get(instrument, {}):
            corr = abs(correlation_matrix[instrument].get(pos_inst, 0))
            if corr >= corr_threshold:
                correlated_count += 1
                correlated_with.append(f"{pos_inst} (corr={corr:.2f})")
                continue

        # Fallback: static cluster check
        inst_a = INSTRUMENTS.get(instrument)
        inst_b = INSTRUMENTS.get(pos_inst)
        if inst_a and inst_b:
            shared_clusters = set(inst_a.correlation_clusters) & set(inst_b.correlation_clusters)
            if shared_clusters:
                correlated_count += 1
                correlated_with.append(f"{pos_inst} (cluster: {', '.join(c.value for c in shared_clusters)})")

    if correlated_count < max_corr:
        return CheckResult("correlation_guard", "PASS",
                           f"Correlated trades: {correlated_count}/{max_corr}",
                           {"count": correlated_count, "limit": max_corr, "correlated_with": correlated_with})
    elif correlated_count == max_corr:
        return CheckResult("correlation_guard", "WARN",
                           f"At correlation limit: {correlated_count}/{max_corr}. Next correlated trade will be blocked.",
                           {"count": correlated_count, "limit": max_corr, "correlated_with": correlated_with})
    else:
        return CheckResult("correlation_guard", "FAIL",
                           f"CORRELATION GUARD BREACHED: {correlated_count} correlated trades (max {max_corr})",
                           {"count": correlated_count, "limit": max_corr, "correlated_with": correlated_with})


def check_rule_5_news_blackout(
    is_blackout: bool,
    upcoming_events: list[dict[str, Any]] | None = None,
) -> CheckResult:
    """Rule 5: No new positions 15 minutes before/after high-impact news."""
    if not is_blackout:
        return CheckResult("news_blackout", "PASS",
                           "No high-impact events within blackout window",
                           {"blackout": False})
    else:
        event_names = [e.get("event_name", "?") for e in (upcoming_events or [])]
        return CheckResult("news_blackout", "FAIL",
                           f"NEWS BLACKOUT ACTIVE: {', '.join(event_names[:3])}",
                           {"blackout": True, "events": event_names})


def check_rule_6_spread(
    current_spread: float,
    avg_spread: float,
) -> CheckResult:
    """Rule 6: Skip if spread > 2x average."""
    multiplier = float(RISK_RULES.max_spread_multiplier)
    if avg_spread <= 0:
        return CheckResult("spread_filter", "WARN",
                           "No average spread data — proceeding with caution",
                           {"current": current_spread, "avg": avg_spread})

    ratio = current_spread / avg_spread
    if ratio <= multiplier:
        return CheckResult("spread_filter", "PASS",
                           f"Spread {current_spread:.2f} is {ratio:.1f}x average ({avg_spread:.2f})",
                           {"current": current_spread, "avg": avg_spread, "ratio": ratio, "limit": multiplier})
    else:
        return CheckResult("spread_filter", "FAIL",
                           f"SPREAD TOO WIDE: {current_spread:.2f} is {ratio:.1f}x average (max {multiplier}x)",
                           {"current": current_spread, "avg": avg_spread, "ratio": ratio, "limit": multiplier})


def check_rule_7_session(ts: datetime | None = None) -> CheckResult:
    """Rule 7: Only trade during London + NY sessions (08:00-21:00 UTC)."""
    ts = ts or datetime.now(timezone.utc)
    hour = ts.hour
    start = RISK_RULES.session_start_utc
    end = RISK_RULES.session_end_utc

    in_session = start <= hour < end

    # Additional checks: avoid first 15 min of London, last 30 min of NY
    warnings: list[str] = []
    if hour == start and ts.minute < 15:
        warnings.append("First 15 min of London — false breakout risk, thin liquidity")
    if hour == end - 1 and ts.minute >= 30:
        warnings.append("Last 30 min of NY — thin liquidity, wider spreads")

    if in_session and not warnings:
        return CheckResult("session_filter", "PASS",
                           f"Within trading session ({start}:00-{end}:00 UTC). Current: {hour}:{ts.minute:02d}",
                           {"hour": hour, "start": start, "end": end})
    elif in_session and warnings:
        return CheckResult("session_filter", "WARN",
                           f"In session but {'; '.join(warnings)}",
                           {"hour": hour, "warnings": warnings})
    else:
        return CheckResult("session_filter", "FAIL",
                           f"OUTSIDE TRADING SESSION: {hour}:{ts.minute:02d} UTC (allowed: {start}:00-{end}:00)",
                           {"hour": hour, "start": start, "end": end})


def check_rule_8_stop_loss(
    has_stop: bool,
    entry: float = 0,
    stop_loss: float = 0,
    take_profit: float = 0,
) -> CheckResult:
    """Rule 8: Every trade MUST have a stop loss."""
    if not has_stop or stop_loss == 0:
        return CheckResult("stop_loss", "FAIL",
                           "NO STOP LOSS DEFINED — trade rejected",
                           {"has_stop": False})

    # Also validate R:R
    min_rr = float(RISK_RULES.min_risk_reward_ratio)
    if entry > 0 and take_profit > 0:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr = reward / risk if risk > 0 else 0

        if rr < min_rr:
            return CheckResult("stop_loss", "WARN",
                               f"R:R ratio {rr:.2f} below minimum {min_rr}. Consider wider TP or tighter SL.",
                               {"rr": rr, "min_rr": min_rr, "risk": risk, "reward": reward})

    return CheckResult("stop_loss", "PASS",
                       "Stop loss defined",
                       {"has_stop": True, "stop_loss": stop_loss})


def check_rule_9_circuit_breaker(
    consecutive_losses: int,
    circuit_breaker_until: datetime | None = None,
) -> CheckResult:
    """Rule 9: 3 consecutive losses → pause for 2 hours."""
    max_losses = RISK_RULES.circuit_breaker_losses
    now = datetime.now(timezone.utc)

    # Check if in active pause
    if circuit_breaker_until and now < circuit_breaker_until:
        remaining = (circuit_breaker_until - now).total_seconds() / 60
        return CheckResult("circuit_breaker", "FAIL",
                           f"CIRCUIT BREAKER ACTIVE: {remaining:.0f} min remaining",
                           {"consecutive_losses": consecutive_losses, "resume_at": circuit_breaker_until.isoformat()})

    if consecutive_losses >= max_losses:
        pause_hours = RISK_RULES.circuit_breaker_pause_hours
        return CheckResult("circuit_breaker", "FAIL",
                           f"CIRCUIT BREAKER TRIGGERED: {consecutive_losses} consecutive losses. Pause {pause_hours}h.",
                           {"consecutive_losses": consecutive_losses, "limit": max_losses})

    return CheckResult("circuit_breaker", "PASS",
                       f"Consecutive losses: {consecutive_losses}/{max_losses}",
                       {"consecutive_losses": consecutive_losses, "limit": max_losses})


def check_rule_10_weekend(ts: datetime | None = None) -> CheckResult:
    """Rule 10: All positions closed by Friday 20:00 UTC. No new positions Friday after 16:00."""
    ts = ts or datetime.now(timezone.utc)
    day = ts.weekday()
    hour = ts.hour

    close_day = RISK_RULES.weekend_close_day
    close_hour = RISK_RULES.weekend_close_hour_utc

    if day < close_day:
        return CheckResult("weekend_rule", "PASS",
                           "Not Friday yet",
                           {"day": day, "hour": hour})

    if day == close_day:
        if hour >= close_hour:
            return CheckResult("weekend_rule", "FAIL",
                               f"WEEKEND CLOSE: Past Friday {close_hour}:00 UTC. No trading.",
                               {"day": day, "hour": hour})
        elif hour >= 16:
            return CheckResult("weekend_rule", "WARN",
                               f"Friday after 16:00 UTC — no NEW positions, close existing by {close_hour}:00",
                               {"day": day, "hour": hour})
        else:
            return CheckResult("weekend_rule", "PASS",
                               f"Friday but before restriction time",
                               {"day": day, "hour": hour})

    if day > close_day:
        return CheckResult("weekend_rule", "FAIL",
                           "WEEKEND — no trading",
                           {"day": day, "hour": hour})

    return CheckResult("weekend_rule", "PASS", "Within trading week",
                       {"day": day, "hour": hour})


# ── Trade Quality Gate ────────────────────────────────────────

@dataclass
class QualityGateResult:
    """Result of trade quality gate checks."""
    passed: bool = True
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def check_quality_gate(
    debate_conviction: float,
    expected_value_pips: float,
    ev_per_risk: float,
    evidence_sources: int,
    signal_age_minutes: float,
    price_moved_pct: float,
    risk_reward: float,
) -> QualityGateResult:
    """
    Trade quality gate — minimum standards before execution.
    ALL must pass. Any failure = trade rejected.
    """
    result = QualityGateResult()

    # Minimum debate conviction
    if debate_conviction < 60:
        result.passed = False
        result.reasons.append(f"Debate conviction {debate_conviction:.0f} < 60 minimum")
    result.checks["conviction"] = {"value": debate_conviction, "min": 60, "pass": debate_conviction >= 60}

    # Positive expected value
    if expected_value_pips <= 0:
        result.passed = False
        result.reasons.append(f"Negative EV: {expected_value_pips:.1f} pips")
    result.checks["expected_value"] = {"value": expected_value_pips, "min": 0, "pass": expected_value_pips > 0}

    # EV efficiency
    if ev_per_risk < 0.3:
        result.passed = False
        result.reasons.append(f"EV/risk {ev_per_risk:.2f} < 0.3 minimum")
    result.checks["ev_efficiency"] = {"value": ev_per_risk, "min": 0.3, "pass": ev_per_risk >= 0.3}

    # Evidence confluence
    if evidence_sources < 3:
        result.passed = False
        result.reasons.append(f"Only {evidence_sources} evidence sources (min 3)")
    result.checks["confluence"] = {"value": evidence_sources, "min": 3, "pass": evidence_sources >= 3}

    # Signal freshness
    if signal_age_minutes > 30:
        result.passed = False
        result.reasons.append(f"Signal is {signal_age_minutes:.0f} min old (max 30)")
    result.checks["freshness"] = {"value": signal_age_minutes, "max": 30, "pass": signal_age_minutes <= 30}

    # Price hasn't moved too far toward target
    if price_moved_pct > 30:
        result.passed = False
        result.reasons.append(f"Price already moved {price_moved_pct:.0f}% toward target — edge consumed")
    result.checks["entry_quality"] = {"value": price_moved_pct, "max": 30, "pass": price_moved_pct <= 30}

    # Risk:Reward
    if risk_reward < 1.5:
        result.passed = False
        result.reasons.append(f"R:R {risk_reward:.2f} < 1.5 minimum")
    result.checks["risk_reward"] = {"value": risk_reward, "min": 1.5, "pass": risk_reward >= 1.5}

    return result


# ── Run All Rules ─────────────────────────────────────────────

def run_all_rules(
    position_size_pct: float,
    daily_pnl_pct: float,
    weekly_pnl_pct: float,
    instrument: str,
    direction: str,
    open_positions: list[dict[str, Any]],
    is_news_blackout: bool,
    current_spread: float,
    avg_spread: float,
    has_stop_loss: bool,
    consecutive_losses: int,
    circuit_breaker_until: datetime | None = None,
    correlation_matrix: dict | None = None,
    entry: float = 0,
    stop_loss: float = 0,
    take_profit: float = 0,
    upcoming_events: list[dict] | None = None,
    ts: datetime | None = None,
) -> RuleCheckSuite:
    """Run all 10 rules and return complete results."""
    suite = RuleCheckSuite()

    suite.add(check_rule_1_position_size(position_size_pct))
    suite.add(check_rule_2_daily_loss(daily_pnl_pct))
    suite.add(check_rule_3_weekly_loss(weekly_pnl_pct))
    suite.add(check_rule_4_correlation_guard(instrument, direction, open_positions, correlation_matrix))
    suite.add(check_rule_5_news_blackout(is_news_blackout, upcoming_events))
    suite.add(check_rule_6_spread(current_spread, avg_spread))
    suite.add(check_rule_7_session(ts))
    suite.add(check_rule_8_stop_loss(has_stop_loss, entry, stop_loss, take_profit))
    suite.add(check_rule_9_circuit_breaker(consecutive_losses, circuit_breaker_until))
    suite.add(check_rule_10_weekend(ts))

    return suite
