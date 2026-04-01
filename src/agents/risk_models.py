"""
Risk Computation Models.

Pure computation — no database, no side effects.
Used by Risk Manager Agent for all quantitative risk decisions.

Models:
  - Dynamic Position Sizing (Kelly, volatility, drawdown, confidence, combined)
  - Portfolio-Level Risk (heat, currency exposure, effective risk)
  - Drawdown Management Protocol (progressive throttle, not cliff)
  - Execution Quality (spread analysis, slippage, order type)
  - Anti-Tilt Detection (overtrading, quality degradation, correlated losses)
  - Open Position Management (trailing, partial TP, time decay, thesis validation)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Layer 2: Dynamic Position Sizing ──────────────────────────

@dataclass
class PositionSizeCalc:
    """Complete position sizing breakdown."""
    kelly_optimal_pct: float = 0.0
    half_kelly_pct: float = 0.0
    volatility_adjusted_pct: float = 0.0
    drawdown_adjusted_pct: float = 0.0
    confidence_adjusted_pct: float = 0.0
    final_size_pct: float = 0.0
    cap_applied: bool = False  # Hit the 1% maximum
    reasoning: list[str] = field(default_factory=list)


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Kelly Criterion: f = (p × b - q) / b
    where p = win probability, b = win/loss ratio, q = 1 - p
    Returns the optimal fraction of capital to risk.
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    b = abs(avg_win / avg_loss)
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b

    # Kelly can be negative (don't trade) or very large (unrealistic)
    return max(0.0, min(kelly, 0.25))  # Cap at 25% before half-kelly


def compute_position_size(
    win_rate: float,
    avg_win_pips: float,
    avg_loss_pips: float,
    current_atr: float,
    normal_atr: float,
    current_drawdown_pct: float,
    max_drawdown_pct: float,
    debate_conviction: float,  # 0-100
    max_risk_pct: float = 1.0,
) -> PositionSizeCalc:
    """
    Combined position sizing model.
    Takes the most conservative of all models, scales by confidence, caps at max.
    """
    result = PositionSizeCalc()

    # 1. Kelly Criterion
    kelly = compute_kelly_fraction(win_rate, avg_win_pips, abs(avg_loss_pips))
    result.kelly_optimal_pct = round(kelly * 100, 3)
    result.half_kelly_pct = round(kelly * 50, 3)  # Half-Kelly = smoother
    result.reasoning.append(f"Kelly={result.kelly_optimal_pct:.2f}%, Half-Kelly={result.half_kelly_pct:.2f}%")

    # 2. Volatility-adjusted
    if current_atr > 0 and normal_atr > 0:
        vol_ratio = normal_atr / current_atr
        vol_adjusted = max_risk_pct * min(vol_ratio, 1.5)  # Don't over-size in low vol
        result.volatility_adjusted_pct = round(vol_adjusted, 3)
        result.reasoning.append(f"Vol-adjusted={vol_adjusted:.2f}% (ATR ratio {vol_ratio:.2f})")
    else:
        result.volatility_adjusted_pct = max_risk_pct * 0.5
        result.reasoning.append("Vol-adjusted=0.50% (no ATR data, using conservative default)")

    # 3. Drawdown-adjusted
    if max_drawdown_pct > 0 and current_drawdown_pct < 0:
        dd_ratio = 1 - (abs(current_drawdown_pct) / max_drawdown_pct)
        dd_adjusted = max_risk_pct * max(dd_ratio, 0.1)  # Floor at 10% of max
        result.drawdown_adjusted_pct = round(dd_adjusted, 3)
        result.reasoning.append(f"DD-adjusted={dd_adjusted:.2f}% (drawdown {current_drawdown_pct:.1f}%)")
    else:
        result.drawdown_adjusted_pct = max_risk_pct
        result.reasoning.append(f"DD-adjusted={max_risk_pct:.2f}% (no drawdown)")

    # 4. Confidence-adjusted
    confidence_factor = debate_conviction / 100
    confidence_adjusted = max_risk_pct * confidence_factor
    result.confidence_adjusted_pct = round(confidence_adjusted, 3)
    result.reasoning.append(f"Confidence-adjusted={confidence_adjusted:.2f}% (conviction {debate_conviction:.0f}/100)")

    # 5. Combined: take the MINIMUM (most conservative)
    candidates = [
        result.half_kelly_pct,
        result.volatility_adjusted_pct,
        result.drawdown_adjusted_pct,
        result.confidence_adjusted_pct,
    ]
    # Filter out zeros and negatives
    valid = [c for c in candidates if c > 0]
    if valid:
        combined = min(valid)
    else:
        combined = 0.0

    # Cap at maximum
    if combined > max_risk_pct:
        combined = max_risk_pct
        result.cap_applied = True
        result.reasoning.append(f"Capped at {max_risk_pct:.2f}% (max risk rule)")

    # Floor: don't take trades smaller than 0.1%
    if 0 < combined < 0.1:
        combined = 0.0
        result.reasoning.append("Size below 0.1% minimum — not worth the commission")

    result.final_size_pct = round(combined, 3)
    result.reasoning.append(f"FINAL SIZE: {result.final_size_pct:.2f}%")

    return result


# ── Layer 3: Portfolio-Level Risk ─────────────────────────────

@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment."""
    total_heat_pct: float = 0.0  # Sum of all open risk
    heat_after_trade: float = 0.0
    max_heat_pct: float = 3.0
    heat_ok: bool = True
    currency_exposure: dict[str, float] = field(default_factory=dict)
    directional_bias: dict[str, float] = field(default_factory=dict)
    max_currency_exposure_pct: float = 0.0
    sector_concentration: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def compute_portfolio_risk(
    open_positions: list[dict[str, Any]],
    new_trade: dict[str, Any] | None = None,
    correlation_matrix: dict[str, dict[str, float]] | None = None,
    max_heat: float = 3.0,
) -> PortfolioRisk:
    """Compute portfolio-level risk including the proposed new trade."""
    risk = PortfolioRisk(max_heat_pct=max_heat)

    # Current heat
    for pos in open_positions:
        risk.total_heat_pct += pos.get("position_size", 0)

    # Heat after adding new trade
    new_size = new_trade.get("position_size", 0) if new_trade else 0
    risk.heat_after_trade = risk.total_heat_pct + new_size
    risk.heat_ok = risk.heat_after_trade <= max_heat

    if not risk.heat_ok:
        risk.warnings.append(
            f"Portfolio heat would be {risk.heat_after_trade:.2f}% (max {max_heat}%)"
        )

    # Currency exposure aggregation
    all_positions = list(open_positions)
    if new_trade:
        all_positions.append(new_trade)

    currency_net: dict[str, float] = {}
    for pos in all_positions:
        instrument = pos.get("instrument", "")
        direction = pos.get("direction", "LONG")
        size = pos.get("position_size", 0)
        parts = instrument.split("/")
        if len(parts) != 2:
            continue

        base, quote = parts
        mult = 1.0 if direction == "LONG" else -1.0
        currency_net[base] = currency_net.get(base, 0) + size * mult
        currency_net[quote] = currency_net.get(quote, 0) - size * mult

    risk.currency_exposure = {k: round(v, 3) for k, v in currency_net.items()}

    # Check for dangerous concentration
    if currency_net:
        max_exposure = max(abs(v) for v in currency_net.values())
        risk.max_currency_exposure_pct = round(max_exposure, 3)
        max_currency = max(currency_net, key=lambda k: abs(currency_net[k]))

        if max_exposure > 2.0:
            risk.warnings.append(
                f"HIGH {max_currency} concentration: {max_exposure:.2f}% net exposure"
            )
        elif max_exposure > 1.5:
            risk.warnings.append(
                f"Elevated {max_currency} exposure: {max_exposure:.2f}%"
            )

    # Directional bias
    usd_exposure = currency_net.get("USD", 0)
    if abs(usd_exposure) > 1.5:
        risk.warnings.append(
            f"Strong USD directional bias: {usd_exposure:+.2f}% — vulnerable to surprise USD events"
        )

    # Sector concentration
    from src.config.instruments import INSTRUMENTS, InstrumentCategory
    sector_risk: dict[str, float] = {}
    for pos in all_positions:
        inst = INSTRUMENTS.get(pos.get("instrument", ""))
        if inst:
            cat = inst.category.value
            sector_risk[cat] = sector_risk.get(cat, 0) + pos.get("position_size", 0)

    risk.sector_concentration = sector_risk
    total_risk = sum(sector_risk.values())
    for sector, amount in sector_risk.items():
        if total_risk > 0 and amount / total_risk > 0.6:
            risk.warnings.append(f"Sector concentration: {amount/total_risk:.0%} in {sector}")

    return risk


# ── Layer 6: Drawdown Management Protocol ────────────────────

@dataclass
class DrawdownState:
    """Current drawdown state with progressive throttle levels."""
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    monthly_pnl_pct: float = 0.0
    daily_alert_level: str = "green"    # green/yellow/orange/red/halt
    weekly_alert_level: str = "green"
    monthly_alert_level: str = "green"
    max_allowed_size_pct: float = 1.0   # Reduced as drawdown deepens
    min_conviction_required: int = 60   # Raised as drawdown deepens
    trading_allowed: bool = True
    halt_reason: str = ""
    actions: list[str] = field(default_factory=list)


def compute_drawdown_state(
    daily_pnl_pct: float,
    weekly_pnl_pct: float,
    monthly_pnl_pct: float = 0.0,
) -> DrawdownState:
    """
    Progressive drawdown management — NOT a binary cliff.
    Throttles position sizes and raises conviction thresholds as losses deepen.
    """
    state = DrawdownState(
        daily_pnl_pct=daily_pnl_pct,
        weekly_pnl_pct=weekly_pnl_pct,
        monthly_pnl_pct=monthly_pnl_pct,
    )

    max_size = 1.0
    min_conviction = 60

    # Daily protocol
    if daily_pnl_pct <= -2.0:
        state.daily_alert_level = "halt"
        state.trading_allowed = False
        state.halt_reason = f"Daily loss limit breached: {daily_pnl_pct:.2f}%"
        state.actions.append("ALL TRADING HALTED for the day")
        return state
    elif daily_pnl_pct <= -1.5:
        state.daily_alert_level = "red"
        max_size = min(max_size, 0.5)
        min_conviction = max(min_conviction, 75)
        state.actions.append(f"Daily -1.5%: size capped at 0.5%, conviction ≥75")
    elif daily_pnl_pct <= -1.0:
        state.daily_alert_level = "orange"
        max_size = min(max_size, 0.75)
        min_conviction = max(min_conviction, 70)
        state.actions.append(f"Daily -1.0%: size capped at 0.75%, conviction ≥70")
    elif daily_pnl_pct <= -0.5:
        state.daily_alert_level = "yellow"
        state.actions.append("Daily -0.5%: yellow alert, monitoring")

    # Weekly protocol
    if weekly_pnl_pct <= -5.0:
        state.weekly_alert_level = "halt"
        state.trading_allowed = False
        state.halt_reason = f"Weekly loss limit breached: {weekly_pnl_pct:.2f}%"
        state.actions.append("ALL TRADING HALTED until Monday")
        return state
    elif weekly_pnl_pct <= -4.0:
        state.weekly_alert_level = "red"
        max_size = min(max_size, 0.25)
        min_conviction = max(min_conviction, 80)
        state.actions.append("Weekly -4%: size capped at 0.25%, high conviction only")
    elif weekly_pnl_pct <= -3.0:
        state.weekly_alert_level = "orange"
        max_size = min(max_size, 0.5)
        min_conviction = max(min_conviction, 75)
        state.actions.append("Weekly -3%: conservation mode, size capped at 0.5%")
    elif weekly_pnl_pct <= -2.0:
        state.weekly_alert_level = "yellow"
        state.actions.append("Weekly -2%: yellow alert")

    # Monthly protocol
    if monthly_pnl_pct <= -8.0:
        state.monthly_alert_level = "halt"
        state.trading_allowed = False
        state.halt_reason = f"Monthly -8%: SYSTEM SHUTDOWN. Manual review required."
        state.actions.append("CRITICAL: System shutdown. Manual review required.")
        return state
    elif monthly_pnl_pct <= -5.0:
        state.monthly_alert_level = "red"
        max_size = min(max_size, 0.5)
        state.actions.append("Monthly -5%: all sizes halved for rest of month")

    state.max_allowed_size_pct = max_size
    state.min_conviction_required = min_conviction

    return state


# ── Layer 7: Execution Quality ────────────────────────────────

@dataclass
class ExecutionQuality:
    """Execution quality assessment."""
    spread_ok: bool = True
    spread_current: float = 0.0
    spread_avg: float = 0.0
    spread_ratio: float = 0.0
    spread_widening: bool = False
    session_liquidity: str = "normal"  # "thin", "normal", "peak"
    slippage_estimate_pips: float = 0.0
    order_type: str = "market"  # "market", "limit"
    delay_recommended: bool = False
    delay_reason: str = ""


def assess_execution_quality(
    current_spread: float,
    avg_spread: float,
    max_spread_multiplier: float = 2.0,
    previous_spread: float | None = None,
    session_hour_utc: int = 14,
) -> ExecutionQuality:
    """Assess execution conditions and recommend order type."""
    result = ExecutionQuality(
        spread_current=current_spread,
        spread_avg=avg_spread,
    )

    if avg_spread > 0:
        result.spread_ratio = round(current_spread / avg_spread, 2)
        result.spread_ok = result.spread_ratio <= max_spread_multiplier

    # Spread direction
    if previous_spread is not None:
        result.spread_widening = current_spread > previous_spread * 1.1

    # Session liquidity
    if 13 <= session_hour_utc <= 16:
        result.session_liquidity = "peak"  # London/NY overlap
    elif 8 <= session_hour_utc <= 12 or 16 < session_hour_utc <= 20:
        result.session_liquidity = "normal"
    else:
        result.session_liquidity = "thin"

    # Delay recommendation
    if result.spread_ratio > 1.5:
        result.delay_recommended = True
        result.delay_reason = f"Spread elevated ({result.spread_ratio:.1f}x avg) — wait 2 min"

    if result.spread_widening:
        result.delay_recommended = True
        result.delay_reason = "Spread actively widening — wait for stabilization"

    if result.session_liquidity == "thin":
        result.delay_recommended = True
        result.delay_reason = "Thin liquidity — consider limit order"

    # Slippage estimate
    if result.session_liquidity == "peak":
        result.slippage_estimate_pips = 0.1
    elif result.session_liquidity == "normal":
        result.slippage_estimate_pips = 0.3
    else:
        result.slippage_estimate_pips = 0.8

    # Order type
    if result.spread_ok and not result.spread_widening and result.session_liquidity != "thin":
        result.order_type = "market"
    else:
        result.order_type = "limit"

    return result


# ── Layer 8: Open Position Management ─────────────────────────

@dataclass
class PositionAction:
    """Recommended action for an open position."""
    trade_id: str
    action: str  # "hold", "trail_stop", "partial_close", "close", "emergency_close"
    new_stop: float | None = None
    close_pct: float = 0.0  # 0-100, percentage to close
    reasoning: str = ""
    urgency: str = "normal"  # "normal", "high", "critical"


def evaluate_open_position(
    trade: dict[str, Any],
    current_price: float,
    atr: float,
    hours_open: float,
    macro_regime_changed: bool = False,
    correlation_broken: bool = False,
) -> PositionAction:
    """Evaluate an open position for management actions."""
    trade_id = trade.get("trade_id", "")
    direction = trade.get("direction", "LONG")
    entry = trade.get("entry_price", 0)
    stop = trade.get("stop_loss", 0)
    tp1 = trade.get("take_profit_1", 0)

    if entry == 0 or atr == 0:
        return PositionAction(trade_id=trade_id, action="hold", reasoning="Insufficient data")

    is_long = direction == "LONG"
    pnl_pips = (current_price - entry) if is_long else (entry - current_price)
    risk_pips = abs(entry - stop)

    # Thesis invalidation
    if macro_regime_changed or correlation_broken:
        return PositionAction(
            trade_id=trade_id,
            action="close",
            reasoning="Thesis invalidated: " +
                      ("macro regime changed" if macro_regime_changed else "correlation broken"),
            urgency="high",
        )

    # Time decay: > 8 hours with < 0.3 ATR move = dead trade
    if hours_open > 8 and abs(pnl_pips) < atr * 0.3:
        return PositionAction(
            trade_id=trade_id,
            action="close",
            reasoning=f"Time decay: {hours_open:.0f}h open, only {pnl_pips:.1f} pips — trade is dead",
            urgency="normal",
        )

    # Adverse: -50% of stop and momentum flipping
    if pnl_pips < -(risk_pips * 0.5):
        return PositionAction(
            trade_id=trade_id,
            action="close",
            reasoning=f"Adverse move: {pnl_pips:.1f} pips (50% of stop consumed). Consider early exit.",
            urgency="high",
        )

    # Trailing stop at 1 ATR profit
    if pnl_pips >= atr:
        new_stop_level = entry if pnl_pips < 2 * atr else (
            current_price - atr if is_long else current_price + atr
        )
        return PositionAction(
            trade_id=trade_id,
            action="trail_stop",
            new_stop=round(new_stop_level, 6),
            reasoning=f"Trade {pnl_pips:.1f} pips in profit (≥1 ATR). Trail stop to {new_stop_level:.6f}",
        )

    # Partial close at TP1
    if tp1 > 0:
        dist_to_tp1 = abs(tp1 - current_price)
        total_tp_dist = abs(tp1 - entry)
        if total_tp_dist > 0 and dist_to_tp1 / total_tp_dist < 0.1:
            return PositionAction(
                trade_id=trade_id,
                action="partial_close",
                close_pct=50.0,
                reasoning="Price within 10% of TP1 — close 50%, trail remainder",
            )

    return PositionAction(trade_id=trade_id, action="hold", reasoning="Position within normal parameters")


# ── Layer 9: Anti-Tilt Detection ──────────────────────────────

@dataclass
class TiltState:
    """Anti-tilt monitoring state."""
    overtrading: bool = False
    quality_degradation: bool = False
    regime_mismatch: bool = False
    correlated_loss_cluster: bool = False
    signals_last_hour: int = 0
    avg_recent_conviction: float = 70.0
    recent_win_rate: float = 0.55
    actions: list[str] = field(default_factory=list)


def detect_tilt(
    signals_last_hour: int,
    normal_signals_per_hour: int = 2,
    recent_convictions: list[float] | None = None,
    recent_results: list[dict[str, Any]] | None = None,
) -> TiltState:
    """Detect systematic behavior problems before they cause damage."""
    state = TiltState(signals_last_hour=signals_last_hour)

    # Overtrading
    if signals_last_hour > normal_signals_per_hour * 3:
        state.overtrading = True
        state.actions.append(
            f"OVERTRADING: {signals_last_hour} signals in 1 hour (normal: {normal_signals_per_hour}). "
            "Pause signal generation for 30 minutes."
        )

    # Quality degradation
    if recent_convictions:
        avg = sum(recent_convictions) / len(recent_convictions)
        state.avg_recent_conviction = round(avg, 1)
        if avg < 55 and len(recent_convictions) >= 5:
            state.quality_degradation = True
            state.actions.append(
                f"QUALITY DEGRADATION: Average conviction {avg:.0f}/100 (last {len(recent_convictions)} trades). "
                "Block trades below conviction 65."
            )

    # Regime mismatch (win rate collapse)
    if recent_results and len(recent_results) >= 15:
        wins = sum(1 for r in recent_results if r.get("outcome") == "win")
        state.recent_win_rate = round(wins / len(recent_results), 2)
        if state.recent_win_rate < 0.30:
            state.regime_mismatch = True
            state.actions.append(
                f"REGIME MISMATCH: Win rate dropped to {state.recent_win_rate:.0%} "
                f"over last {len(recent_results)} trades. Reduce size to 0.25%, review strategy."
            )

    # Correlated loss cluster
    if recent_results:
        recent_losses = [r for r in recent_results[-5:] if r.get("outcome") == "loss"]
        if len(recent_losses) >= 3:
            # Check if same direction/cluster
            directions = [r.get("direction", "") for r in recent_losses]
            if len(set(directions)) == 1:
                state.correlated_loss_cluster = True
                state.actions.append(
                    f"CORRELATED LOSSES: {len(recent_losses)} consecutive losses, "
                    f"all {directions[0]}. Block this direction for 4 hours."
                )

    return state


# ── Stop Loss Validation ──────────────────────────────────────

def validate_stop_loss(
    entry: float,
    stop_loss: float,
    direction: str,
    atr: float,
    nearest_sr: float | None = None,
    pip_size: float = 0.0001,
) -> dict[str, Any]:
    """
    Validate that a stop loss is at a LOGICAL level.
    A stop that's too tight = certain stopout.
    A stop beyond S/R = wasted risk.
    """
    risk_pips = abs(entry - stop_loss) / pip_size
    atr_pips = atr / pip_size if pip_size > 0 else 0

    result: dict[str, Any] = {
        "valid": True,
        "risk_pips": round(risk_pips, 1),
        "atr_pips": round(atr_pips, 1),
        "atr_ratio": round(risk_pips / atr_pips, 2) if atr_pips > 0 else 0,
        "warnings": [],
        "suggested_stop": None,
    }

    # Too tight: stop < 0.5 ATR from entry
    if atr_pips > 0 and risk_pips < atr_pips * 0.5:
        result["warnings"].append(
            f"Stop too tight: {risk_pips:.0f} pips < 0.5 ATR ({atr_pips*0.5:.0f} pips). "
            "High probability of normal-volatility stopout."
        )
        # Suggest wider stop
        buffer = atr * 1.5
        if direction == "LONG":
            result["suggested_stop"] = round(entry - buffer, 6)
        else:
            result["suggested_stop"] = round(entry + buffer, 6)

    # Too wide: stop > 3 ATR from entry
    if atr_pips > 0 and risk_pips > atr_pips * 3:
        result["warnings"].append(
            f"Stop very wide: {risk_pips:.0f} pips > 3 ATR ({atr_pips*3:.0f} pips). "
            "Position size will be very small."
        )

    # Check against S/R levels
    if nearest_sr is not None:
        is_long = direction == "LONG"
        sr_dist = abs(nearest_sr - entry)

        if is_long and stop_loss > nearest_sr:
            result["warnings"].append(
                f"Stop above support at {nearest_sr:.6f} — illogical placement. "
                "Place stop BELOW support."
            )
        elif not is_long and stop_loss < nearest_sr:
            result["warnings"].append(
                f"Stop below resistance at {nearest_sr:.6f} — illogical placement. "
                "Place stop ABOVE resistance."
            )

    if result["warnings"]:
        result["valid"] = len([w for w in result["warnings"] if "too tight" in w.lower()]) == 0

    return result
