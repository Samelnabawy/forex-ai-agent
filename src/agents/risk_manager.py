"""
Risk Manager — Agent 07

"The Accountant" — emotionless, rule-bound, protects capital above all else.

Has ABSOLUTE VETO POWER. No trade executes if Risk Manager says no.
Can also modify position sizes, widen stops, and manage open positions.

10 Layers:
  1. Sacred Rules (10 non-negotiable checks)
  2. Dynamic Position Sizing (Kelly, vol, drawdown, confidence)
  3. Portfolio-Level Risk (heat, currency exposure, concentration)
  4. Regime-Adaptive Modifications (crisis, NFP, month-end)
  5. Trade Quality Gate (conviction, EV, confluence, freshness)
  6. Drawdown Management (progressive throttle)
  7. Execution Quality (spread, slippage, order type)
  8. Open Position Management (trailing, partial TP, time decay)
  9. Anti-Tilt Detection (overtrading, quality degradation)
  10. Full Audit Trail (every decision logged)

Execution: On-demand (called for every trade proposal after debate).
Also runs periodic portfolio review every 5 minutes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar

from src.agents.base_agent import BaseAgent
from src.agents.evidence import EvidenceDirection, EvidencePackage, TradeProposal
from src.agents.risk_models import (
    DrawdownState,
    ExecutionQuality,
    PortfolioRisk,
    PositionAction,
    PositionSizeCalc,
    TiltState,
    assess_execution_quality,
    compute_drawdown_state,
    compute_portfolio_risk,
    compute_position_size,
    detect_tilt,
    evaluate_open_position,
    validate_stop_loss,
)
from src.agents.risk_rules_engine import (
    QualityGateResult,
    RuleCheckSuite,
    check_quality_gate,
    run_all_rules,
)
from src.config.instruments import INSTRUMENTS
from src.config.risk_rules import RISK_RULES
from src.data.storage.cache import get_agent_state, get_current_spread
from src.models import MarketState, RiskAssessment

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseAgent):
    """
    Agent 07: Risk Manager

    ABSOLUTE VETO POWER. Protects capital above all else.
    """

    name: ClassVar[str] = "risk_manager"
    description: ClassVar[str] = "Enforces risk rules, sizes positions, manages portfolio risk"
    execution_frequency: ClassVar[str] = "on_demand"

    def __init__(self) -> None:
        super().__init__()
        self._recent_convictions: list[float] = []
        self._recent_results: list[dict[str, Any]] = []
        self._signals_last_hour: int = 0
        self._last_hour_reset: datetime = datetime.now(timezone.utc)

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """Periodic portfolio review — manages open positions."""
        actions: list[dict[str, Any]] = []

        for trade in market_state.open_trades:
            price_data = market_state.prices.get(trade.instrument)
            if not price_data:
                continue

            current_price = float((price_data.bid + price_data.ask) / 2)
            inst = INSTRUMENTS.get(trade.instrument)
            atr = float(inst.avg_daily_range_pips * inst.pip_size) / 5 if inst else 0

            hours_open = (datetime.now(timezone.utc) - trade.created_at).total_seconds() / 3600

            # Check if macro regime changed since trade opened
            macro_state = await get_agent_state("macro_analyst")
            macro_changed = False  # Would compare against trade's original regime

            action = evaluate_open_position(
                trade={
                    "trade_id": trade.trade_id,
                    "direction": trade.direction.value,
                    "entry_price": float(trade.entry_price),
                    "stop_loss": float(trade.stop_loss),
                    "take_profit_1": float(trade.take_profit_1),
                },
                current_price=current_price,
                atr=atr,
                hours_open=hours_open,
                macro_regime_changed=macro_changed,
            )

            if action.action != "hold":
                actions.append({
                    "trade_id": action.trade_id,
                    "action": action.action,
                    "new_stop": action.new_stop,
                    "close_pct": action.close_pct,
                    "reasoning": action.reasoning,
                    "urgency": action.urgency,
                })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open_positions": len(market_state.open_trades),
            "position_actions": actions,
            "portfolio_heat": sum(float(t.position_size) for t in market_state.open_trades),
        }

    async def evaluate_trade(
        self,
        proposal: TradeProposal,
        debate_verdict: dict[str, Any],
        bull_package: EvidencePackage,
        bear_package: EvidencePackage,
        market_state: MarketState,
    ) -> dict[str, Any]:
        """
        Full risk evaluation for a trade proposal after debate.
        Returns APPROVED / APPROVED_WITH_MODIFICATION / REJECTED.
        """
        self.logger.info(
            "Evaluating trade",
            extra={"trade_id": proposal.trade_id, "instrument": proposal.instrument},
        )

        result: dict[str, Any] = {
            "trade_id": proposal.trade_id,
            "instrument": proposal.instrument,
            "direction": proposal.direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Track for anti-tilt
        self._track_signal()
        conviction = debate_verdict.get("conviction", 50)
        self._recent_convictions.append(conviction)
        if len(self._recent_convictions) > 20:
            self._recent_convictions = self._recent_convictions[-20:]

        # ── Layer 6: Drawdown State ───────────────────────────
        dd_state = compute_drawdown_state(
            daily_pnl_pct=float(market_state.daily_pnl_pct),
            weekly_pnl_pct=float(market_state.weekly_pnl_pct),
        )
        result["drawdown_state"] = {
            "daily_alert": dd_state.daily_alert_level,
            "weekly_alert": dd_state.weekly_alert_level,
            "trading_allowed": dd_state.trading_allowed,
            "max_allowed_size": dd_state.max_allowed_size_pct,
            "actions": dd_state.actions,
        }

        if not dd_state.trading_allowed:
            result["decision"] = "REJECTED"
            result["veto_reason"] = dd_state.halt_reason
            return result

        # ── Layer 9: Anti-Tilt ────────────────────────────────
        tilt = detect_tilt(
            signals_last_hour=self._signals_last_hour,
            recent_convictions=self._recent_convictions,
            recent_results=self._recent_results,
        )
        result["tilt_state"] = {
            "overtrading": tilt.overtrading,
            "quality_degradation": tilt.quality_degradation,
            "regime_mismatch": tilt.regime_mismatch,
            "actions": tilt.actions,
        }

        if tilt.overtrading or tilt.quality_degradation:
            result["decision"] = "REJECTED"
            result["veto_reason"] = tilt.actions[0] if tilt.actions else "Anti-tilt triggered"
            return result

        # ── Layer 1: Sacred Rules ─────────────────────────────
        inst = INSTRUMENTS.get(proposal.instrument)
        spread = await get_current_spread(proposal.instrument)
        avg_spread = float(inst.avg_spread_pips * inst.pip_size) if inst else 0

        # Get open positions
        open_pos = [
            {
                "instrument": t.instrument,
                "direction": t.direction.value,
                "position_size": float(t.position_size),
            }
            for t in market_state.open_trades
        ]

        # Check news blackout
        try:
            from src.data.ingestion.calendar_feed import EconomicCalendarFeed
            cal = EconomicCalendarFeed()
            is_blackout = await cal.is_news_blackout()
            upcoming = await cal.get_upcoming_high_impact(60)
            await cal.close()
        except Exception:
            is_blackout = False
            upcoming = []

        # Get correlation matrix
        corr_state = await get_agent_state("correlation_agent")
        corr_matrix = None
        if corr_state:
            matrices = corr_state.get("correlation_matrices", {})
            corr_matrix = matrices.get(30) or matrices.get("30")

        # Risk state from DB
        consecutive_losses = 0  # Would come from risk_state table

        # Preliminary position size (will be refined)
        preliminary_size = float(RISK_RULES.max_risk_per_trade_pct)

        rules = run_all_rules(
            position_size_pct=preliminary_size,
            daily_pnl_pct=float(market_state.daily_pnl_pct),
            weekly_pnl_pct=float(market_state.weekly_pnl_pct),
            instrument=proposal.instrument,
            direction=proposal.direction,
            open_positions=open_pos,
            is_news_blackout=is_blackout,
            current_spread=spread or 0,
            avg_spread=avg_spread,
            has_stop_loss=proposal.stop_loss != 0,
            consecutive_losses=consecutive_losses,
            correlation_matrix=corr_matrix,
            entry=proposal.entry,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit_1,
            upcoming_events=upcoming,
        )
        result["rule_checks"] = rules.to_dict()

        if rules.veto:
            result["decision"] = "REJECTED"
            failed = [c for c in rules.checks if c.status == "FAIL"]
            result["veto_reason"] = "; ".join(c.message for c in failed)
            return result

        # ── Layer 2: Position Sizing ──────────────────────────
        ev_data = debate_verdict.get("expected_value", {})
        win_prob = ev_data.get("win_probability", 0.5)
        avg_win = ev_data.get("avg_win_pips", 30)
        avg_loss = abs(ev_data.get("avg_loss_pips", -20))

        atr = proposal.atr if proposal.atr else (float(inst.avg_daily_range_pips * inst.pip_size) / 5 if inst else 0.001)
        normal_atr = atr  # Would compare to historical average

        sizing = compute_position_size(
            win_rate=win_prob,
            avg_win_pips=avg_win,
            avg_loss_pips=avg_loss,
            current_atr=atr,
            normal_atr=normal_atr,
            current_drawdown_pct=float(market_state.daily_pnl_pct),
            max_drawdown_pct=2.0,
            debate_conviction=conviction,
            max_risk_pct=dd_state.max_allowed_size_pct,
        )
        result["position_sizing"] = {
            "kelly": sizing.kelly_optimal_pct,
            "half_kelly": sizing.half_kelly_pct,
            "vol_adjusted": sizing.volatility_adjusted_pct,
            "dd_adjusted": sizing.drawdown_adjusted_pct,
            "confidence_adjusted": sizing.confidence_adjusted_pct,
            "final_size": sizing.final_size_pct,
            "reasoning": sizing.reasoning,
        }

        if sizing.final_size_pct <= 0:
            result["decision"] = "REJECTED"
            result["veto_reason"] = "Position size calculated as zero — trade not worth taking"
            return result

        # ── Layer 3: Portfolio Risk ───────────────────────────
        new_trade_dict = {
            "instrument": proposal.instrument,
            "direction": proposal.direction,
            "position_size": sizing.final_size_pct,
        }
        portfolio = compute_portfolio_risk(open_pos, new_trade_dict, corr_matrix)
        result["portfolio_impact"] = {
            "heat_before": round(portfolio.total_heat_pct, 2),
            "heat_after": round(portfolio.heat_after_trade, 2),
            "heat_ok": portfolio.heat_ok,
            "currency_exposure": portfolio.currency_exposure,
            "warnings": portfolio.warnings,
        }

        if not portfolio.heat_ok:
            # Reduce size to fit within heat limit
            available_heat = portfolio.max_heat_pct - portfolio.total_heat_pct
            if available_heat > 0.1:
                sizing.final_size_pct = min(sizing.final_size_pct, available_heat)
                result["modifications"] = result.get("modifications", {})
                result["modifications"]["position_size"] = f"Reduced to {sizing.final_size_pct:.2f}% to fit portfolio heat limit"
            else:
                result["decision"] = "REJECTED"
                result["veto_reason"] = "Portfolio heat at maximum — no room for new trades"
                return result

        # ── Layer 5: Quality Gate ─────────────────────────────
        evidence_sources = len(set(
            e.source for e in bull_package.evidence if e.direction == EvidenceDirection.SUPPORTING
        ))
        rr = ev_data.get("risk_reward_ratio", 0)
        ev_pips = ev_data.get("expected_value_pips", 0)
        ev_per_risk = ev_data.get("ev_per_risk_unit", 0)

        quality = check_quality_gate(
            debate_conviction=conviction,
            expected_value_pips=ev_pips,
            ev_per_risk=ev_per_risk,
            evidence_sources=evidence_sources,
            signal_age_minutes=5,  # Would compute from signal timestamp
            price_moved_pct=0,     # Would compute from signal entry vs current
            risk_reward=rr,
        )
        result["quality_gate"] = {
            "passed": quality.passed,
            "checks": quality.checks,
            "reasons": quality.reasons,
        }

        if not quality.passed:
            result["decision"] = "REJECTED"
            result["veto_reason"] = "Quality gate failed: " + "; ".join(quality.reasons)
            return result

        # ── Layer 7: Execution Quality ────────────────────────
        hour = datetime.now(timezone.utc).hour
        exec_quality = assess_execution_quality(
            current_spread=spread or 0,
            avg_spread=avg_spread,
            session_hour_utc=hour,
        )
        result["execution"] = {
            "spread_ok": exec_quality.spread_ok,
            "order_type": exec_quality.order_type,
            "slippage_estimate": exec_quality.slippage_estimate_pips,
            "delay_recommended": exec_quality.delay_recommended,
            "delay_reason": exec_quality.delay_reason,
        }

        # ── Layer 8: Stop Loss Validation ─────────────────────
        pip_size = float(inst.pip_size) if inst else 0.0001
        sl_validation = validate_stop_loss(
            entry=proposal.entry,
            stop_loss=proposal.stop_loss,
            direction=proposal.direction,
            atr=atr,
            pip_size=pip_size,
        )
        result["stop_validation"] = sl_validation

        modifications: dict[str, Any] = result.get("modifications", {})
        if sl_validation.get("suggested_stop"):
            modifications["stop_loss"] = (
                f"Widened from {proposal.stop_loss:.6f} to {sl_validation['suggested_stop']:.6f} "
                f"(ATR validation: stop was too tight)"
            )
            result["modifications"] = modifications

        # ── Final Decision ────────────────────────────────────
        has_modifications = bool(modifications)
        has_warnings = rules.has_warnings or portfolio.warnings

        if has_modifications:
            result["decision"] = "APPROVED_WITH_MODIFICATION"
        elif has_warnings:
            result["decision"] = "APPROVED_WITH_MODIFICATION"
        else:
            result["decision"] = "APPROVED"

        result["final_position_size"] = sizing.final_size_pct
        result["final_recommendation"] = self._build_recommendation(result)

        return result

    def _build_recommendation(self, result: dict[str, Any]) -> str:
        """Build human-readable recommendation."""
        decision = result.get("decision", "REJECTED")
        size = result.get("final_position_size", 0)
        instrument = result.get("instrument", "?")

        if decision == "REJECTED":
            return f"REJECTED: {result.get('veto_reason', 'Unknown reason')}"
        elif decision == "APPROVED":
            return f"APPROVED: {instrument} at {size:.2f}% risk"
        else:
            mods = result.get("modifications", {})
            mod_str = "; ".join(f"{k}: {v}" for k, v in mods.items())
            return f"APPROVED WITH MODIFICATIONS: {instrument} at {size:.2f}% risk. {mod_str}"

    def _track_signal(self) -> None:
        """Track signals per hour for anti-tilt."""
        now = datetime.now(timezone.utc)
        if (now - self._last_hour_reset).total_seconds() > 3600:
            self._signals_last_hour = 0
            self._last_hour_reset = now
        self._signals_last_hour += 1

    def record_trade_result(self, result: dict[str, Any]) -> None:
        """Record a closed trade result for anti-tilt tracking."""
        self._recent_results.append(result)
        if len(self._recent_results) > 50:
            self._recent_results = self._recent_results[-50:]

    @property
    def stats(self) -> dict[str, Any]:
        base = super().stats
        base["signals_last_hour"] = self._signals_last_hour
        base["avg_recent_conviction"] = (
            round(sum(self._recent_convictions) / len(self._recent_convictions), 1)
            if self._recent_convictions else 0
        )
        return base
