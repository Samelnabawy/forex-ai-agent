"""
Portfolio Manager — Agent 08

"The Decision Maker" — calm, decisive, synthesizes conflicting
information into clear action. The CEO of the trading desk.

Responsibilities:
  1. Signal triage (noise → exceptional, controls LLM spend)
  2. Decision pipeline orchestration (context → debate → risk → execute)
  3. Decision matrix (weighted agent votes, not simple majority)
  4. Capital allocation (best opportunities get most capital)
  5. Active signal management (queue, dedup, expiry, priority)
  6. Trade execution pipeline (Telegram + trade logger)
  7. Performance tracking and attribution
  8. System health monitoring
  9. Strategy adaptation based on results
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, ClassVar

from src.agents.base_agent import BaseAgent
from src.agents.evidence import EvidenceDirection, TradeProposal
from src.config.instruments import INSTRUMENTS
from src.config.risk_rules import RISK_RULES
from src.data.storage.cache import dequeue_signal, get_agent_state, get_queue_length
from src.models import MarketState, TradeDecision, Direction

logger = logging.getLogger(__name__)


# ── Signal Triage ─────────────────────────────────────────────

class TriageLevel:
    NOISE = "noise"           # < 50 confluence → drop
    WEAK = "weak"             # 50-60 → risk check only, skip debate
    STANDARD = "standard"     # 60-75 → full debate
    STRONG = "strong"         # 75-90 → full debate + priority
    EXCEPTIONAL = "exceptional"  # 90+ → full debate + extra queries + alert


def triage_signal(signal: dict[str, Any]) -> str:
    """Classify a signal into a triage level."""
    confluence = signal.get("confluence_score", 0)
    confidence = signal.get("confidence", 0)
    score = confluence if confluence > 1 else confidence * 100

    if score >= 90:
        return TriageLevel.EXCEPTIONAL
    elif score >= 75:
        return TriageLevel.STRONG
    elif score >= 60:
        return TriageLevel.STANDARD
    elif score >= 50:
        return TriageLevel.WEAK
    else:
        return TriageLevel.NOISE


# ── Decision Matrix ───────────────────────────────────────────

class AgentVoteResult:
    """Weighted vote from a single agent."""

    def __init__(
        self,
        agent: str,
        vote: str,        # "BUY", "SELL", "NEUTRAL", "SKIP", "WAIT"
        confidence: float,  # 0-1
        weight: float = 1.0,
    ) -> None:
        self.agent = agent
        self.vote = vote
        self.confidence = confidence
        self.weight = weight

    @property
    def weighted_score(self) -> float:
        """Positive for BUY, negative for SELL, 0 for neutral."""
        multiplier = {
            "BUY": 1, "STRONG_BUY": 1.2, "LONG": 1,
            "SELL": -1, "STRONG_SELL": -1.2, "SHORT": -1,
            "WAIT": 0, "SKIP": 0, "NEUTRAL": 0,
        }.get(self.vote.upper(), 0)
        return multiplier * self.confidence * self.weight


def compute_decision_matrix(
    votes: list[AgentVoteResult],
    debate_conviction: float,
    expected_value_pips: float,
    ev_per_risk: float,
    correlation_confirmed: bool,
    sentiment_extreme: bool,
    fear_greed_score: float,
) -> dict[str, Any]:
    """
    Enhanced decision matrix — not a simple majority vote.

    Weights agent votes by confidence, adjusts for debate quality,
    EV override, contradiction resolution, and correlation confirmation.
    """
    total_weighted = sum(v.weighted_score for v in votes)
    total_weight = sum(abs(v.weighted_score) for v in votes) or 1

    # Base agreement ratio: -1 (all disagree) to +1 (all agree)
    agreement = total_weighted / total_weight

    # Count agreements
    buy_votes = sum(1 for v in votes if v.weighted_score > 0)
    sell_votes = sum(1 for v in votes if v.weighted_score < 0)
    neutral_votes = sum(1 for v in votes if v.weighted_score == 0)
    total_votes = len(votes)

    # Base position from spec matrix
    if buy_votes == total_votes or sell_votes == total_votes:
        base_action = "full"  # All agree
        base_size = 1.0
    elif buy_votes >= total_votes - 1 and neutral_votes <= 1:
        base_action = "standard"  # 3 agree, 1 neutral
        base_size = 0.75
    elif buy_votes >= total_votes - 1 and sell_votes <= 1:
        base_action = "reduced"  # 3 agree, 1 disagrees
        base_size = 0.5
    elif abs(buy_votes - sell_votes) <= 1:
        base_action = "skip"  # 2 vs 2
        base_size = 0.0
    else:
        base_action = "standard"
        base_size = 0.5

    # Enhancement 1: Debate quality weighting
    conviction_factor = debate_conviction / 75  # 75 is "good" conviction
    base_size *= min(conviction_factor, 1.3)  # Can boost up to 30% for high conviction

    # Enhancement 2: EV override
    if ev_per_risk > 0.6:
        base_size = min(base_size * 1.2, 1.0)  # Excellent EV → boost
    elif ev_per_risk < 0.15 and base_size > 0:
        base_size *= 0.5  # Marginal EV → reduce

    # Enhancement 3: Correlation confirmation boost
    if correlation_confirmed:
        base_size = min(base_size * 1.1, 1.0)

    # Enhancement 4: Sentiment extreme override
    if sentiment_extreme:
        # Contrarian signal from sentiment → reduce size regardless
        base_size *= 0.7

    # Cap at 1.0 (1% risk)
    base_size = max(0, min(base_size, 1.0))

    # Determine direction
    direction = "LONG" if total_weighted > 0 else "SHORT" if total_weighted < 0 else "NEUTRAL"

    # Final action
    if base_size <= 0:
        action = "SKIP"
    elif base_size >= 0.9:
        action = "EXECUTE_FULL"
    elif base_size >= 0.6:
        action = "EXECUTE_STANDARD"
    elif base_size >= 0.3:
        action = "EXECUTE_REDUCED"
    else:
        action = "SKIP"

    return {
        "action": action,
        "direction": direction,
        "base_size_pct": round(base_size * float(RISK_RULES.max_risk_per_trade_pct), 3),
        "agreement_ratio": round(agreement, 3),
        "votes": {v.agent: {"vote": v.vote, "confidence": v.confidence, "weighted": round(v.weighted_score, 3)} for v in votes},
        "buy_votes": buy_votes,
        "sell_votes": sell_votes,
        "neutral_votes": neutral_votes,
        "enhancements": {
            "conviction_factor": round(conviction_factor, 2),
            "ev_per_risk": round(ev_per_risk, 3),
            "correlation_confirmed": correlation_confirmed,
            "sentiment_extreme": sentiment_extreme,
        },
    }


# ── Active Signal Manager ────────────────────────────────────

class SignalManager:
    """Manages the queue of active signals."""

    def __init__(self, max_pending: int = 5, max_age_minutes: int = 30) -> None:
        self._pending: list[dict[str, Any]] = []
        self._processed_ids: set[str] = set()
        self._max_pending = max_pending
        self._max_age_minutes = max_age_minutes

    def add_signal(self, signal: dict[str, Any]) -> bool:
        """Add a signal to the queue. Returns False if duplicate or queue full."""
        instrument = signal.get("instrument", "")
        direction = signal.get("direction", "")
        sig_id = f"{instrument}:{direction}"

        # Duplicate check
        if sig_id in self._processed_ids:
            return False
        for existing in self._pending:
            if existing.get("instrument") == instrument and existing.get("direction") == direction:
                # Replace if new signal is stronger
                if signal.get("confluence_score", 0) > existing.get("confluence_score", 0):
                    self._pending.remove(existing)
                    break
                else:
                    return False

        self._pending.append(signal)
        self._pending.sort(key=lambda s: s.get("confluence_score", 0), reverse=True)

        # Cap queue
        if len(self._pending) > self._max_pending:
            self._pending = self._pending[:self._max_pending]

        return True

    def get_next(self) -> dict[str, Any] | None:
        """Get the highest-priority signal."""
        self._expire_old()
        if not self._pending:
            return None
        signal = self._pending.pop(0)
        sig_id = f"{signal.get('instrument', '')}:{signal.get('direction', '')}"
        self._processed_ids.add(sig_id)
        return signal

    def _expire_old(self) -> None:
        """Remove signals older than max age."""
        now = datetime.now(timezone.utc)
        self._pending = [
            s for s in self._pending
            if self._signal_age(s) < self._max_age_minutes
        ]

    @staticmethod
    def _signal_age(signal: dict[str, Any]) -> float:
        """Signal age in minutes."""
        ts_str = signal.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            return (datetime.now(timezone.utc) - ts).total_seconds() / 60
        except (ValueError, TypeError):
            return 0

    def clear_processed(self) -> None:
        """Reset processed IDs (call periodically)."""
        self._processed_ids.clear()

    @property
    def queue_depth(self) -> int:
        return len(self._pending)


# ── Performance Tracker ───────────────────────────────────────

class PerformanceTracker:
    """Tracks trading performance for strategy adaptation."""

    def __init__(self) -> None:
        self._trades: list[dict[str, Any]] = []
        self._agent_accuracy: dict[str, dict[str, int]] = {}

    def record_closed_trade(self, trade: dict[str, Any]) -> None:
        self._trades.append(trade)
        if len(self._trades) > 200:
            self._trades = self._trades[-200:]

        # Track agent accuracy
        outcome = trade.get("outcome", "")
        for agent, vote in trade.get("agent_votes", {}).items():
            if agent not in self._agent_accuracy:
                self._agent_accuracy[agent] = {"correct": 0, "incorrect": 0, "total": 0}

            self._agent_accuracy[agent]["total"] += 1
            direction = trade.get("direction", "")
            vote_agrees = (
                (vote in ("BUY", "STRONG_BUY", "LONG") and direction == "LONG") or
                (vote in ("SELL", "STRONG_SELL", "SHORT") and direction == "SHORT")
            )
            if (vote_agrees and outcome == "win") or (not vote_agrees and outcome == "loss"):
                self._agent_accuracy[agent]["correct"] += 1
            else:
                self._agent_accuracy[agent]["incorrect"] += 1

    def get_stats(self, window: int = 30) -> dict[str, Any]:
        """Get performance stats over last N trades."""
        recent = self._trades[-window:] if self._trades else []
        if not recent:
            return {"trades": 0, "win_rate": 0, "avg_pnl": 0}

        wins = sum(1 for t in recent if t.get("outcome") == "win")
        total_pnl = sum(t.get("pnl_pips", 0) for t in recent)

        return {
            "trades": len(recent),
            "wins": wins,
            "losses": len(recent) - wins,
            "win_rate": round(wins / len(recent), 3) if recent else 0,
            "avg_pnl_pips": round(total_pnl / len(recent), 1) if recent else 0,
            "total_pnl_pips": round(total_pnl, 1),
            "agent_accuracy": {
                agent: round(d["correct"] / d["total"], 3) if d["total"] > 0 else 0
                for agent, d in self._agent_accuracy.items()
            },
        }

    def get_strategy_adjustments(self) -> dict[str, Any]:
        """Determine strategy adjustments based on performance."""
        stats = self.get_stats(30)
        adjustments: dict[str, Any] = {"mode": "normal", "changes": []}

        win_rate = stats.get("win_rate", 0.5)
        trades = stats.get("trades", 0)

        if trades < 10:
            return adjustments  # Not enough data

        if win_rate < 0.35:
            adjustments["mode"] = "conservative"
            adjustments["changes"].append("Win rate below 35% — raise min confluence to 70")
            adjustments["min_confluence"] = 70
            adjustments["max_concurrent"] = 2
        elif win_rate < 0.45:
            adjustments["mode"] = "cautious"
            adjustments["changes"].append("Win rate below 45% — raise min confluence to 65")
            adjustments["min_confluence"] = 65
        elif win_rate > 0.65:
            adjustments["mode"] = "aggressive"
            adjustments["changes"].append("Win rate above 65% — lower min confluence to 55")
            adjustments["min_confluence"] = 55
            adjustments["max_concurrent"] = 3

        # Agent-specific adjustments
        for agent, accuracy in stats.get("agent_accuracy", {}).items():
            if accuracy < 0.35 and trades >= 20:
                adjustments["changes"].append(
                    f"Agent {agent} accuracy at {accuracy:.0%} — reduce its weight"
                )

        return adjustments


# ── The Agent ─────────────────────────────────────────────────

class PortfolioManagerAgent(BaseAgent):
    """
    Agent 08: Portfolio Manager (Orchestrator)

    The CEO of the trading desk. Makes final decisions.
    """

    name: ClassVar[str] = "portfolio_manager"
    description: ClassVar[str] = "Final decision maker, synthesizes all agent inputs"
    execution_frequency: ClassVar[str] = "on_signal"

    def __init__(self) -> None:
        super().__init__()
        self._signal_manager = SignalManager()
        self._performance = PerformanceTracker()
        self._active_trades: list[dict[str, Any]] = []
        self._total_decisions = 0
        self._executions = 0
        self._skips = 0

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """
        Main loop: process signals from the queue.
        Called periodically by the scheduler.
        """
        # Pull signals from Redis queue
        signals_processed = 0
        decisions: list[dict[str, Any]] = []

        while True:
            signal = await dequeue_signal(timeout=1)
            if signal is None:
                break

            # Triage
            level = triage_signal(signal)
            if level == TriageLevel.NOISE:
                continue

            if not self._signal_manager.add_signal(signal):
                continue  # Duplicate or queue full

            signals_processed += 1

        # Process top signal from queue
        next_signal = self._signal_manager.get_next()
        if next_signal:
            decision = await self._process_signal(next_signal, market_state)
            decisions.append(decision)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals_processed": signals_processed,
            "queue_depth": self._signal_manager.queue_depth,
            "decisions": decisions,
            "active_trades": len(self._active_trades),
            "performance": self._performance.get_stats(),
            "strategy_adjustments": self._performance.get_strategy_adjustments(),
        }

    async def _process_signal(
        self,
        signal: dict[str, Any],
        market_state: MarketState,
    ) -> dict[str, Any]:
        """Process a single signal through the full decision pipeline."""
        self._total_decisions += 1
        instrument = signal.get("instrument", "")
        direction = signal.get("direction", "LONG")

        self.logger.info(
            "Processing signal",
            extra={"instrument": instrument, "direction": direction,
                   "confluence": signal.get("confluence_score", 0)},
        )

        # Build trade proposal
        trade_id = f"T-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._total_decisions:03d}"
        proposal = TradeProposal(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            entry=signal.get("entry", 0),
            stop_loss=signal.get("stop_loss", 0),
            take_profit_1=signal.get("take_profit", 0),
            take_profit_2=signal.get("take_profit_2"),
            atr=signal.get("atr", 0),
            timeframe=signal.get("timeframe", "1h"),
            technical_signal=signal,
        )

        # Preflight: quick risk check before expensive debate
        preflight = await self._preflight_check(proposal, market_state)
        if preflight.get("blocked"):
            self._skips += 1
            return {
                "trade_id": trade_id,
                "decision": "REJECTED_PREFLIGHT",
                "reason": preflight.get("reason", ""),
                "instrument": instrument,
            }

        # Check active trade limit
        if len(self._active_trades) >= RISK_RULES.max_concurrent_trades:
            self._skips += 1
            return {
                "trade_id": trade_id,
                "decision": "REJECTED_MAX_TRADES",
                "reason": f"At max concurrent trades ({RISK_RULES.max_concurrent_trades})",
                "instrument": instrument,
            }

        # Triage level determines processing depth
        level = triage_signal(signal)

        if level == TriageLevel.WEAK:
            # Skip debate, just run risk check
            decision = await self._quick_decision(proposal, signal, market_state)
        else:
            # Full pipeline: debate → risk → decide
            decision = await self._full_pipeline(proposal, signal, market_state)

        # Execute if approved
        if decision.get("final_decision") in ("EXECUTE", "EXECUTE_FULL", "EXECUTE_STANDARD", "EXECUTE_REDUCED"):
            await self._execute_trade(decision, proposal)
            self._executions += 1
        else:
            self._skips += 1

        return decision

    async def _preflight_check(
        self,
        proposal: TradeProposal,
        market_state: MarketState,
    ) -> dict[str, Any]:
        """Quick preflight — catch hard failures before spending on debate."""
        blocked = False
        reason = ""

        # Daily/weekly halt
        if float(market_state.daily_pnl_pct) <= float(RISK_RULES.daily_loss_limit_pct):
            blocked = True
            reason = f"Daily loss limit breached: {market_state.daily_pnl_pct}%"

        if float(market_state.weekly_pnl_pct) <= float(RISK_RULES.weekly_loss_limit_pct):
            blocked = True
            reason = f"Weekly loss limit breached: {market_state.weekly_pnl_pct}%"

        # Session check
        now = datetime.now(timezone.utc)
        if not (RISK_RULES.session_start_utc <= now.hour < RISK_RULES.session_end_utc):
            blocked = True
            reason = f"Outside trading session ({now.hour}:00 UTC)"

        # Weekend check
        if now.weekday() > RISK_RULES.weekend_close_day:
            blocked = True
            reason = "Weekend — no trading"
        elif now.weekday() == RISK_RULES.weekend_close_day and now.hour >= RISK_RULES.weekend_close_hour_utc:
            blocked = True
            reason = f"Friday after {RISK_RULES.weekend_close_hour_utc}:00 UTC"

        # No stop loss
        if proposal.stop_loss == 0:
            blocked = True
            reason = "No stop loss defined"

        return {"blocked": blocked, "reason": reason}

    async def _quick_decision(
        self,
        proposal: TradeProposal,
        signal: dict[str, Any],
        market_state: MarketState,
    ) -> dict[str, Any]:
        """Quick decision for weak signals — skip debate, risk check only."""
        # Just gather agent votes and apply decision matrix
        votes = await self._collect_votes(proposal)
        matrix = compute_decision_matrix(
            votes=votes,
            debate_conviction=signal.get("confluence_score", 50),
            expected_value_pips=0,
            ev_per_risk=0,
            correlation_confirmed=False,
            sentiment_extreme=False,
            fear_greed_score=50,
        )

        return {
            "trade_id": proposal.trade_id,
            "instrument": proposal.instrument,
            "direction": proposal.direction,
            "final_decision": matrix["action"],
            "decision_type": "quick",
            "matrix": matrix,
        }

    async def _full_pipeline(
        self,
        proposal: TradeProposal,
        signal: dict[str, Any],
        market_state: MarketState,
    ) -> dict[str, Any]:
        """Full pipeline: debate → risk → decide."""
        # Run debate
        from src.orchestration.debate import DebateEngine
        debate = DebateEngine()
        verdict = await debate.run_debate(proposal, market_state)
        verdict_dict = verdict.to_dict()

        # Run risk manager
        from src.agents.risk_manager import RiskManagerAgent
        risk_mgr = RiskManagerAgent()
        risk_result = await risk_mgr.evaluate_trade(
            proposal=proposal,
            debate_verdict=verdict_dict,
            bull_package=verdict.bull_package,
            bear_package=verdict.bear_package,
            market_state=market_state,
        )

        # If risk rejects, that's final — ABSOLUTE VETO
        if risk_result.get("decision") == "REJECTED":
            return {
                "trade_id": proposal.trade_id,
                "instrument": proposal.instrument,
                "direction": proposal.direction,
                "final_decision": "REJECTED",
                "decision_type": "full_pipeline",
                "veto_by": "risk_manager",
                "veto_reason": risk_result.get("veto_reason", ""),
                "debate": verdict_dict,
                "risk": risk_result,
            }

        # Collect votes and apply decision matrix
        votes = await self._collect_votes(proposal)

        # Get sentiment info
        sent_state = await get_agent_state("sentiment_agent")
        fg = sent_state.get("fear_greed_index", {}) if sent_state else {}
        fg_score = fg.get("score", 50)
        sentiment_extreme = fg.get("contrarian_signal", "none") != "none"

        # Get correlation info
        corr_state = await get_agent_state("correlation_agent")
        corr_confirmed = False
        if corr_state:
            for cv in corr_state.get("cross_validation", {}).get("technical_signals", []):
                if cv.get("instrument") == proposal.instrument:
                    corr_confirmed = cv.get("correlation_assessment") == "CONFIRMED"

        ev_data = verdict_dict.get("expected_value", {})

        matrix = compute_decision_matrix(
            votes=votes,
            debate_conviction=verdict_dict.get("conviction", 50),
            expected_value_pips=ev_data.get("expected_value_pips", 0),
            ev_per_risk=ev_data.get("ev_per_risk_unit", 0),
            correlation_confirmed=corr_confirmed,
            sentiment_extreme=sentiment_extreme,
            fear_greed_score=fg_score,
        )

        # Final position size from risk manager
        final_size = risk_result.get("final_position_size", matrix.get("base_size_pct", 0))

        final_decision = matrix["action"]
        if final_decision.startswith("EXECUTE") and risk_result.get("decision") == "APPROVED_WITH_MODIFICATION":
            final_decision = "EXECUTE_MODIFIED"

        return {
            "trade_id": proposal.trade_id,
            "instrument": proposal.instrument,
            "direction": proposal.direction,
            "entry": proposal.entry,
            "stop_loss": proposal.stop_loss,
            "take_profit_1": proposal.take_profit_1,
            "take_profit_2": proposal.take_profit_2,
            "final_decision": final_decision,
            "final_position_size": final_size,
            "decision_type": "full_pipeline",
            "matrix": matrix,
            "debate": verdict_dict,
            "risk": risk_result,
            "confidence": verdict_dict.get("conviction", 0) / 100,
        }

    async def _collect_votes(self, proposal: TradeProposal) -> list[AgentVoteResult]:
        """Collect votes from all analysis agents."""
        votes: list[AgentVoteResult] = []

        # Technical Agent
        tech = await get_agent_state("technical_analyst")
        if tech:
            for sig in tech.get("signals", []):
                if sig.get("instrument") == proposal.instrument:
                    votes.append(AgentVoteResult(
                        "technical", sig.get("signal", "NEUTRAL"),
                        sig.get("confidence", 0.5), weight=1.0,
                    ))
                    break

        # Macro Agent
        macro = await get_agent_state("macro_analyst")
        if macro:
            preferred = macro.get("preferred_pairs", [])
            for pair in preferred:
                if pair.get("pair") == proposal.instrument:
                    votes.append(AgentVoteResult(
                        "macro", pair.get("direction", "NEUTRAL"),
                        {"high": 0.85, "medium": 0.65, "low": 0.45}.get(pair.get("conviction", "low"), 0.5),
                        weight=1.0,
                    ))
                    break
            if not any(v.agent == "macro" for v in votes):
                votes.append(AgentVoteResult("macro", "NEUTRAL", 0.5, weight=0.8))

        # Correlation Agent
        corr = await get_agent_state("correlation_agent")
        if corr:
            for cv in corr.get("cross_validation", {}).get("technical_signals", []):
                if cv.get("instrument") == proposal.instrument:
                    verdict = cv.get("correlation_assessment", "NEUTRAL")
                    conf = cv.get("adjusted_confidence", 0.5)
                    vote_map = {"CONFIRMED": "BUY", "CONTRADICTED": "SELL", "NEUTRAL": "NEUTRAL"}
                    if proposal.direction == "SHORT":
                        vote_map = {"CONFIRMED": "SELL", "CONTRADICTED": "BUY", "NEUTRAL": "NEUTRAL"}
                    votes.append(AgentVoteResult(
                        "correlation", vote_map.get(verdict, "NEUTRAL"), conf, weight=1.0,
                    ))
                    break

        # Sentiment Agent
        sent = await get_agent_state("sentiment_agent")
        if sent:
            currency_sent = sent.get("currency_sentiment", {})
            inst = INSTRUMENTS.get(proposal.instrument)
            if inst:
                base_score = currency_sent.get(inst.base_currency, {}).get("score", 0)
                if base_score > 1:
                    sent_vote = "BUY" if proposal.direction == "LONG" else "NEUTRAL"
                elif base_score < -1:
                    sent_vote = "SELL" if proposal.direction == "SHORT" else "NEUTRAL"
                else:
                    sent_vote = "NEUTRAL"
                votes.append(AgentVoteResult("sentiment", sent_vote, abs(base_score) / 5, weight=0.8))

        return votes

    async def _execute_trade(
        self,
        decision: dict[str, Any],
        proposal: TradeProposal,
    ) -> None:
        """Execute an approved trade — send to Telegram, optionally place on broker, log."""
        self.logger.info(
            "EXECUTING TRADE",
            extra={
                "trade_id": proposal.trade_id,
                "instrument": proposal.instrument,
                "direction": proposal.direction,
                "size": decision.get("final_position_size", 0),
            },
        )

        # Send Telegram signal
        try:
            from src.execution.telegram_bot import send_trade_signal
            await send_trade_signal(decision, proposal)
        except Exception as e:
            self.logger.error("Telegram send failed", extra={"error": str(e)})

        # Place order on Capital.com if AUTO_EXECUTE is enabled
        broker_deal_id = None
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            if settings.auto_execute and settings.capital_api_key.get_secret_value():
                broker_deal_id = await self._place_broker_order(decision, proposal)
        except Exception as e:
            self.logger.error("Broker execution failed", extra={"error": str(e)})

        # Log trade
        try:
            from src.execution.trade_logger import log_new_trade
            await log_new_trade(decision, proposal)
        except Exception as e:
            self.logger.error("Trade logging failed", extra={"error": str(e)})

        # Track active trade
        self._active_trades.append({
            "trade_id": proposal.trade_id,
            "instrument": proposal.instrument,
            "direction": proposal.direction,
            "entry": proposal.entry,
            "stop_loss": proposal.stop_loss,
            "take_profit_1": proposal.take_profit_1,
            "atr": proposal.atr,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "broker_deal_id": broker_deal_id,
            "position_size_pct": decision.get("final_position_size", 0),
        })

    async def _place_broker_order(
        self,
        decision: dict[str, Any],
        proposal: TradeProposal,
    ) -> str | None:
        """Calculate lot size and place order on Capital.com. Returns position deal_id."""
        from src.execution.broker_api import get_broker

        broker = get_broker()
        if not await broker._ensure_session():
            self.logger.error("Broker auth failed — cannot execute")
            return None

        # Get account equity for position sizing
        acct = await broker.get_account()
        if not acct:
            self.logger.error("Cannot fetch account — skipping broker execution")
            return None

        equity = acct["available"]
        instrument = proposal.instrument
        inst = INSTRUMENTS.get(instrument)
        if not inst:
            self.logger.error("Unknown instrument for broker", extra={"instrument": instrument})
            return None

        # Calculate lot size: (equity × position_size_pct) / (risk_pips × pip_value_per_lot)
        position_size_pct = decision.get("final_position_size", 0)
        risk_pips = abs(float(proposal.entry) - float(proposal.stop_loss)) / float(inst.pip_size)
        pip_value = float(inst.pip_value_per_lot)

        if risk_pips <= 0 or pip_value <= 0:
            self.logger.error("Invalid risk calculation", extra={"risk_pips": risk_pips, "pip_value": pip_value})
            return None

        risk_amount = equity * (position_size_pct / 100)
        lots = risk_amount / (risk_pips * pip_value)

        # Get min size from broker
        market_info = await broker.get_market_info(instrument)
        min_size = float(market_info.get("min_size", 0.01)) if market_info else 0.01

        # Capital.com uses raw units for forex (1 lot = 100,000 units)
        # Convert lots to units depending on instrument type
        if inst.category.value in ("major", "minor", "cross"):
            size = max(round(lots * 100_000, 0), min_size)  # forex: lots → units
        else:
            size = max(round(lots, 2), min_size)  # commodities: lots directly

        direction = "BUY" if proposal.direction == "LONG" else "SELL"

        self.logger.info(
            "Placing broker order",
            extra={
                "instrument": instrument,
                "direction": direction,
                "size": size,
                "equity": equity,
                "risk_pct": position_size_pct,
                "risk_pips": round(risk_pips, 1),
                "lots": round(lots, 4),
            },
        )

        result = await broker.open_position(
            instrument=instrument,
            direction=direction,
            size=size,
            stop_loss=float(proposal.stop_loss),
            take_profit=float(proposal.take_profit_1),
        )

        if result and result.get("status") == "ACCEPTED":
            deal_id = result.get("position_deal_id", result.get("deal_id"))
            self.logger.info("Broker order ACCEPTED", extra={"deal_id": deal_id, "level": result.get("level")})
            return deal_id

        self.logger.error("Broker order REJECTED", extra={"result": result})
        return None

    @property
    def stats(self) -> dict[str, Any]:
        base = super().stats
        base["total_decisions"] = self._total_decisions
        base["executions"] = self._executions
        base["skips"] = self._skips
        base["active_trades"] = len(self._active_trades)
        base["queue_depth"] = self._signal_manager.queue_depth
        base["performance"] = self._performance.get_stats()
        return base
