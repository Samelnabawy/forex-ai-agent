"""
Core data models shared across all agents and pipeline components.
Every agent input/output is a Pydantic model — no raw dicts allowed.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────

class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStrength(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TradeStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED_WIN = "closed_win"
    CLOSED_LOSS = "closed_loss"
    CANCELLED = "cancelled"
    VETOED = "vetoed"


class MacroRegime(StrEnum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    TRANSITIONING = "transitioning"


class RiskCheckResult(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class AgentVote(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    WAIT = "WAIT"
    SKIP = "SKIP"


# ── Market Data ───────────────────────────────────────────────

class Candle(BaseModel):
    """Single OHLCV candle."""
    instrument: str
    timeframe: str
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")


class PriceTick(BaseModel):
    """Real-time price update from WebSocket."""
    instrument: str
    bid: Decimal
    ask: Decimal
    ts: datetime
    spread: Decimal = Decimal("0")

    @model_validator(mode="after")
    def compute_spread(self) -> "PriceTick":
        if self.spread == Decimal("0") and self.bid and self.ask:
            self.spread = self.ask - self.bid
        return self


class MarketState(BaseModel):
    """
    Snapshot of the current market state for all instruments.
    Passed to every agent on each analysis cycle.
    """
    ts: datetime
    prices: dict[str, PriceTick] = Field(default_factory=dict)
    candles: dict[str, dict[str, list[Candle]]] = Field(default_factory=dict)  # instrument → timeframe → candles
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    macro_regime: MacroRegime = MacroRegime.NEUTRAL
    open_trades: list[TradeRecord] = Field(default_factory=list)
    daily_pnl_pct: Decimal = Decimal("0")
    weekly_pnl_pct: Decimal = Decimal("0")


# ── Agent Outputs ─────────────────────────────────────────────

class TechnicalSignal(BaseModel):
    """Output from the Technical Analyst agent."""
    instrument: str
    ts: datetime
    signal: SignalStrength
    confidence: Decimal = Field(ge=0, le=1)
    timeframe: str
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    indicators: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class MacroOutlook(BaseModel):
    """Output from the Macro Analyst agent."""
    ts: datetime
    macro_regime: MacroRegime
    usd_outlook: str
    currency_scores: dict[str, int] = Field(default_factory=dict)  # currency → -5 to +5
    preferred_pairs: list[dict[str, Any]] = Field(default_factory=list)
    gold_outlook: str = ""
    oil_outlook: str = ""
    upcoming_events: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str = ""


class CorrelationSignal(BaseModel):
    """Output from the Correlation Agent."""
    ts: datetime
    market_regime: str
    signals: list[dict[str, Any]] = Field(default_factory=list)
    correlation_health: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SentimentReading(BaseModel):
    """Output from the Sentiment Agent."""
    ts: datetime
    overall_sentiment: dict[str, Any] = Field(default_factory=dict)
    currency_sentiment: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cot_signals: dict[str, str] = Field(default_factory=dict)
    upcoming_event_analysis: dict[str, Any] = Field(default_factory=dict)


class ResearchCase(BaseModel):
    """Output from Bull or Bear Researcher."""
    case: str  # "BULLISH" or "BEARISH"
    trade: str  # e.g. "LONG EUR/USD"
    evidence: list[str] = Field(default_factory=list)
    proposed_trade: dict[str, Any] = Field(default_factory=dict)
    confidence: Decimal = Field(ge=0, le=1)
    risk_flags: list[str] = Field(default_factory=list)
    recommendation: str = ""


class RiskAssessment(BaseModel):
    """Output from the Risk Manager."""
    trade_id: str
    decision: str  # "APPROVED", "APPROVED_WITH_MODIFICATION", "REJECTED"
    checks: dict[str, str] = Field(default_factory=dict)  # check_name → "PASS"/"WARNING"/"FAIL"
    modifications: dict[str, Any] = Field(default_factory=dict)
    final_recommendation: str = ""


class TradeDecision(BaseModel):
    """Final output from the Portfolio Manager — the trade order."""
    trade_id: str
    final_decision: str  # "EXECUTE", "SKIP", "WAIT"
    instrument: str
    direction: Direction
    entry: Decimal
    stop_loss: Decimal
    take_profit_1: Decimal
    take_profit_2: Decimal | None = None
    position_size: Decimal  # % of capital
    confidence: Decimal = Field(ge=0, le=1)
    reasoning_summary: str = ""
    agent_votes: dict[str, str] = Field(default_factory=dict)
    execute_after: datetime | None = None


# ── Trade Tracking ────────────────────────────────────────────

class TradeRecord(BaseModel):
    """A trade in the system — from creation to close."""
    trade_id: str
    instrument: str
    direction: Direction
    entry_price: Decimal
    stop_loss: Decimal
    take_profit_1: Decimal
    take_profit_2: Decimal | None = None
    position_size: Decimal
    confidence: Decimal
    status: TradeStatus = TradeStatus.PENDING
    exit_price: Decimal | None = None
    pnl_pips: Decimal | None = None
    pnl_percent: Decimal | None = None
    agent_reasoning: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Historical Brain ──────────────────────────────────────────

class HistoricalMatch(BaseModel):
    """A single match from the Historical Brain RAG query."""
    event_id: int
    event_name: str
    event_type: str
    ts: datetime
    similarity_score: float = Field(ge=0, le=1)
    description: str = ""
    price_impact: dict[str, Any] = Field(default_factory=dict)
    market_context: dict[str, Any] = Field(default_factory=dict)


class BrainQuery(BaseModel):
    """Query sent to the Historical Brain."""
    query_text: str
    filters: dict[str, Any] = Field(default_factory=dict)  # event_type, affected_pairs, date range
    top_k: int = Field(default=10, ge=1, le=50)
    requesting_agent: str = ""
