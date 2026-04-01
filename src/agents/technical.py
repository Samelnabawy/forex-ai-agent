"""
Technical Analyst Agent — Agent 01

"The Numbers Guy" — purely data-driven, no opinions, just pattern recognition.

Scans all 9 instruments every 5 minutes across execution timeframes (5m, 15m, 1h).
Uses context timeframes (4h, Daily) for trend alignment.
Detects confluent setups and publishes structured TechnicalSignal outputs.

Setup Detection Logic:
  1. Compute full indicator stack per instrument per timeframe
  2. Score individual signals (trend, momentum, volatility, volume, structure)
  3. Check multi-timeframe alignment
  4. Calculate confluence score → if above threshold → generate signal
  5. Compute entry, stop loss (ATR-based), take profit (R:R based)

Does NOT execute trades. Passes signals to Bull/Bear researchers and Portfolio Manager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.indicators import (
    IndicatorSnapshot,
    SignalType,
    TrendDirection,
    compute_all_indicators,
)
from src.config.instruments import (
    ALL_SYMBOLS,
    CONTEXT_TIMEFRAMES,
    EXECUTION_TIMEFRAMES,
    INSTRUMENTS,
)
from src.config.risk_rules import RISK_RULES
from src.data.storage.cache import enqueue_signal
from src.models import MarketState, TechnicalSignal

logger = logging.getLogger(__name__)

# ── Setup Detection Thresholds ────────────────────────────────

# Minimum confluence score to generate a signal (0-100)
MIN_CONFLUENCE_SCORE = 60

# Minimum ADX for trending market (below = ranging, different strategy)
MIN_ADX_TRENDING = 25

# Signal weights by category
SIGNAL_WEIGHTS = {
    "trend": 25,
    "momentum": 25,
    "volatility": 10,
    "volume": 15,
    "structure": 15,
    "mtf_alignment": 10,  # multi-timeframe
}

# ATR multiplier for stop loss calculation
STOP_LOSS_ATR_MULTIPLIER = 1.5


# ── Signal Scoring ────────────────────────────────────────────

@dataclass
class SignalScore:
    """Scored signal from one instrument/timeframe."""
    instrument: str
    timeframe: str
    direction: str  # "LONG" or "SHORT"
    confluence_score: float  # 0-100
    breakdown: dict[str, float] = field(default_factory=dict)
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    take_profit_2: float = 0.0
    atr: float = 0.0
    indicators: dict[str, Any] = field(default_factory=dict)
    patterns: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)


def score_trend(snapshot: IndicatorSnapshot) -> tuple[float, str, list[str]]:
    """Score trend indicators. Returns (score 0-100, direction, reasoning)."""
    t = snapshot.trend
    score = 0.0
    bullish_pts = 0.0
    bearish_pts = 0.0
    reasons: list[str] = []

    # EMA alignment (strong signal)
    if t.ema_alignment == "bullish":
        bullish_pts += 30
        reasons.append("EMA alignment bullish (20>50>200)")
    elif t.ema_alignment == "bearish":
        bearish_pts += 30
        reasons.append("EMA alignment bearish (20<50<200)")

    # ADX strength
    if t.adx and t.adx > MIN_ADX_TRENDING:
        strength_bonus = min((t.adx - MIN_ADX_TRENDING) / 25 * 20, 20)
        if t.adx_plus_di and t.adx_minus_di:
            if t.adx_plus_di > t.adx_minus_di:
                bullish_pts += strength_bonus
                reasons.append(f"ADX {t.adx:.1f} trending, +DI > -DI")
            else:
                bearish_pts += strength_bonus
                reasons.append(f"ADX {t.adx:.1f} trending, -DI > +DI")

    # Ichimoku
    if t.ichimoku_signal == "above_cloud":
        bullish_pts += 20
        reasons.append("Price above Ichimoku cloud")
    elif t.ichimoku_signal == "below_cloud":
        bearish_pts += 20
        reasons.append("Price below Ichimoku cloud")

    # Supertrend
    if t.supertrend_direction == "bullish":
        bullish_pts += 15
        reasons.append("Supertrend bullish")
    elif t.supertrend_direction == "bearish":
        bearish_pts += 15
        reasons.append("Supertrend bearish")

    # Tenkan/Kijun cross
    if t.ichimoku_tenkan and t.ichimoku_kijun:
        if t.ichimoku_tenkan > t.ichimoku_kijun:
            bullish_pts += 10
        else:
            bearish_pts += 10

    # Determine direction and normalize score
    max_possible = 95  # theoretical max points
    if bullish_pts > bearish_pts:
        direction = "LONG"
        score = min(bullish_pts / max_possible * 100, 100)
    elif bearish_pts > bullish_pts:
        direction = "SHORT"
        score = min(bearish_pts / max_possible * 100, 100)
    else:
        direction = "NEUTRAL"
        score = 0

    return score, direction, reasons


def score_momentum(snapshot: IndicatorSnapshot, direction: str) -> tuple[float, list[str]]:
    """Score momentum indicators aligned to direction."""
    m = snapshot.momentum
    score = 0.0
    reasons: list[str] = []
    max_possible = 100.0

    is_long = direction == "LONG"

    # RSI
    if m.rsi is not None:
        if is_long and m.rsi_signal == "oversold":
            score += 25
            reasons.append(f"RSI oversold ({m.rsi:.1f})")
        elif not is_long and m.rsi_signal == "overbought":
            score += 25
            reasons.append(f"RSI overbought ({m.rsi:.1f})")
        elif is_long and 40 < m.rsi < 60:
            score += 10  # neutral, not fighting momentum
        elif not is_long and 40 < m.rsi < 60:
            score += 10

    # MACD cross
    if m.macd_cross == "bullish" and is_long:
        score += 25
        reasons.append("MACD bullish cross")
    elif m.macd_cross == "bearish" and not is_long:
        score += 25
        reasons.append("MACD bearish cross")

    # Stochastic
    if is_long and m.stoch_signal == "oversold":
        score += 15
        reasons.append(f"Stochastic oversold ({m.stoch_k:.1f})")
    elif not is_long and m.stoch_signal == "overbought":
        score += 15
        reasons.append(f"Stochastic overbought ({m.stoch_k:.1f})")

    # Williams %R
    if is_long and m.williams_signal == "oversold":
        score += 10
        reasons.append(f"Williams %R oversold ({m.williams_r:.1f})")
    elif not is_long and m.williams_signal == "overbought":
        score += 10
        reasons.append(f"Williams %R overbought ({m.williams_r:.1f})")

    # CCI
    if is_long and m.cci_signal == "oversold":
        score += 10
        reasons.append(f"CCI oversold ({m.cci:.1f})")
    elif not is_long and m.cci_signal == "overbought":
        score += 10
        reasons.append(f"CCI overbought ({m.cci:.1f})")

    # MACD histogram strength
    if m.macd_histogram is not None:
        if is_long and m.macd_histogram > 0:
            score += 15
        elif not is_long and m.macd_histogram < 0:
            score += 15

    return min(score / max_possible * 100, 100), reasons


def score_volatility(snapshot: IndicatorSnapshot, direction: str) -> tuple[float, list[str]]:
    """Score volatility conditions."""
    v = snapshot.volatility
    score = 50.0  # Start neutral — volatility can confirm or warn
    reasons: list[str] = []
    is_long = direction == "LONG"

    # Bollinger Band position
    if is_long and v.bb_position == "below_lower":
        score += 20
        reasons.append("Price below lower BB — potential reversal")
    elif not is_long and v.bb_position == "above_upper":
        score += 20
        reasons.append("Price above upper BB — potential reversal")

    # Squeeze (low volatility → expect expansion)
    if v.squeeze:
        score += 20
        reasons.append("BB/Keltner squeeze — volatility expansion imminent")

    # ATR — moderate ATR is best (too low = no movement, too high = wild)
    if v.atr_pct is not None:
        if 0.3 < v.atr_pct < 1.5:
            score += 10
            reasons.append(f"ATR {v.atr_pct:.2f}% — healthy volatility")

    return min(score, 100), reasons


def score_volume(snapshot: IndicatorSnapshot, direction: str) -> tuple[float, list[str]]:
    """Score volume confirmation."""
    vol = snapshot.volume
    score = 50.0  # Neutral default
    reasons: list[str] = []
    is_long = direction == "LONG"

    # VWAP
    if is_long and vol.price_vs_vwap == "above":
        score += 15
        reasons.append("Price above VWAP")
    elif not is_long and vol.price_vs_vwap == "below":
        score += 15
        reasons.append("Price below VWAP")

    # OBV trend
    if is_long and vol.obv_trend == "rising":
        score += 15
        reasons.append("OBV rising — volume confirms uptrend")
    elif not is_long and vol.obv_trend == "falling":
        score += 15
        reasons.append("OBV falling — volume confirms downtrend")

    # MFI
    if is_long and vol.mfi_signal == "oversold":
        score += 20
        reasons.append(f"MFI oversold ({vol.mfi:.1f}) — buying opportunity")
    elif not is_long and vol.mfi_signal == "overbought":
        score += 20
        reasons.append(f"MFI overbought ({vol.mfi:.1f}) — selling pressure")

    return min(score, 100), reasons


def score_structure(
    snapshot: IndicatorSnapshot, direction: str
) -> tuple[float, list[str]]:
    """Score structural levels (S/R, pivots, fibs)."""
    s = snapshot.structure
    price = snapshot.current_price
    score = 50.0
    reasons: list[str] = []
    is_long = direction == "LONG"

    # Near support (good for longs) or resistance (good for shorts)
    if is_long and s.nearest_support:
        dist_pct = (price - s.nearest_support) / price * 100
        if dist_pct < 0.3:  # Within 0.3% of support
            score += 25
            reasons.append(f"Price near support {s.nearest_support:.5f}")

    if not is_long and s.nearest_resistance:
        dist_pct = (s.nearest_resistance - price) / price * 100
        if dist_pct < 0.3:
            score += 25
            reasons.append(f"Price near resistance {s.nearest_resistance:.5f}")

    # Pivot points
    if is_long and s.daily_s1:
        if abs(price - s.daily_s1) / price < 0.002:
            score += 15
            reasons.append("Price at daily S1 pivot")
    if not is_long and s.daily_r1:
        if abs(price - s.daily_r1) / price < 0.002:
            score += 15
            reasons.append("Price at daily R1 pivot")

    # Candlestick patterns
    bullish_patterns = {"hammer", "bullish_engulfing", "morning_star", "three_white_soldiers"}
    bearish_patterns = {"shooting_star", "bearish_engulfing", "evening_star", "three_black_crows"}

    for p in s.candlestick_patterns:
        if is_long and p in bullish_patterns:
            score += 10
            reasons.append(f"Bullish pattern: {p}")
        elif not is_long and p in bearish_patterns:
            score += 10
            reasons.append(f"Bearish pattern: {p}")

    return min(score, 100), reasons


def score_mtf_alignment(
    exec_snapshot: IndicatorSnapshot,
    context_snapshots: list[IndicatorSnapshot],
    direction: str,
) -> tuple[float, list[str]]:
    """Score multi-timeframe alignment with higher TF context."""
    if not context_snapshots:
        return 50.0, ["No context timeframes available"]

    score = 0.0
    reasons: list[str] = []
    is_long = direction == "LONG"

    for ctx in context_snapshots:
        t = ctx.trend
        # Check EMA alignment on higher TF
        if is_long and t.ema_alignment == "bullish":
            score += 35
            reasons.append(f"{ctx.timeframe} trend bullish (EMA aligned)")
        elif not is_long and t.ema_alignment == "bearish":
            score += 35
            reasons.append(f"{ctx.timeframe} trend bearish (EMA aligned)")
        elif t.ema_alignment == "mixed":
            score += 10
            reasons.append(f"{ctx.timeframe} trend mixed")

        # Ichimoku on higher TF
        if is_long and t.ichimoku_signal == "above_cloud":
            score += 15
            reasons.append(f"{ctx.timeframe} above Ichimoku cloud")
        elif not is_long and t.ichimoku_signal == "below_cloud":
            score += 15
            reasons.append(f"{ctx.timeframe} below Ichimoku cloud")

    # Normalize
    max_possible = len(context_snapshots) * 50
    return min(score / max_possible * 100, 100) if max_possible > 0 else 50.0, reasons


# ── Trade Parameter Calculation ───────────────────────────────

def compute_trade_params(
    snapshot: IndicatorSnapshot,
    direction: str,
    instrument_symbol: str,
) -> dict[str, float]:
    """Compute entry, stop loss, and take profit using ATR."""
    inst = INSTRUMENTS[instrument_symbol]
    price = snapshot.current_price
    atr = snapshot.volatility.atr or 0

    if atr == 0:
        # Fallback: use pip-based stop
        atr = float(inst.avg_daily_range_pips * inst.pip_size) / 10

    stop_distance = atr * STOP_LOSS_ATR_MULTIPLIER
    min_rr = float(RISK_RULES.min_risk_reward_ratio)

    if direction == "LONG":
        entry = price
        stop_loss = price - stop_distance
        take_profit_1 = price + stop_distance * min_rr
        take_profit_2 = price + stop_distance * (min_rr + 1.0)
    else:
        entry = price
        stop_loss = price + stop_distance
        take_profit_1 = price - stop_distance * min_rr
        take_profit_2 = price - stop_distance * (min_rr + 1.0)

    return {
        "entry": round(entry, 6),
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit_1, 6),
        "take_profit_2": round(take_profit_2, 6),
        "atr": round(atr, 6),
        "stop_pips": round(stop_distance / float(inst.pip_size), 1),
    }


# ── The Agent ─────────────────────────────────────────────────

class TechnicalAnalystAgent(BaseAgent):
    """
    Agent 01: Technical Analyst

    Scans all 9 instruments across execution timeframes.
    Computes full indicator stack, scores confluence, generates signals.
    """

    name: ClassVar[str] = "technical_analyst"
    description: ClassVar[str] = "Quantitative pattern recognition across all instruments"
    execution_frequency: ClassVar[str] = "5m"

    def __init__(self) -> None:
        super().__init__()
        self._signals_generated = 0

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """
        Core analysis: scan all instruments, score setups, generate signals.
        Returns summary of all scanned instruments and any signals found.
        """
        signals: list[dict[str, Any]] = []
        scan_results: list[dict[str, Any]] = []

        for symbol in ALL_SYMBOLS:
            try:
                result = await self._analyze_instrument(symbol, market_state)
                scan_results.append(result["scan"])
                if result.get("signal"):
                    signals.append(result["signal"])
            except Exception as e:
                self.logger.error(
                    "Failed to analyze instrument",
                    extra={"instrument": symbol, "error": str(e)},
                )

        # Enqueue signals for downstream agents
        for sig in signals:
            await enqueue_signal(sig)
            self._signals_generated += 1

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instruments_scanned": len(scan_results),
            "signals_generated": len(signals),
            "signals": signals,
            "scan_summary": scan_results,
        }

    async def _analyze_instrument(
        self,
        symbol: str,
        market_state: MarketState,
    ) -> dict[str, Any]:
        """Analyze a single instrument across all timeframes."""

        # Build DataFrames from market state candles
        candle_data = market_state.candles.get(symbol, {})

        # Compute indicators for each execution timeframe
        exec_snapshots: dict[str, IndicatorSnapshot] = {}
        for tf in EXECUTION_TIMEFRAMES:
            df = self._candles_to_dataframe(candle_data.get(tf, []))
            daily_df = self._candles_to_dataframe(candle_data.get("1d", []))
            exec_snapshots[tf] = compute_all_indicators(
                df, symbol, tf, daily_df=daily_df
            )

        # Compute context timeframe indicators
        ctx_snapshots: list[IndicatorSnapshot] = []
        for tf in CONTEXT_TIMEFRAMES:
            df = self._candles_to_dataframe(candle_data.get(tf, []))
            if len(df) > 0:
                ctx_snapshots.append(compute_all_indicators(df, symbol, tf))

        # Score each execution timeframe
        best_signal: SignalScore | None = None

        for tf, snapshot in exec_snapshots.items():
            if snapshot.current_price == 0:
                continue

            signal = self._score_setup(snapshot, ctx_snapshots, symbol)

            if signal and signal.confluence_score >= MIN_CONFLUENCE_SCORE:
                if best_signal is None or signal.confluence_score > best_signal.confluence_score:
                    best_signal = signal

        # Build result
        scan = {
            "instrument": symbol,
            "price": exec_snapshots.get("1h", IndicatorSnapshot(symbol, "1h", 0)).current_price,
            "best_score": best_signal.confluence_score if best_signal else 0,
            "direction": best_signal.direction if best_signal else "NEUTRAL",
        }

        signal_dict = None
        if best_signal:
            self.logger.info(
                "Setup detected",
                extra={
                    "instrument": symbol,
                    "timeframe": best_signal.timeframe,
                    "direction": best_signal.direction,
                    "confluence": best_signal.confluence_score,
                },
            )
            signal_dict = self._build_signal_output(best_signal)

        return {"scan": scan, "signal": signal_dict}

    def _score_setup(
        self,
        snapshot: IndicatorSnapshot,
        ctx_snapshots: list[IndicatorSnapshot],
        symbol: str,
    ) -> SignalScore | None:
        """Score a potential setup and return SignalScore if viable."""

        # 1. Score trend → determines direction
        trend_score, direction, trend_reasons = score_trend(snapshot)
        if direction == "NEUTRAL":
            return None

        # 2. Score momentum aligned to direction
        momentum_score, momentum_reasons = score_momentum(snapshot, direction)

        # 3. Score volatility
        volatility_score, volatility_reasons = score_volatility(snapshot, direction)

        # 4. Score volume
        volume_score, volume_reasons = score_volume(snapshot, direction)

        # 5. Score structure
        structure_score, structure_reasons = score_structure(snapshot, direction)

        # 6. Score multi-timeframe alignment
        mtf_score, mtf_reasons = score_mtf_alignment(snapshot, ctx_snapshots, direction)

        # Weighted confluence score
        confluence = (
            trend_score * SIGNAL_WEIGHTS["trend"]
            + momentum_score * SIGNAL_WEIGHTS["momentum"]
            + volatility_score * SIGNAL_WEIGHTS["volatility"]
            + volume_score * SIGNAL_WEIGHTS["volume"]
            + structure_score * SIGNAL_WEIGHTS["structure"]
            + mtf_score * SIGNAL_WEIGHTS["mtf_alignment"]
        ) / 100  # Normalize (weights sum to 100)

        # Compute trade parameters
        params = compute_trade_params(snapshot, direction, snapshot.instrument)

        all_reasons = trend_reasons + momentum_reasons + volatility_reasons + \
                      volume_reasons + structure_reasons + mtf_reasons

        return SignalScore(
            instrument=snapshot.instrument,
            timeframe=snapshot.timeframe,
            direction=direction,
            confluence_score=round(confluence, 1),
            breakdown={
                "trend": round(trend_score, 1),
                "momentum": round(momentum_score, 1),
                "volatility": round(volatility_score, 1),
                "volume": round(volume_score, 1),
                "structure": round(structure_score, 1),
                "mtf_alignment": round(mtf_score, 1),
            },
            entry=params["entry"],
            stop_loss=params["stop_loss"],
            take_profit=params["take_profit"],
            take_profit_2=params["take_profit_2"],
            atr=params["atr"],
            indicators={
                "rsi": snapshot.momentum.rsi,
                "macd_cross": snapshot.momentum.macd_cross,
                "ema_alignment": snapshot.trend.ema_alignment,
                "adx": snapshot.trend.adx,
                "ichimoku": snapshot.trend.ichimoku_signal,
                "supertrend": snapshot.trend.supertrend_direction,
                "bb_position": snapshot.volatility.bb_position,
                "squeeze": snapshot.volatility.squeeze,
                "vwap": snapshot.volume.price_vs_vwap,
                "mfi": snapshot.volume.mfi,
                "obv_trend": snapshot.volume.obv_trend,
                "higher_tf_trend": f"{ctx_snapshots[0].trend.ema_alignment} ({ctx_snapshots[0].timeframe})" if ctx_snapshots else "N/A",
            },
            patterns=snapshot.structure.candlestick_patterns,
            reasoning_parts=all_reasons,
        )

    def _build_signal_output(self, score: SignalScore) -> dict[str, Any]:
        """Build the structured signal output (TechnicalSignal format)."""
        return {
            "agent": self.name,
            "instrument": score.instrument,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": "BUY" if score.direction == "LONG" else "SELL",
            "direction": score.direction,
            "confidence": round(score.confluence_score / 100, 2),
            "timeframe": score.timeframe,
            "entry": score.entry,
            "stop_loss": score.stop_loss,
            "take_profit": score.take_profit,
            "take_profit_2": score.take_profit_2,
            "atr": score.atr,
            "confluence_score": score.confluence_score,
            "score_breakdown": score.breakdown,
            "indicators": score.indicators,
            "patterns": score.patterns,
            "reasoning": " | ".join(score.reasoning_parts),
        }

    @staticmethod
    def _candles_to_dataframe(candles: list) -> pd.DataFrame:
        """Convert Candle model list to pandas DataFrame."""
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        records = []
        for c in candles:
            records.append({
                "ts": c.ts if hasattr(c, "ts") else c.get("ts"),
                "open": float(c.open if hasattr(c, "open") else c.get("open", 0)),
                "high": float(c.high if hasattr(c, "high") else c.get("high", 0)),
                "low": float(c.low if hasattr(c, "low") else c.get("low", 0)),
                "close": float(c.close if hasattr(c, "close") else c.get("close", 0)),
                "volume": float(c.volume if hasattr(c, "volume") else c.get("volume", 0)),
            })

        df = pd.DataFrame(records)
        if "ts" in df.columns:
            df = df.sort_values("ts").reset_index(drop=True)
        return df

    @property
    def stats(self) -> dict[str, Any]:
        base = super().stats
        base["signals_generated"] = self._signals_generated
        return base
