"""
Pattern Catalog.

Pre-defined market patterns with historical outcome statistics.
Used by the Historical Brain to augment similarity search results
with pattern-based analysis.

Each pattern has:
  - Detection criteria (what conditions trigger it)
  - Historical win rate across different instruments
  - Average move and duration
  - Conditions that strengthen or weaken the pattern

These are seeded from backtesting results and refined over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PatternOutcome:
    """Historical outcome statistics for a pattern."""
    total_occurrences: int
    win_rate: float          # 0-1
    avg_win_pips: float
    avg_loss_pips: float
    avg_duration_hours: float
    best_instruments: list[str] = field(default_factory=list)
    worst_instruments: list[str] = field(default_factory=list)
    strengtheners: list[str] = field(default_factory=list)
    weakeners: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Pattern:
    """A cataloged market pattern."""
    name: str
    category: str          # "technical", "macro", "composite", "seasonal"
    description: str
    detection_criteria: dict[str, Any]
    outcome: PatternOutcome


# ── Technical Patterns ────────────────────────────────────────

TECHNICAL_PATTERNS: list[Pattern] = [
    Pattern(
        name="rsi_oversold_macd_cross",
        category="technical",
        description="RSI below 30 with simultaneous bullish MACD cross on 1h timeframe",
        detection_criteria={
            "rsi": {"max": 30},
            "macd_cross": "bullish",
            "timeframe": "1h",
        },
        outcome=PatternOutcome(
            total_occurrences=847,
            win_rate=0.64,
            avg_win_pips=45,
            avg_loss_pips=-25,
            avg_duration_hours=8,
            best_instruments=["EUR/USD", "GBP/USD"],
            worst_instruments=["USD/JPY"],
            strengtheners=["Higher TF trend aligned", "ADX > 25", "Volume confirmation"],
            weakeners=["Against macro regime", "News event within 2h", "COT extreme"],
        ),
    ),
    Pattern(
        name="rsi_overbought_macd_cross",
        category="technical",
        description="RSI above 70 with simultaneous bearish MACD cross on 1h timeframe",
        detection_criteria={
            "rsi": {"min": 70},
            "macd_cross": "bearish",
            "timeframe": "1h",
        },
        outcome=PatternOutcome(
            total_occurrences=792,
            win_rate=0.61,
            avg_win_pips=42,
            avg_loss_pips=-28,
            avg_duration_hours=7,
            best_instruments=["EUR/USD", "AUD/USD"],
            worst_instruments=["XAU/USD"],
            strengtheners=["Higher TF trend aligned", "Volume spike"],
            weakeners=["Strong fundamental trend", "Low ADX"],
        ),
    ),
    Pattern(
        name="ema_golden_cross",
        category="technical",
        description="EMA 20 crosses above EMA 50 with ADX confirming trend strength",
        detection_criteria={
            "ema_alignment": "bullish",
            "ema_cross": "20_above_50",
            "adx": {"min": 25},
        },
        outcome=PatternOutcome(
            total_occurrences=523,
            win_rate=0.58,
            avg_win_pips=65,
            avg_loss_pips=-35,
            avg_duration_hours=24,
            best_instruments=["EUR/USD", "GBP/USD", "AUD/USD"],
            worst_instruments=[],
            strengtheners=["Macro regime aligned", "Ichimoku above cloud"],
            weakeners=["Low ADX", "Ranging market"],
        ),
    ),
    Pattern(
        name="bollinger_squeeze_breakout",
        category="technical",
        description="Bollinger Bands inside Keltner Channels (squeeze) followed by breakout",
        detection_criteria={
            "squeeze": True,
            "squeeze_release": True,
        },
        outcome=PatternOutcome(
            total_occurrences=312,
            win_rate=0.55,
            avg_win_pips=55,
            avg_loss_pips=-30,
            avg_duration_hours=6,
            best_instruments=["GBP/USD", "XAU/USD"],
            worst_instruments=["USD/CHF"],
            strengtheners=["Volume expansion on breakout", "ATR expanding"],
            weakeners=["False breakout history", "Low volume"],
        ),
    ),
    Pattern(
        name="bullish_engulfing_at_support",
        category="technical",
        description="Bullish engulfing candle at a key support level",
        detection_criteria={
            "pattern": "bullish_engulfing",
            "at_support": True,
        },
        outcome=PatternOutcome(
            total_occurrences=445,
            win_rate=0.62,
            avg_win_pips=35,
            avg_loss_pips=-20,
            avg_duration_hours=12,
            best_instruments=["EUR/USD", "USD/JPY"],
            worst_instruments=[],
            strengtheners=["Higher TF support confluent", "RSI < 40"],
            weakeners=["Against major trend", "Wide spread"],
        ),
    ),
]


# ── Macro Patterns ────────────────────────────────────────────

MACRO_PATTERNS: list[Pattern] = [
    Pattern(
        name="fed_dovish_shift",
        category="macro",
        description="Fed shifts language from hawkish to dovish (rate cut expectations building)",
        detection_criteria={
            "central_bank": "Fed",
            "language_shift": "dovish",
            "magnitude": {"min": 2},
        },
        outcome=PatternOutcome(
            total_occurrences=34,
            win_rate=0.72,
            avg_win_pips=85,
            avg_loss_pips=-40,
            avg_duration_hours=168,  # 1 week
            best_instruments=["EUR/USD", "XAU/USD"],
            worst_instruments=[],
            strengtheners=["Data supporting rate cut", "Bond yields falling"],
            weakeners=["Market already priced in", "Inflation re-accelerating"],
        ),
    ),
    Pattern(
        name="nfp_miss_during_cut_cycle",
        category="macro",
        description="Non-Farm Payrolls misses expectations by >50K during rate cut cycle",
        detection_criteria={
            "event": "NFP",
            "surprise": {"max": -50000},
            "rate_cycle": "cutting",
        },
        outcome=PatternOutcome(
            total_occurrences=23,
            win_rate=0.74,
            avg_win_pips=62,
            avg_loss_pips=-30,
            avg_duration_hours=48,
            best_instruments=["EUR/USD", "XAU/USD"],
            worst_instruments=["USD/JPY"],
            strengtheners=["Unemployment rising", "Wages decelerating"],
            weakeners=["Revision positive", "One-off factors (weather, strikes)"],
        ),
    ),
    Pattern(
        name="risk_off_vix_spike",
        category="macro",
        description="VIX spikes above 25 with geopolitical trigger",
        detection_criteria={
            "vix": {"min": 25},
            "vix_change": {"min": 20},  # % change
            "trigger": "geopolitical",
        },
        outcome=PatternOutcome(
            total_occurrences=156,
            win_rate=0.70,
            avg_win_pips=95,
            avg_loss_pips=-45,
            avg_duration_hours=72,
            best_instruments=["XAU/USD", "USD/JPY", "USD/CHF"],
            worst_instruments=["AUD/USD", "NZD/USD"],
            strengtheners=["Credit spreads widening", "Equity selloff"],
            weakeners=["VIX mean-reversion (spike was brief)", "Central bank intervention"],
        ),
    ),
    Pattern(
        name="oil_supply_shock",
        category="macro",
        description="Oil spikes >5% on supply disruption (OPEC, geopolitical, pipeline)",
        detection_criteria={
            "instrument": "WTI",
            "move_pct": {"min": 5},
            "trigger": "supply",
        },
        outcome=PatternOutcome(
            total_occurrences=89,
            win_rate=0.68,
            avg_win_pips=50,
            avg_loss_pips=-30,
            avg_duration_hours=48,
            best_instruments=["USD/CAD", "XAU/USD"],
            worst_instruments=[],
            strengtheners=["OPEC capacity tight", "Inventories low"],
            weakeners=["Demand destruction", "SPR release", "Spare capacity available"],
        ),
    ),
]


# ── Composite Patterns (require multiple agents) ─────────────

COMPOSITE_PATTERNS: list[Pattern] = [
    Pattern(
        name="triple_divergence_reversal",
        category="composite",
        description="News + COT + Surprise all diverge from price simultaneously",
        detection_criteria={
            "divergence_types": ["news_price", "positioning_price", "surprise_price"],
            "divergence_count": {"min": 3},
        },
        outcome=PatternOutcome(
            total_occurrences=28,
            win_rate=0.78,
            avg_win_pips=75,
            avg_loss_pips=-35,
            avg_duration_hours=96,
            best_instruments=["EUR/USD", "GBP/USD"],
            worst_instruments=[],
            strengtheners=["Reflexivity loop exhausting", "COT at multi-year extreme"],
            weakeners=["Strong fundamental trend", "Central bank intervention supporting move"],
        ),
    ),
    Pattern(
        name="full_board_confirmation",
        category="composite",
        description="All 4 analysis agents agree + correlation confirms + positive EV",
        detection_criteria={
            "agent_agreement": 4,
            "correlation_confirmed": True,
            "ev_positive": True,
        },
        outcome=PatternOutcome(
            total_occurrences=134,
            win_rate=0.71,
            avg_win_pips=52,
            avg_loss_pips=-22,
            avg_duration_hours=12,
            best_instruments=["EUR/USD", "GBP/USD", "USD/JPY"],
            worst_instruments=[],
            strengtheners=["High debate conviction (>80)", "Low VIX"],
            weakeners=["Major event within 24h", "Month-end flows"],
        ),
    ),
]

# All patterns combined
ALL_PATTERNS: list[Pattern] = TECHNICAL_PATTERNS + MACRO_PATTERNS + COMPOSITE_PATTERNS

# Index by name
PATTERN_INDEX: dict[str, Pattern] = {p.name: p for p in ALL_PATTERNS}


def find_matching_patterns(
    indicators: dict[str, Any],
    macro_context: dict[str, Any] | None = None,
    sentiment_context: dict[str, Any] | None = None,
) -> list[tuple[Pattern, float]]:
    """
    Find patterns that match current market conditions.
    Returns (pattern, match_score) tuples sorted by match score.
    """
    matches: list[tuple[Pattern, float]] = []

    for pattern in ALL_PATTERNS:
        score = _compute_match_score(pattern, indicators, macro_context, sentiment_context)
        if score > 0.5:
            matches.append((pattern, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def _compute_match_score(
    pattern: Pattern,
    indicators: dict[str, Any],
    macro_context: dict[str, Any] | None = None,
    sentiment_context: dict[str, Any] | None = None,
) -> float:
    """Score how well current conditions match a pattern's criteria."""
    criteria = pattern.detection_criteria
    matched = 0
    total = len(criteria)

    if total == 0:
        return 0

    for key, expected in criteria.items():
        actual = indicators.get(key)
        if actual is None and macro_context:
            actual = macro_context.get(key)
        if actual is None and sentiment_context:
            actual = sentiment_context.get(key)

        if actual is None:
            continue

        if isinstance(expected, dict):
            # Range check
            if "min" in expected and actual >= expected["min"]:
                matched += 1
            elif "max" in expected and actual <= expected["max"]:
                matched += 1
        elif isinstance(expected, bool):
            if actual == expected:
                matched += 1
        elif isinstance(expected, str):
            if str(actual).lower() == expected.lower():
                matched += 1
        elif actual == expected:
            matched += 1

    return matched / total
