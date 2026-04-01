"""
Correlation Mathematics Engine.

Pure computation — no database, no agent logic, no side effects.
Takes price series, returns statistical relationships.

Capabilities:
  - Rolling Pearson + Spearman correlation matrices (30/90/252 day)
  - Z-score anomaly detection (correlation deviation from historical norm)
  - Lead-lag detection (temporal relationships between instruments)
  - DXY decomposition (which currency drove the dollar move?)
  - Beta analysis (sensitivity of each pair to DXY, VIX, Oil, Gold)
  - Cointegration testing (Engle-Granger for mean-reversion signals)
  - Cluster strength scoring (is the USD/safe-haven/commodity cluster acting as one?)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# DXY component weights (ICE US Dollar Index)
DXY_WEIGHTS: dict[str, float] = {
    "EUR/USD": -0.576,   # Negative because EUR/USD up = USD down = DXY down
    "USD/JPY": 0.136,
    "GBP/USD": -0.119,
    "USD/CAD": 0.091,
    "USD/CHF": 0.036,
    # SEK 4.2% — we don't trade it, so we acknowledge the gap
}

# Standard correlation windows
WINDOWS = [30, 90, 252]


# ── Data Containers ───────────────────────────────────────────

@dataclass
class CorrelationSnapshot:
    """Full correlation state at a point in time."""
    timestamp: str = ""
    matrices: dict[int, pd.DataFrame] = field(default_factory=dict)  # window → NxN matrix
    anomalies: list[CorrelationAnomaly] = field(default_factory=list)
    cluster_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class CorrelationAnomaly:
    """A pair whose correlation deviates significantly from historical."""
    pair_a: str
    pair_b: str
    current_correlation: float
    historical_mean: float
    historical_std: float
    z_score: float
    window: int
    status: str  # "diverging", "converging", "regime_shift"
    severity: str  # "low", "medium", "high", "extreme"


@dataclass
class LeadLagResult:
    """Temporal relationship between two instruments."""
    leader: str
    follower: str
    optimal_lag_minutes: int
    correlation_at_lag: float
    confidence: float
    direction: str  # "positive" (move together) or "negative" (move opposite)


@dataclass
class CascadeSignal:
    """Predicted cascade from a trigger instrument."""
    trigger_instrument: str
    trigger_move_pct: float
    trigger_direction: str  # "up" or "down"
    predicted_effects: list[CascadeEffect] = field(default_factory=list)


@dataclass
class CascadeEffect:
    """Single predicted effect in a cascade."""
    instrument: str
    expected_direction: str  # "up" or "down"
    expected_magnitude_pct: float
    confidence: float
    lag_minutes_low: int
    lag_minutes_high: int
    reasoning: str


@dataclass
class DXYDecomposition:
    """Breakdown of a DXY move by currency component."""
    dxy_change_pct: float
    components: dict[str, float]  # pair → contribution to DXY move
    dominant_driver: str
    is_broad_usd: bool  # True if >3 currencies contributing
    interpretation: str


@dataclass
class BetaProfile:
    """Sensitivity of an instrument to market factors."""
    instrument: str
    beta_dxy: float | None = None
    beta_vix: float | None = None
    beta_oil: float | None = None
    beta_gold: float | None = None
    r_squared: dict[str, float] = field(default_factory=dict)


@dataclass
class CointegrationResult:
    """Cointegration test between two instruments."""
    pair_a: str
    pair_b: str
    is_cointegrated: bool
    p_value: float
    current_spread: float
    spread_z_score: float
    signal: str  # "long_a_short_b", "short_a_long_b", "neutral"
    half_life_bars: float | None = None


# ── Correlation Matrix Computation ────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price DataFrame. Columns = instruments."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_correlation_matrices(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Compute rolling Pearson correlation matrices for multiple windows.

    Args:
        prices: DataFrame with columns = instrument symbols, rows = timestamps
        windows: list of lookback periods (default: [30, 90, 252])

    Returns:
        dict mapping window → correlation matrix (DataFrame)
    """
    windows = windows or WINDOWS
    returns = compute_returns(prices)
    matrices: dict[int, pd.DataFrame] = {}

    for window in windows:
        if len(returns) < window:
            matrices[window] = returns.corr()  # Use all available data
        else:
            matrices[window] = returns.tail(window).corr()

    return matrices


def compute_rank_correlation(prices: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Spearman rank correlation — captures non-linear relationships.
    More robust to outliers than Pearson.
    """
    returns = compute_returns(prices)
    if len(returns) < window:
        return returns.corr(method="spearman")
    return returns.tail(window).corr(method="spearman")


# ── Z-Score Anomaly Detection ─────────────────────────────────

def detect_correlation_anomalies(
    current_matrix: pd.DataFrame,
    historical_matrices: list[pd.DataFrame],
    z_threshold: float = 2.0,
) -> list[CorrelationAnomaly]:
    """
    Detect pairs whose current correlation deviates significantly
    from their historical distribution.

    A z-score > 2 means the relationship has shifted beyond normal variation.
    This is how we detect regime changes in real time.
    """
    anomalies: list[CorrelationAnomaly] = []
    instruments = current_matrix.columns.tolist()

    if len(historical_matrices) < 5:
        return anomalies  # Need enough history

    for i, inst_a in enumerate(instruments):
        for j, inst_b in enumerate(instruments):
            if i >= j:
                continue  # Upper triangle only

            current = current_matrix.loc[inst_a, inst_b]
            historical_values = [m.loc[inst_a, inst_b] for m in historical_matrices
                                 if inst_a in m.columns and inst_b in m.columns]

            if len(historical_values) < 5:
                continue

            mean = np.mean(historical_values)
            std = np.std(historical_values)
            if std < 0.01:
                continue  # No meaningful variation

            z = (current - mean) / std

            if abs(z) >= z_threshold:
                # Classify the anomaly
                if abs(current) < abs(mean) - std:
                    status = "diverging"
                elif abs(current) > abs(mean) + std:
                    status = "converging"
                else:
                    status = "regime_shift"

                severity = "low"
                if abs(z) >= 3.0:
                    severity = "extreme"
                elif abs(z) >= 2.5:
                    severity = "high"
                elif abs(z) >= 2.0:
                    severity = "medium"

                anomalies.append(CorrelationAnomaly(
                    pair_a=inst_a,
                    pair_b=inst_b,
                    current_correlation=round(float(current), 4),
                    historical_mean=round(float(mean), 4),
                    historical_std=round(float(std), 4),
                    z_score=round(float(z), 2),
                    window=len(historical_values),
                    status=status,
                    severity=severity,
                ))

    # Sort by absolute z-score (most anomalous first)
    anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)
    return anomalies


# ── Lead-Lag Detection ────────────────────────────────────────

def detect_lead_lag(
    prices: pd.DataFrame,
    max_lag_bars: int = 12,
    min_correlation: float = 0.3,
) -> list[LeadLagResult]:
    """
    Detect temporal lead-lag relationships between instruments.

    Tests: if instrument A moves now, does instrument B move N bars later?
    Uses cross-correlation on returns.

    For 5-minute candles, max_lag_bars=12 tests up to 60 minutes of lag.
    """
    returns = compute_returns(prices)
    instruments = returns.columns.tolist()
    results: list[LeadLagResult] = []

    for i, leader in enumerate(instruments):
        for j, follower in enumerate(instruments):
            if i == j:
                continue

            leader_returns = returns[leader].dropna()
            follower_returns = returns[follower].dropna()

            # Align series
            aligned = pd.concat([leader_returns, follower_returns], axis=1).dropna()
            if len(aligned) < max_lag_bars + 20:
                continue

            leader_series = aligned.iloc[:, 0].values
            follower_series = aligned.iloc[:, 1].values

            best_lag = 0
            best_corr = 0.0

            for lag in range(1, max_lag_bars + 1):
                # Correlate leader[:-lag] with follower[lag:]
                if lag >= len(leader_series):
                    break
                corr = np.corrcoef(leader_series[:-lag], follower_series[lag:])[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            if abs(best_corr) >= min_correlation and best_lag > 0:
                results.append(LeadLagResult(
                    leader=leader,
                    follower=follower,
                    optimal_lag_minutes=best_lag * 5,  # Assuming 5-min bars
                    correlation_at_lag=round(float(best_corr), 4),
                    confidence=round(abs(float(best_corr)), 2),
                    direction="positive" if best_corr > 0 else "negative",
                ))

    # Sort by confidence
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


# ── Cascade Prediction ────────────────────────────────────────

# Historical cascade patterns — empirical relationships
CASCADE_RULES: list[dict[str, Any]] = [
    {
        "trigger": "WTI",
        "threshold_pct": 2.0,
        "effects": [
            {"instrument": "USD/CAD", "direction_if_up": "down", "magnitude_pct": 0.3,
             "confidence": 0.82, "lag_min": (15, 45), "reason": "Oil up → CAD strong → USD/CAD down"},
            {"instrument": "XAU/USD", "direction_if_up": "up", "magnitude_pct": 0.5,
             "confidence": 0.65, "lag_min": (30, 60), "reason": "Oil up → inflation expectations → Gold up"},
            {"instrument": "AUD/USD", "direction_if_up": "up", "magnitude_pct": 0.15,
             "confidence": 0.55, "lag_min": (30, 90), "reason": "Oil up → commodity complex bid → AUD up"},
        ],
    },
    {
        "trigger": "XAU/USD",
        "threshold_pct": 1.5,
        "effects": [
            {"instrument": "USD/CHF", "direction_if_up": "down", "magnitude_pct": 0.2,
             "confidence": 0.60, "lag_min": (10, 30), "reason": "Gold up → safe haven bid → CHF strong"},
            {"instrument": "USD/JPY", "direction_if_up": "down", "magnitude_pct": 0.15,
             "confidence": 0.55, "lag_min": (10, 30), "reason": "Gold up → risk-off → JPY strong"},
            {"instrument": "AUD/USD", "direction_if_up": "up", "magnitude_pct": 0.2,
             "confidence": 0.50, "lag_min": (30, 60), "reason": "Gold up → mining economy boost → AUD up"},
        ],
    },
    {
        "trigger": "EUR/USD",
        "threshold_pct": 0.5,
        "effects": [
            {"instrument": "GBP/USD", "direction_if_up": "up", "magnitude_pct": 0.3,
             "confidence": 0.75, "lag_min": (5, 20), "reason": "EUR/USD up → USD weakness → GBP/USD follows"},
            {"instrument": "USD/CHF", "direction_if_up": "down", "magnitude_pct": 0.25,
             "confidence": 0.70, "lag_min": (5, 15), "reason": "EUR strength → CHF follows (European bloc)"},
        ],
    },
    {
        "trigger": "USD/JPY",
        "threshold_pct": 0.5,
        "effects": [
            {"instrument": "USD/CHF", "direction_if_up": "up", "magnitude_pct": 0.2,
             "confidence": 0.60, "lag_min": (5, 20), "reason": "USD strength → CHF follows safe-haven cluster"},
            {"instrument": "XAU/USD", "direction_if_up": "up", "magnitude_pct": 0.3,
             "confidence": 0.55, "lag_min": (15, 45), "reason": "JPY weakening → less safe-haven demand → Gold softer"},
        ],
    },
]

# VIX-driven regime cascades
VIX_CASCADE: dict[str, dict[str, Any]] = {
    "spike": {  # VIX > 25 or +20% in a session
        "USD/JPY": {"direction": "down", "magnitude": 0.4, "confidence": 0.75},
        "USD/CHF": {"direction": "down", "magnitude": 0.3, "confidence": 0.65},
        "XAU/USD": {"direction": "up", "magnitude": 0.8, "confidence": 0.80},
        "AUD/USD": {"direction": "down", "magnitude": 0.3, "confidence": 0.70},
        "NZD/USD": {"direction": "down", "magnitude": 0.25, "confidence": 0.65},
    },
    "collapse": {  # VIX < 15 or -20% in a session
        "USD/JPY": {"direction": "up", "magnitude": 0.3, "confidence": 0.60},
        "AUD/USD": {"direction": "up", "magnitude": 0.25, "confidence": 0.60},
        "NZD/USD": {"direction": "up", "magnitude": 0.2, "confidence": 0.55},
        "XAU/USD": {"direction": "down", "magnitude": 0.5, "confidence": 0.55},
    },
}


def detect_cascades(
    current_prices: dict[str, float],
    previous_prices: dict[str, float],
    vix_current: float | None = None,
    vix_previous: float | None = None,
) -> list[CascadeSignal]:
    """
    Detect active cascade triggers based on recent price moves.
    Returns predicted effects with timing and confidence.
    """
    cascades: list[CascadeSignal] = []

    # Check instrument-driven cascades
    for rule in CASCADE_RULES:
        trigger = rule["trigger"]
        threshold = rule["threshold_pct"]

        curr = current_prices.get(trigger)
        prev = previous_prices.get(trigger)
        if curr is None or prev is None or prev == 0:
            continue

        move_pct = ((curr - prev) / prev) * 100

        if abs(move_pct) >= threshold:
            direction = "up" if move_pct > 0 else "down"
            effects: list[CascadeEffect] = []

            for eff in rule["effects"]:
                eff_direction = eff["direction_if_up"] if direction == "up" else (
                    "up" if eff["direction_if_up"] == "down" else "down"
                )
                effects.append(CascadeEffect(
                    instrument=eff["instrument"],
                    expected_direction=eff_direction,
                    expected_magnitude_pct=eff["magnitude_pct"],
                    confidence=eff["confidence"],
                    lag_minutes_low=eff["lag_min"][0],
                    lag_minutes_high=eff["lag_min"][1],
                    reasoning=eff["reason"],
                ))

            cascades.append(CascadeSignal(
                trigger_instrument=trigger,
                trigger_move_pct=round(move_pct, 3),
                trigger_direction=direction,
                predicted_effects=effects,
            ))

    # Check VIX-driven cascades
    if vix_current is not None and vix_previous is not None and vix_previous > 0:
        vix_change_pct = ((vix_current - vix_previous) / vix_previous) * 100

        vix_mode = None
        if vix_current > 25 or vix_change_pct > 20:
            vix_mode = "spike"
        elif vix_current < 15 or vix_change_pct < -20:
            vix_mode = "collapse"

        if vix_mode and vix_mode in VIX_CASCADE:
            effects = []
            for inst, params in VIX_CASCADE[vix_mode].items():
                effects.append(CascadeEffect(
                    instrument=inst,
                    expected_direction=params["direction"],
                    expected_magnitude_pct=params["magnitude"],
                    confidence=params["confidence"],
                    lag_minutes_low=5,
                    lag_minutes_high=30,
                    reasoning=f"VIX {vix_mode}: {vix_current:.1f} ({vix_change_pct:+.1f}%)",
                ))

            cascades.append(CascadeSignal(
                trigger_instrument="VIX",
                trigger_move_pct=round(vix_change_pct, 2),
                trigger_direction="up" if vix_change_pct > 0 else "down",
                predicted_effects=effects,
            ))

    return cascades


# ── DXY Decomposition ────────────────────────────────────────

def decompose_dxy(
    current_prices: dict[str, float],
    previous_prices: dict[str, float],
    dxy_current: float | None = None,
    dxy_previous: float | None = None,
) -> DXYDecomposition:
    """
    Decompose a DXY move into currency component contributions.

    If DXY rose 0.5%, was it because EUR fell (57.6% weight) or because
    all currencies weakened? This distinction matters for signal quality.

    Broad USD move = high confidence on all USD pairs.
    Single-currency driven = only high confidence on that pair.
    """
    dxy_change = 0.0
    if dxy_current and dxy_previous and dxy_previous > 0:
        dxy_change = ((dxy_current - dxy_previous) / dxy_previous) * 100

    components: dict[str, float] = {}
    for pair, weight in DXY_WEIGHTS.items():
        curr = current_prices.get(pair)
        prev = previous_prices.get(pair)
        if curr and prev and prev > 0:
            pair_change_pct = ((curr - prev) / prev) * 100
            # Contribution = pair change × DXY weight
            # Negative weight pairs: EUR/USD up → DXY down
            contribution = pair_change_pct * weight
            components[pair] = round(contribution, 4)

    # Identify dominant driver
    dominant = max(components, key=lambda k: abs(components[k])) if components else "unknown"

    # Is this a broad USD move?
    contributing = sum(1 for v in components.values() if abs(v) > 0.01)
    is_broad = contributing >= 3

    # Interpretation
    if is_broad:
        interpretation = f"Broad USD {'strength' if dxy_change > 0 else 'weakness'} — {contributing} currencies contributing"
    elif dominant != "unknown":
        interpretation = f"{dominant}-driven — not a broad USD move"
    else:
        interpretation = "Insufficient data for decomposition"

    return DXYDecomposition(
        dxy_change_pct=round(dxy_change, 4),
        components=components,
        dominant_driver=dominant,
        is_broad_usd=is_broad,
        interpretation=interpretation,
    )


# ── Beta Analysis ─────────────────────────────────────────────

def compute_betas(
    instrument_prices: pd.Series,
    factor_prices: dict[str, pd.Series],
    window: int = 90,
) -> BetaProfile:
    """
    Compute beta (sensitivity) of an instrument to market factors.
    Beta = covariance(instrument, factor) / variance(factor)

    Interpretation:
      beta_dxy = 1.2 → instrument moves 1.2% for every 1% DXY move
      beta_vix = -0.5 → instrument falls 0.5% for every 1% VIX rise
    """
    inst_returns = np.log(instrument_prices / instrument_prices.shift(1)).dropna().tail(window)
    profile = BetaProfile(instrument=instrument_prices.name or "unknown")

    for factor_name, factor_series in factor_prices.items():
        factor_returns = np.log(factor_series / factor_series.shift(1)).dropna().tail(window)

        # Align
        aligned = pd.concat([inst_returns, factor_returns], axis=1).dropna()
        if len(aligned) < 20:
            continue

        x = aligned.iloc[:, 1].values
        y = aligned.iloc[:, 0].values

        # Simple OLS beta
        cov = np.cov(y, x)[0, 1]
        var = np.var(x)
        beta = cov / var if var > 0 else 0

        # R-squared
        correlation = np.corrcoef(y, x)[0, 1]
        r2 = correlation ** 2

        if factor_name == "DXY":
            profile.beta_dxy = round(float(beta), 4)
        elif factor_name == "VIX":
            profile.beta_vix = round(float(beta), 4)
        elif factor_name == "OIL" or factor_name == "WTI":
            profile.beta_oil = round(float(beta), 4)
        elif factor_name == "GOLD" or factor_name == "XAU/USD":
            profile.beta_gold = round(float(beta), 4)

        profile.r_squared[factor_name] = round(float(r2), 4)

    return profile


# ── Cointegration Testing ─────────────────────────────────────

def check_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
    significance: float = 0.05,
) -> CointegrationResult:
    """
    Engle-Granger cointegration test between two price series.

    Cointegrated pairs maintain a long-run equilibrium — when the spread
    deviates, it tends to revert. This is a separate alpha source from
    correlation-based trading.

    Returns signal: if spread is >2 std from mean, suggest mean-reversion trade.
    """
    name_a = series_a.name or "A"
    name_b = series_b.name or "B"

    # Align
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 60:
        return CointegrationResult(
            pair_a=str(name_a), pair_b=str(name_b),
            is_cointegrated=False, p_value=1.0,
            current_spread=0, spread_z_score=0, signal="neutral",
        )

    a = aligned.iloc[:, 0].values
    b = aligned.iloc[:, 1].values

    # OLS regression: a = alpha + beta * b + residuals
    beta = np.cov(a, b)[0, 1] / np.var(b) if np.var(b) > 0 else 0
    alpha = np.mean(a) - beta * np.mean(b)
    spread = a - (alpha + beta * b)

    # ADF test on residuals (simplified — test if spread is stationary)
    # Using scipy for a simplified version
    spread_series = pd.Series(spread)
    spread_diff = spread_series.diff().dropna()

    if len(spread_diff) < 20:
        return CointegrationResult(
            pair_a=str(name_a), pair_b=str(name_b),
            is_cointegrated=False, p_value=1.0,
            current_spread=0, spread_z_score=0, signal="neutral",
        )

    # Simplified ADF: regress spread_diff on lagged spread
    lagged = spread_series.shift(1).iloc[1:].values
    diff = spread_diff.values

    if len(lagged) != len(diff):
        min_len = min(len(lagged), len(diff))
        lagged = lagged[:min_len]
        diff = diff[:min_len]

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(lagged, diff)

    # If slope is significantly negative → spread is mean-reverting → cointegrated
    is_coint = p_value < significance and slope < 0

    # Current spread state
    current_spread = float(spread[-1])
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread))
    z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

    # Half-life of mean reversion
    half_life = None
    if slope < 0:
        half_life = -np.log(2) / slope

    # Trading signal
    signal = "neutral"
    if is_coint:
        if z_score > 2.0:
            signal = f"short_{name_a}_long_{name_b}"
        elif z_score < -2.0:
            signal = f"long_{name_a}_short_{name_b}"

    return CointegrationResult(
        pair_a=str(name_a),
        pair_b=str(name_b),
        is_cointegrated=is_coint,
        p_value=round(float(p_value), 4),
        current_spread=round(current_spread, 6),
        spread_z_score=round(float(z_score), 2),
        half_life_bars=round(float(half_life), 1) if half_life and half_life > 0 else None,
        signal=signal,
    )


# ── Cluster Analysis ──────────────────────────────────────────

def compute_cluster_coherence(
    correlation_matrix: pd.DataFrame,
    cluster_members: list[str],
) -> float:
    """
    Score how cohesively a cluster is moving (0-100).

    High score = cluster members are highly correlated (moving as one).
    Low score = cluster is breaking apart (divergence within the cluster).

    Used to validate: "Is USD weakness broad?" "Is the safe-haven bid real?"
    """
    members_in_matrix = [m for m in cluster_members if m in correlation_matrix.columns]
    if len(members_in_matrix) < 2:
        return 0.0

    correlations: list[float] = []
    for i, a in enumerate(members_in_matrix):
        for j, b in enumerate(members_in_matrix):
            if i < j:
                corr = abs(correlation_matrix.loc[a, b])
                correlations.append(corr)

    if not correlations:
        return 0.0

    # Average absolute correlation, scaled to 0-100
    avg_corr = np.mean(correlations)
    return round(float(avg_corr * 100), 1)


def compute_all_cluster_scores(
    correlation_matrix: pd.DataFrame,
) -> dict[str, float]:
    """Score all predefined clusters."""
    from src.config.instruments import CLUSTER_MEMBERS

    scores: dict[str, float] = {}
    for cluster_name, members in CLUSTER_MEMBERS.items():
        scores[cluster_name.value] = compute_cluster_coherence(correlation_matrix, members)

    return scores


# ── Crowding Risk ─────────────────────────────────────────────

def compute_effective_exposure(
    open_positions: list[dict[str, Any]],
    correlation_matrix: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate effective portfolio exposure considering correlations.

    If we're long EUR/USD (1%) and long GBP/USD (1%) with 0.85 correlation,
    effective USD-short exposure is NOT 2% — it's more like 1.85%.
    But the RISK is higher than 2% independent positions because
    they'll both lose simultaneously.

    Returns effective exposure per currency and crowding warnings.
    """
    if not open_positions:
        return {"currency_exposure": {}, "crowding_warnings": [], "diversification_score": 100}

    # Calculate net currency exposure
    currency_exposure: dict[str, float] = {}
    for pos in open_positions:
        instrument = pos.get("instrument", "")
        direction = pos.get("direction", "LONG")
        size = pos.get("position_size", 0)

        # Parse base/quote from instrument
        parts = instrument.split("/")
        if len(parts) != 2:
            continue

        base, quote = parts
        multiplier = 1.0 if direction == "LONG" else -1.0

        currency_exposure[base] = currency_exposure.get(base, 0) + size * multiplier
        currency_exposure[quote] = currency_exposure.get(quote, 0) - size * multiplier

    # Check for correlated position crowding
    crowding_warnings: list[str] = []
    instruments = [p["instrument"] for p in open_positions if "instrument" in p]

    for i, inst_a in enumerate(instruments):
        for j, inst_b in enumerate(instruments):
            if i >= j:
                continue
            if inst_a in correlation_matrix.columns and inst_b in correlation_matrix.columns:
                corr = abs(correlation_matrix.loc[inst_a, inst_b])
                if corr > 0.7:
                    # Same direction = amplified risk
                    dir_a = open_positions[i].get("direction", "LONG")
                    dir_b = open_positions[j].get("direction", "LONG")
                    if dir_a == dir_b:
                        crowding_warnings.append(
                            f"CROWDED: {inst_a} and {inst_b} same direction, corr={corr:.2f} — effective doubled exposure"
                        )
                    else:
                        crowding_warnings.append(
                            f"HEDGED: {inst_a} and {inst_b} opposite direction, corr={corr:.2f} — partially offsetting"
                        )

    # Diversification score (higher = more diversified)
    if len(instruments) < 2:
        div_score = 100.0
    else:
        avg_corr = 0.0
        count = 0
        for i, a in enumerate(instruments):
            for j, b in enumerate(instruments):
                if i < j and a in correlation_matrix.columns and b in correlation_matrix.columns:
                    avg_corr += abs(correlation_matrix.loc[a, b])
                    count += 1
        avg_corr = avg_corr / count if count > 0 else 0
        div_score = round((1 - avg_corr) * 100, 1)

    return {
        "currency_exposure": currency_exposure,
        "crowding_warnings": crowding_warnings,
        "diversification_score": div_score,
    }
