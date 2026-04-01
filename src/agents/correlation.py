"""
Correlation Agent — Agent 03

"The Chess Player" — sees the entire board, not individual pieces.
Thinks in relationships and second-order effects.

This is the system's unique edge. Single-pair bots cannot do this.

Execution: Every 1 minute (lightweight math, no LLM).

What it does:
  1. Computes rolling correlation matrices (30/90/252 day)
  2. Detects anomalies (z-score deviation from historical norms)
  3. Predicts cascades (oil drops → front-run USD/CAD, Gold, AUD before they move)
  4. Decomposes DXY moves (broad USD or single-currency driven?)
  5. Detects lead-lag relationships (VIX leads JPY by 5-15 min)
  6. Scores cluster coherence (is the USD-weakness story confirmed across all pairs?)
  7. Cross-validates Agent 1 + Agent 2 signals
  8. Calculates crowding risk for the Risk Manager

Decision Authority:
  Can VETO trades that violate correlation logic (e.g., going long EUR/USD
  and short GBP/USD when correlation is 0.85 = doubling risk).
  Can BOOST confidence when the full board confirms a signal.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.correlation_math import (
    CascadeSignal,
    CointegrationResult,
    CorrelationAnomaly,
    DXYDecomposition,
    LeadLagResult,
    compute_all_cluster_scores,
    compute_betas,
    compute_correlation_matrices,
    compute_effective_exposure,
    compute_rank_correlation,
    decompose_dxy,
    detect_cascades,
    detect_correlation_anomalies,
    detect_lead_lag,
    test_cointegration,
)
from src.config.instruments import (
    ALL_SYMBOLS,
    CLUSTER_MEMBERS,
    CorrelationCluster,
    INSTRUMENTS,
)
from src.data.storage.cache import get_agent_state, get_all_prices, get_latest_price
from src.models import MarketState

logger = logging.getLogger(__name__)

# Pairs to test for cointegration (economically meaningful pairs)
COINTEGRATION_PAIRS: list[tuple[str, str]] = [
    ("EUR/USD", "GBP/USD"),     # European bloc
    ("AUD/USD", "NZD/USD"),     # Antipodean bloc
    ("USD/JPY", "USD/CHF"),     # Safe haven bloc
    ("AUD/USD", "XAU/USD"),     # Commodity/mining link
    ("USD/CAD", "WTI"),         # Oil link (inverse relationship)
]


class CorrelationAgent(BaseAgent):
    """
    Agent 03: Correlation Agent

    Monitors cross-instrument relationships in real time.
    Lightweight computation — runs every minute.
    """

    name: ClassVar[str] = "correlation_agent"
    description: ClassVar[str] = "Cross-instrument intelligence, cascades, divergences"
    execution_frequency: ClassVar[str] = "1m"

    def __init__(self) -> None:
        super().__init__()
        # Rolling history of correlation matrices for anomaly detection
        self._historical_matrices: list[pd.DataFrame] = []
        self._max_matrix_history = 60  # Keep last 60 snapshots
        self._last_prices: dict[str, float] = {}
        self._last_vix: float | None = None
        self._last_dxy: float | None = None

    async def analyze(self, market_state: MarketState) -> dict[str, Any]:
        """
        Core analysis: compute correlations, detect anomalies/cascades,
        cross-validate Agent 1 + Agent 2, assess crowding risk.
        """
        now = datetime.now(timezone.utc)

        # Build price DataFrame from candle close prices
        prices_df = self._build_price_dataframe(market_state)
        current_prices = self._get_current_prices(market_state)
        previous_prices = self._last_prices.copy()

        # Get DXY and VIX
        dxy_current = await self._get_supplementary("DXY")
        vix_current = await self._get_supplementary("VIX")

        result: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "instrument": None,  # Cross-instrument agent
        }

        # ── 1. Correlation Matrices ───────────────────────────
        matrices = {}
        if len(prices_df) >= 30:
            matrices = compute_correlation_matrices(prices_df)
            result["correlation_matrices"] = {
                window: matrix.to_dict() for window, matrix in matrices.items()
            }
        else:
            result["correlation_matrices"] = {}
            result["_note"] = "Insufficient data for correlation computation"

        # Store for anomaly detection history
        if 90 in matrices:
            self._historical_matrices.append(matrices[90])
            if len(self._historical_matrices) > self._max_matrix_history:
                self._historical_matrices = self._historical_matrices[-self._max_matrix_history:]

        # ── 2. Anomaly Detection ──────────────────────────────
        anomalies: list[dict[str, Any]] = []
        if 30 in matrices and len(self._historical_matrices) >= 5:
            raw_anomalies = detect_correlation_anomalies(
                current_matrix=matrices[30],
                historical_matrices=self._historical_matrices,
                z_threshold=2.0,
            )
            anomalies = [
                {
                    "pair_a": a.pair_a,
                    "pair_b": a.pair_b,
                    "current": a.current_correlation,
                    "historical_mean": a.historical_mean,
                    "z_score": a.z_score,
                    "status": a.status,
                    "severity": a.severity,
                }
                for a in raw_anomalies
            ]
        result["anomalies"] = anomalies

        # ── 3. Cascade Detection ──────────────────────────────
        cascades: list[dict[str, Any]] = []
        if current_prices and previous_prices:
            raw_cascades = detect_cascades(
                current_prices=current_prices,
                previous_prices=previous_prices,
                vix_current=vix_current,
                vix_previous=self._last_vix,
            )
            cascades = [self._cascade_to_dict(c) for c in raw_cascades]
        result["active_cascades"] = cascades

        # ── 4. DXY Decomposition ──────────────────────────────
        dxy_decomp: dict[str, Any] = {}
        if current_prices and previous_prices:
            decomp = decompose_dxy(
                current_prices=current_prices,
                previous_prices=previous_prices,
                dxy_current=dxy_current,
                dxy_previous=self._last_dxy,
            )
            dxy_decomp = {
                "dxy_change_pct": decomp.dxy_change_pct,
                "components": decomp.components,
                "dominant_driver": decomp.dominant_driver,
                "is_broad_usd": decomp.is_broad_usd,
                "interpretation": decomp.interpretation,
            }
        result["dxy_decomposition"] = dxy_decomp

        # ── 5. Cluster Coherence ──────────────────────────────
        cluster_scores: dict[str, float] = {}
        if 30 in matrices:
            cluster_scores = compute_all_cluster_scores(matrices[30])
        result["cluster_coherence"] = cluster_scores

        # ── 6. Lead-Lag Detection (periodic — expensive) ──────
        # Only compute every ~10 runs to save CPU
        lead_lag: list[dict[str, Any]] = []
        if len(prices_df) >= 100 and self._should_run_expensive():
            raw_ll = detect_lead_lag(prices_df, max_lag_bars=12, min_correlation=0.3)
            lead_lag = [
                {
                    "leader": ll.leader,
                    "follower": ll.follower,
                    "lag_minutes": ll.optimal_lag_minutes,
                    "correlation": ll.correlation_at_lag,
                    "direction": ll.direction,
                }
                for ll in raw_ll[:10]  # Top 10
            ]
        result["lead_lag_relationships"] = lead_lag

        # ── 7. Cointegration (periodic) ───────────────────────
        cointegration: list[dict[str, Any]] = []
        if len(prices_df) >= 100 and self._should_run_expensive():
            for pair_a, pair_b in COINTEGRATION_PAIRS:
                if pair_a in prices_df.columns and pair_b in prices_df.columns:
                    coint = test_cointegration(prices_df[pair_a], prices_df[pair_b])
                    if coint.is_cointegrated or abs(coint.spread_z_score) > 1.5:
                        cointegration.append({
                            "pair_a": coint.pair_a,
                            "pair_b": coint.pair_b,
                            "cointegrated": coint.is_cointegrated,
                            "spread_z_score": coint.spread_z_score,
                            "half_life": coint.half_life_bars,
                            "signal": coint.signal,
                        })
        result["cointegration"] = cointegration

        # ── 8. Cross-Validate Agent 1 + Agent 2 ──────────────
        cross_validation = await self._cross_validate_agents(
            matrices.get(30), cluster_scores, dxy_decomp, current_prices
        )
        result["cross_validation"] = cross_validation

        # ── 9. Crowding Risk ──────────────────────────────────
        crowding: dict[str, Any] = {}
        if 30 in matrices and market_state.open_trades:
            open_pos = [
                {
                    "instrument": t.instrument,
                    "direction": t.direction.value,
                    "position_size": float(t.position_size),
                }
                for t in market_state.open_trades
            ]
            crowding = compute_effective_exposure(open_pos, matrices[30])
        result["crowding_risk"] = crowding

        # ── 10. Overall Market Regime ─────────────────────────
        result["market_regime"] = self._classify_regime(
            cluster_scores, vix_current, anomalies, cascades
        )

        # Store current state for next iteration
        self._last_prices = current_prices
        self._last_vix = vix_current
        self._last_dxy = dxy_current

        # Confidence for this analysis
        data_completeness = len(current_prices) / len(ALL_SYMBOLS) if ALL_SYMBOLS else 0
        result["confidence"] = round(min(data_completeness, 1.0), 2)

        return result

    # ── Agent Cross-Validation ────────────────────────────────

    async def _cross_validate_agents(
        self,
        corr_matrix: pd.DataFrame | None,
        cluster_scores: dict[str, float],
        dxy_decomp: dict[str, Any],
        current_prices: dict[str, float],
    ) -> dict[str, Any]:
        """
        Cross-validate Technical Agent and Macro Agent signals against
        correlation evidence.

        This is where the Correlation Agent adds unique value:
        - Technical says BUY EUR/USD → does the USD cluster confirm weakness?
        - Macro says risk-off → are safe havens actually being bid?
        """
        validation: dict[str, Any] = {
            "technical_signals": [],
            "macro_regime_confirmed": None,
            "adjustments": [],
        }

        # Get Technical Agent's latest signals
        tech_state = await get_agent_state("technical_analyst")
        if tech_state and tech_state.get("signals"):
            for signal in tech_state["signals"]:
                instrument = signal.get("instrument", "")
                direction = signal.get("direction", "")
                confidence = signal.get("confidence", 0)

                assessment = self._validate_signal(
                    instrument, direction, corr_matrix, cluster_scores,
                    dxy_decomp, current_prices
                )
                validation["technical_signals"].append({
                    "instrument": instrument,
                    "direction": direction,
                    "original_confidence": confidence,
                    "correlation_assessment": assessment["verdict"],
                    "adjusted_confidence": assessment["adjusted_confidence"],
                    "reasons": assessment["reasons"],
                })

        # Validate Macro Agent's regime call
        macro_state = await get_agent_state("macro_analyst")
        if macro_state:
            macro_regime = macro_state.get("macro_regime", "neutral")
            regime_confirmed = self._validate_macro_regime(
                macro_regime, cluster_scores
            )
            validation["macro_regime_confirmed"] = regime_confirmed

        return validation

    def _validate_signal(
        self,
        instrument: str,
        direction: str,
        corr_matrix: pd.DataFrame | None,
        cluster_scores: dict[str, float],
        dxy_decomp: dict[str, Any],
        current_prices: dict[str, float],
    ) -> dict[str, Any]:
        """
        Validate a single Technical Agent signal against correlation evidence.
        Returns verdict (confirm/contradict/neutral) and adjusted confidence.
        """
        reasons: list[str] = []
        confidence_adjustment = 0.0  # -0.3 to +0.3

        inst = INSTRUMENTS.get(instrument)
        if not inst:
            return {"verdict": "unknown", "adjusted_confidence": 0.5, "reasons": ["Unknown instrument"]}

        is_long = direction == "LONG"
        is_usd_quote = inst.quote_currency == "USD"

        # 1. Check cluster confirmation
        for cluster in inst.correlation_clusters:
            score = cluster_scores.get(cluster.value, 0)
            if score > 70:
                # Cluster is moving together — check direction
                reasons.append(f"{cluster.value} cluster coherent ({score:.0f}%) — confirming")
                confidence_adjustment += 0.1
            elif score < 30:
                reasons.append(f"{cluster.value} cluster divergent ({score:.0f}%) — warning")
                confidence_adjustment -= 0.1

        # 2. Check DXY confirmation for USD pairs
        if is_usd_quote and dxy_decomp.get("is_broad_usd") is not None:
            dxy_change = dxy_decomp.get("dxy_change_pct", 0)

            if is_long and dxy_change < -0.1:
                reasons.append(f"DXY falling ({dxy_change:+.3f}%) — broad USD weakness confirms long")
                confidence_adjustment += 0.15
            elif is_long and dxy_change > 0.1:
                reasons.append(f"DXY rising ({dxy_change:+.3f}%) — USD strength contradicts long")
                confidence_adjustment -= 0.15
            elif not is_long and dxy_change > 0.1:
                reasons.append(f"DXY rising ({dxy_change:+.3f}%) — USD strength confirms short")
                confidence_adjustment += 0.15
            elif not is_long and dxy_change < -0.1:
                reasons.append(f"DXY falling ({dxy_change:+.3f}%) — USD weakness contradicts short")
                confidence_adjustment -= 0.15

            # Broad vs narrow
            if dxy_decomp.get("is_broad_usd"):
                reasons.append("Broad USD move — high confidence in direction")
                confidence_adjustment += 0.05
            else:
                driver = dxy_decomp.get("dominant_driver", "")
                if driver and driver != instrument:
                    reasons.append(f"DXY move driven by {driver}, not {instrument} — lower conviction")
                    confidence_adjustment -= 0.05

        # 3. Check correlated pair confirmation
        if corr_matrix is not None and instrument in corr_matrix.columns:
            for other_sym in ALL_SYMBOLS:
                if other_sym == instrument or other_sym not in corr_matrix.columns:
                    continue
                corr = corr_matrix.loc[instrument, other_sym]
                if abs(corr) > 0.7:
                    # Highly correlated pair — check if it confirms
                    other_price = current_prices.get(other_sym)
                    prev_other = self._last_prices.get(other_sym)
                    if other_price and prev_other and prev_other > 0:
                        other_change = (other_price - prev_other) / prev_other
                        expected_sign = 1 if (is_long and corr > 0) or (not is_long and corr < 0) else -1

                        if np.sign(other_change) == expected_sign and abs(other_change) > 0.0001:
                            reasons.append(f"{other_sym} confirming (corr {corr:.2f})")
                            confidence_adjustment += 0.05
                        elif np.sign(other_change) != expected_sign and abs(other_change) > 0.0001:
                            reasons.append(f"{other_sym} contradicting (corr {corr:.2f})")
                            confidence_adjustment -= 0.1

        # Clamp adjustment
        confidence_adjustment = max(-0.3, min(0.3, confidence_adjustment))

        # Determine verdict
        if confidence_adjustment > 0.1:
            verdict = "CONFIRMED"
        elif confidence_adjustment < -0.1:
            verdict = "CONTRADICTED"
        else:
            verdict = "NEUTRAL"

        return {
            "verdict": verdict,
            "adjusted_confidence": round(0.5 + confidence_adjustment, 2),
            "confidence_adjustment": round(confidence_adjustment, 2),
            "reasons": reasons,
        }

    def _validate_macro_regime(
        self,
        macro_regime: str,
        cluster_scores: dict[str, float],
    ) -> dict[str, Any]:
        """
        Validate Macro Agent's regime classification against correlation evidence.

        If Macro says "risk_off" but safe havens aren't being bid
        (low cluster coherence), the regime call is questionable.
        """
        safe_haven_score = cluster_scores.get("safe_haven", 50)
        commodity_score = cluster_scores.get("commodity", 50)
        usd_score = cluster_scores.get("usd_strength", 50)

        confirmed = True
        reasons: list[str] = []

        if macro_regime == "risk_off":
            if safe_haven_score < 40:
                confirmed = False
                reasons.append(f"Safe haven cluster not cohesive ({safe_haven_score:.0f}%) — JPY/CHF/Gold not confirming risk-off")
            else:
                reasons.append(f"Safe haven cluster confirming ({safe_haven_score:.0f}%)")

        elif macro_regime == "risk_on":
            if commodity_score < 40:
                reasons.append(f"Commodity cluster weak ({commodity_score:.0f}%) — risk-on not fully confirmed")
                confirmed = False
            else:
                reasons.append(f"Commodity cluster confirming ({commodity_score:.0f}%)")

        return {
            "regime": macro_regime,
            "confirmed": confirmed,
            "cluster_evidence": {
                "safe_haven": safe_haven_score,
                "commodity": commodity_score,
                "usd_strength": usd_score,
            },
            "reasons": reasons,
        }

    # ── Regime Classification ─────────────────────────────────

    def _classify_regime(
        self,
        cluster_scores: dict[str, float],
        vix: float | None,
        anomalies: list[dict[str, Any]],
        cascades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Classify current market regime from correlation evidence."""
        regime = "normal"
        confidence = 0.5
        notes: list[str] = []

        # VIX-based regime
        if vix is not None:
            if vix > 30:
                regime = "crisis"
                confidence = 0.85
                notes.append(f"VIX at {vix:.1f} — crisis conditions")
            elif vix > 25:
                regime = "risk_off"
                confidence = 0.75
                notes.append(f"VIX elevated at {vix:.1f}")
            elif vix < 14:
                regime = "complacent"
                confidence = 0.65
                notes.append(f"VIX very low at {vix:.1f} — complacent market")

        # Anomaly-based regime shift detection
        severe_anomalies = [a for a in anomalies if a.get("severity") in ("high", "extreme")]
        if len(severe_anomalies) >= 3:
            regime = "regime_shift"
            confidence = 0.70
            notes.append(f"{len(severe_anomalies)} severe correlation anomalies — possible regime change")

        # Active cascades
        if cascades:
            notes.append(f"{len(cascades)} active cascade(s) detected")

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "vix": vix,
            "notes": notes,
        }

    # ── Helpers ────────────────────────────────────────────────

    def _build_price_dataframe(self, market_state: MarketState) -> pd.DataFrame:
        """Build a price DataFrame from MarketState candles (1h close prices)."""
        series_dict: dict[str, pd.Series] = {}

        for symbol in ALL_SYMBOLS:
            candle_data = market_state.candles.get(symbol, {})
            # Prefer 1h candles for correlation (good balance of noise vs signal)
            candles = candle_data.get("1h", [])
            if not candles:
                candles = candle_data.get("4h", [])

            if candles:
                timestamps = []
                closes = []
                for c in candles:
                    ts = c.ts if hasattr(c, "ts") else c.get("ts")
                    close = float(c.close if hasattr(c, "close") else c.get("close", 0))
                    if ts and close > 0:
                        timestamps.append(ts)
                        closes.append(close)

                if timestamps:
                    s = pd.Series(closes, index=pd.DatetimeIndex(timestamps), name=symbol)
                    series_dict[symbol] = s

        if not series_dict:
            return pd.DataFrame()

        df = pd.DataFrame(series_dict)
        df = df.sort_index().dropna(how="all")
        return df

    def _get_current_prices(self, market_state: MarketState) -> dict[str, float]:
        """Extract current mid prices from MarketState."""
        prices: dict[str, float] = {}
        for symbol, tick in market_state.prices.items():
            mid = float((tick.bid + tick.ask) / 2)
            if mid > 0:
                prices[symbol] = mid
        return prices

    async def _get_supplementary(self, name: str) -> float | None:
        """Get DXY or VIX from Redis cache."""
        try:
            cached = await get_latest_price(name)
            if cached:
                return float(cached.get("mid", 0))
        except Exception:
            pass
        return None

    def _should_run_expensive(self) -> bool:
        """Run expensive computations (lead-lag, cointegration) every ~10 iterations."""
        return self._execution_count % 10 == 0

    @staticmethod
    def _cascade_to_dict(cascade: CascadeSignal) -> dict[str, Any]:
        return {
            "trigger": cascade.trigger_instrument,
            "trigger_move_pct": cascade.trigger_move_pct,
            "trigger_direction": cascade.trigger_direction,
            "effects": [
                {
                    "instrument": e.instrument,
                    "direction": e.expected_direction,
                    "magnitude_pct": e.expected_magnitude_pct,
                    "confidence": e.confidence,
                    "lag_minutes": f"{e.lag_minutes_low}-{e.lag_minutes_high}",
                    "reasoning": e.reasoning,
                }
                for e in cascade.predicted_effects
            ],
        }

    @property
    def stats(self) -> dict[str, Any]:
        base = super().stats
        base["matrix_history_size"] = len(self._historical_matrices)
        base["tracked_instruments"] = len(self._last_prices)
        return base
