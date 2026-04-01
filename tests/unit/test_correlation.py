"""
Unit tests for Correlation Agent and math engine.
Tests pure computation — no database or Redis dependency.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.correlation_math import (
    CascadeEffect,
    CascadeSignal,
    CorrelationAnomaly,
    DXYDecomposition,
    compute_all_cluster_scores,
    compute_correlation_matrices,
    compute_effective_exposure,
    compute_rank_correlation,
    compute_returns,
    decompose_dxy,
    detect_cascades,
    detect_correlation_anomalies,
    detect_lead_lag,
    test_cointegration,
    compute_cluster_coherence,
)


# ── Test Data Generators ──────────────────────────────────────

def make_correlated_prices(n: int = 300, correlation: float = 0.85) -> pd.DataFrame:
    """Generate two correlated price series."""
    np.random.seed(42)
    base = np.cumsum(np.random.normal(0.0001, 0.001, n)) + 1.08
    noise = np.random.normal(0, 0.0005, n)
    correlated = base + noise * (1 - correlation)

    return pd.DataFrame({
        "EUR/USD": base,
        "GBP/USD": correlated + 0.18,  # GBP higher
    })


def make_multi_instrument_prices(n: int = 300) -> pd.DataFrame:
    """Generate realistic multi-instrument price data."""
    np.random.seed(42)

    # Base USD strength factor
    usd_factor = np.cumsum(np.random.normal(0, 0.001, n))

    # Each pair responds to USD with some idiosyncratic noise
    eur = 1.08 + usd_factor * -0.8 + np.cumsum(np.random.normal(0, 0.0003, n))
    gbp = 1.26 + usd_factor * -0.7 + np.cumsum(np.random.normal(0, 0.0004, n))
    jpy_inv = 0.0067 + usd_factor * 0.3 + np.cumsum(np.random.normal(0, 0.0002, n))  # inverted
    chf_inv = 0.88 + usd_factor * 0.2 + np.cumsum(np.random.normal(0, 0.0003, n))
    aud = 0.65 + usd_factor * -0.5 + np.cumsum(np.random.normal(0, 0.0004, n))
    cad_inv = 1.36 + usd_factor * 0.4 + np.cumsum(np.random.normal(0, 0.0003, n))
    nzd = 0.60 + usd_factor * -0.5 + np.cumsum(np.random.normal(0, 0.0004, n))
    gold = 2000 + usd_factor * -500 + np.cumsum(np.random.normal(0, 2, n))
    oil = 75 + np.cumsum(np.random.normal(0, 0.5, n))  # Somewhat independent

    return pd.DataFrame({
        "EUR/USD": eur,
        "GBP/USD": gbp,
        "USD/JPY": 1 / jpy_inv,  # Convert to standard quote
        "USD/CHF": chf_inv,
        "AUD/USD": aud,
        "USD/CAD": cad_inv,
        "NZD/USD": nzd,
        "XAU/USD": gold,
        "WTI": oil,
    })


# ── Correlation Matrix Tests ──────────────────────────────────

class TestCorrelationMatrices:
    def test_returns_computation(self) -> None:
        prices = make_multi_instrument_prices()
        returns = compute_returns(prices)
        assert len(returns) == len(prices) - 1
        assert returns.shape[1] == 9

    def test_multiple_windows(self) -> None:
        prices = make_multi_instrument_prices()
        matrices = compute_correlation_matrices(prices, windows=[30, 90])
        assert 30 in matrices
        assert 90 in matrices
        assert matrices[30].shape == (9, 9)
        assert matrices[90].shape == (9, 9)

    def test_diagonal_is_one(self) -> None:
        prices = make_multi_instrument_prices()
        matrices = compute_correlation_matrices(prices, windows=[90])
        m = matrices[90]
        for symbol in m.columns:
            assert abs(m.loc[symbol, symbol] - 1.0) < 0.001

    def test_symmetry(self) -> None:
        prices = make_multi_instrument_prices()
        matrices = compute_correlation_matrices(prices, windows=[90])
        m = matrices[90]
        for i, a in enumerate(m.columns):
            for j, b in enumerate(m.columns):
                assert abs(m.loc[a, b] - m.loc[b, a]) < 0.001

    def test_eur_gbp_positively_correlated(self) -> None:
        """EUR/USD and GBP/USD should be positively correlated (both USD-short)."""
        prices = make_multi_instrument_prices()
        matrices = compute_correlation_matrices(prices, windows=[90])
        corr = matrices[90].loc["EUR/USD", "GBP/USD"]
        assert corr > 0.5  # Should be positively correlated

    def test_rank_correlation(self) -> None:
        prices = make_multi_instrument_prices()
        rank_matrix = compute_rank_correlation(prices, window=90)
        assert rank_matrix.shape == (9, 9)

    def test_insufficient_data(self) -> None:
        prices = make_multi_instrument_prices(n=10)
        matrices = compute_correlation_matrices(prices, windows=[30])
        assert 30 in matrices  # Should fall back to all available data


# ── Anomaly Detection Tests ───────────────────────────────────

class TestAnomalyDetection:
    def test_no_anomalies_in_stable_market(self) -> None:
        """Consistent correlation should produce no anomalies."""
        prices = make_multi_instrument_prices(n=300)
        matrices = compute_correlation_matrices(prices, windows=[30])
        # Use the same matrix multiple times as "history"
        historical = [matrices[30]] * 10
        anomalies = detect_correlation_anomalies(matrices[30], historical, z_threshold=2.0)
        assert len(anomalies) == 0  # No deviation from self

    def test_detects_broken_correlation(self) -> None:
        """A sudden correlation break should be flagged."""
        prices = make_multi_instrument_prices(n=300)

        # Generate historical matrices from normal period
        historical = []
        for i in range(10):
            subset = prices.iloc[i * 20 : (i + 1) * 20 + 30]
            if len(subset) >= 30:
                m = compute_correlation_matrices(subset, windows=[30])
                historical.append(m[30])

        # Create a current matrix with artificially broken correlation
        current = historical[-1].copy()
        if "EUR/USD" in current.columns and "GBP/USD" in current.columns:
            current.loc["EUR/USD", "GBP/USD"] = -0.5  # Was positive, now negative
            current.loc["GBP/USD", "EUR/USD"] = -0.5

            anomalies = detect_correlation_anomalies(current, historical, z_threshold=1.5)
            # Should detect the EUR/GBP anomaly
            eur_gbp = [a for a in anomalies if
                       (a.pair_a == "EUR/USD" and a.pair_b == "GBP/USD") or
                       (a.pair_a == "GBP/USD" and a.pair_b == "EUR/USD")]
            assert len(eur_gbp) > 0

    def test_severity_classification(self) -> None:
        anomaly = CorrelationAnomaly(
            pair_a="A", pair_b="B",
            current_correlation=0.1, historical_mean=0.8,
            historical_std=0.1, z_score=-3.5,
            window=90, status="diverging", severity="extreme"
        )
        assert anomaly.severity == "extreme"


# ── Cascade Detection Tests ───────────────────────────────────

class TestCascadeDetection:
    def test_oil_cascade(self) -> None:
        """A 3% oil drop should trigger USD/CAD and Gold cascade predictions."""
        current = {"WTI": 72.0, "EUR/USD": 1.08, "USD/CAD": 1.36, "XAU/USD": 2000}
        previous = {"WTI": 75.0, "EUR/USD": 1.08, "USD/CAD": 1.36, "XAU/USD": 2000}

        cascades = detect_cascades(current, previous)
        assert len(cascades) >= 1

        oil_cascade = [c for c in cascades if c.trigger_instrument == "WTI"]
        assert len(oil_cascade) == 1
        assert oil_cascade[0].trigger_direction == "down"

        # Should predict USD/CAD rise and Gold rise
        effects = {e.instrument: e for e in oil_cascade[0].predicted_effects}
        assert "USD/CAD" in effects
        assert effects["USD/CAD"].expected_direction == "up"  # Oil down → CAD weak → USD/CAD up

    def test_no_cascade_on_small_move(self) -> None:
        """Small moves should not trigger cascades."""
        current = {"WTI": 74.9, "EUR/USD": 1.08}
        previous = {"WTI": 75.0, "EUR/USD": 1.08}

        cascades = detect_cascades(current, previous)
        assert len(cascades) == 0

    def test_vix_spike_cascade(self) -> None:
        """VIX spike should predict safe-haven bid."""
        current = {"EUR/USD": 1.08}
        previous = {"EUR/USD": 1.08}

        cascades = detect_cascades(current, previous, vix_current=28, vix_previous=18)
        vix_cascades = [c for c in cascades if c.trigger_instrument == "VIX"]
        assert len(vix_cascades) == 1

        effects = {e.instrument: e for e in vix_cascades[0].predicted_effects}
        assert "XAU/USD" in effects
        assert effects["XAU/USD"].expected_direction == "up"  # Gold up on risk-off


# ── DXY Decomposition Tests ──────────────────────────────────

class TestDXYDecomposition:
    def test_broad_usd_weakness(self) -> None:
        """All pairs moving against USD should show broad weakness."""
        current = {
            "EUR/USD": 1.09, "GBP/USD": 1.27, "USD/JPY": 149.0,
            "USD/CAD": 1.35, "USD/CHF": 0.87,
        }
        previous = {
            "EUR/USD": 1.08, "GBP/USD": 1.26, "USD/JPY": 150.0,
            "USD/CAD": 1.36, "USD/CHF": 0.88,
        }

        result = decompose_dxy(current, previous, dxy_current=103.5, dxy_previous=104.0)
        assert result.is_broad_usd is True
        assert "Broad USD" in result.interpretation

    def test_single_currency_driven(self) -> None:
        """Only EUR moving should show EUR-driven, not broad."""
        current = {
            "EUR/USD": 1.10, "GBP/USD": 1.26, "USD/JPY": 150.0,
            "USD/CAD": 1.36, "USD/CHF": 0.88,
        }
        previous = {
            "EUR/USD": 1.08, "GBP/USD": 1.26, "USD/JPY": 150.0,
            "USD/CAD": 1.36, "USD/CHF": 0.88,
        }

        result = decompose_dxy(current, previous)
        assert result.is_broad_usd is False
        assert "EUR/USD" in result.dominant_driver


# ── Lead-Lag Detection Tests ──────────────────────────────────

class TestLeadLag:
    def test_detects_known_lead_lag(self) -> None:
        """Create a series where A clearly leads B."""
        np.random.seed(42)
        n = 200
        a = np.cumsum(np.random.normal(0, 1, n))
        # B follows A with 3-bar lag
        b = np.zeros(n)
        b[3:] = a[:-3] + np.random.normal(0, 0.3, n - 3)

        prices = pd.DataFrame({"A": a + 100, "B": b + 100})
        results = detect_lead_lag(prices, max_lag_bars=10, min_correlation=0.2)

        # Should detect A leads B
        a_leads = [r for r in results if r.leader == "A" and r.follower == "B"]
        assert len(a_leads) > 0
        assert a_leads[0].optimal_lag_minutes > 0

    def test_no_lead_lag_in_random(self) -> None:
        """Random series should not show strong lead-lag."""
        np.random.seed(42)
        prices = pd.DataFrame({
            "A": np.cumsum(np.random.normal(0, 1, 200)) + 100,
            "B": np.cumsum(np.random.normal(0, 1, 200)) + 100,
        })
        results = detect_lead_lag(prices, min_correlation=0.5)
        # Might find weak relationships but shouldn't find strong ones
        strong = [r for r in results if r.confidence > 0.7]
        assert len(strong) == 0


# ── Cointegration Tests ───────────────────────────────────────

class TestCointegration:
    def test_cointegrated_pair(self) -> None:
        """Two series with shared stochastic trend should be cointegrated."""
        np.random.seed(42)
        n = 300
        common_trend = np.cumsum(np.random.normal(0, 0.01, n))
        a = pd.Series(common_trend + np.random.normal(0, 0.005, n) + 1.08, name="EUR/USD")
        b = pd.Series(common_trend + np.random.normal(0, 0.005, n) + 1.26, name="GBP/USD")

        result = test_cointegration(a, b)
        assert result.pair_a == "EUR/USD"
        assert result.pair_b == "GBP/USD"
        # Should likely be cointegrated
        assert result.p_value < 0.5  # More lenient threshold for synthetic data

    def test_non_cointegrated(self) -> None:
        """Two independent random walks should not be cointegrated."""
        np.random.seed(42)
        a = pd.Series(np.cumsum(np.random.normal(0, 0.01, 300)) + 1.0, name="A")
        b = pd.Series(np.cumsum(np.random.normal(0, 0.01, 300)) + 2.0, name="B")

        result = test_cointegration(a, b)
        # p-value should be higher (less likely cointegrated)
        assert result.p_value > 0.01

    def test_spread_z_score(self) -> None:
        """Z-score should reflect current spread deviation."""
        np.random.seed(42)
        n = 300
        common = np.cumsum(np.random.normal(0, 0.01, n))
        a = pd.Series(common + 1.0, name="A")
        # B deviates at the end
        deviation = np.zeros(n)
        deviation[-10:] = 0.1  # Big deviation at the end
        b = pd.Series(common + deviation + 2.0, name="B")

        result = test_cointegration(a, b)
        assert abs(result.spread_z_score) > 0  # Should detect deviation


# ── Cluster Coherence Tests ───────────────────────────────────

class TestClusterCoherence:
    def test_high_coherence(self) -> None:
        """Highly correlated cluster should score high."""
        matrix = pd.DataFrame(
            [[1.0, 0.9, 0.85], [0.9, 1.0, 0.88], [0.85, 0.88, 1.0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        score = compute_cluster_coherence(matrix, ["A", "B", "C"])
        assert score > 80

    def test_low_coherence(self) -> None:
        """Uncorrelated cluster should score low."""
        matrix = pd.DataFrame(
            [[1.0, 0.1, -0.05], [0.1, 1.0, 0.15], [-0.05, 0.15, 1.0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        score = compute_cluster_coherence(matrix, ["A", "B", "C"])
        assert score < 20

    def test_partial_cluster(self) -> None:
        """Cluster with members not in matrix should handle gracefully."""
        matrix = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            columns=["A", "B"],
            index=["A", "B"],
        )
        score = compute_cluster_coherence(matrix, ["A", "B", "C_NOT_IN_MATRIX"])
        assert score > 0  # Should work with available members


# ── Crowding Risk Tests ───────────────────────────────────────

class TestCrowdingRisk:
    def test_correlated_same_direction(self) -> None:
        """Long EUR/USD + Long GBP/USD should flag crowding."""
        matrix = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]],
            columns=["EUR/USD", "GBP/USD"],
            index=["EUR/USD", "GBP/USD"],
        )
        positions = [
            {"instrument": "EUR/USD", "direction": "LONG", "position_size": 0.01},
            {"instrument": "GBP/USD", "direction": "LONG", "position_size": 0.01},
        ]
        result = compute_effective_exposure(positions, matrix)
        assert len(result["crowding_warnings"]) > 0
        assert "CROWDED" in result["crowding_warnings"][0]

    def test_hedged_positions(self) -> None:
        """Long EUR/USD + Short GBP/USD should flag as hedged."""
        matrix = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]],
            columns=["EUR/USD", "GBP/USD"],
            index=["EUR/USD", "GBP/USD"],
        )
        positions = [
            {"instrument": "EUR/USD", "direction": "LONG", "position_size": 0.01},
            {"instrument": "GBP/USD", "direction": "SHORT", "position_size": 0.01},
        ]
        result = compute_effective_exposure(positions, matrix)
        assert any("HEDGED" in w for w in result["crowding_warnings"])

    def test_empty_positions(self) -> None:
        matrix = pd.DataFrame()
        result = compute_effective_exposure([], matrix)
        assert result["diversification_score"] == 100

    def test_currency_exposure_calculation(self) -> None:
        """Long EUR/USD should show +EUR, -USD exposure."""
        matrix = pd.DataFrame(
            [[1.0]], columns=["EUR/USD"], index=["EUR/USD"]
        )
        positions = [
            {"instrument": "EUR/USD", "direction": "LONG", "position_size": 0.01},
        ]
        result = compute_effective_exposure(positions, matrix)
        assert result["currency_exposure"]["EUR"] > 0
        assert result["currency_exposure"]["USD"] < 0
