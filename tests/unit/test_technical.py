"""
Unit tests for Technical Analyst Agent and indicator engine.
Tests pure computation — no database or Redis dependency.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.indicators import (
    IndicatorSnapshot,
    compute_all_indicators,
    compute_momentum,
    compute_structure,
    compute_trend,
    compute_volatility,
    compute_volume,
    _compute_rsi,
    _compute_macd,
    _compute_atr,
    _compute_stochastic,
    _compute_williams_r,
    _compute_cci,
    _compute_fibonacci,
    _detect_candlestick_patterns,
    _cluster_levels,
)
from src.agents.technical import (
    MIN_CONFLUENCE_SCORE,
    score_trend,
    score_momentum,
    score_volatility,
    score_volume,
    score_structure,
    compute_trade_params,
    TechnicalAnalystAgent,
)


# ── Test Data Generators ──────────────────────────────────────

def make_trending_up_df(n: int = 250, start_price: float = 1.0800) -> pd.DataFrame:
    """Generate a DataFrame with a clear uptrend."""
    np.random.seed(42)
    prices = [start_price]
    for _ in range(n - 1):
        change = np.random.normal(0.0002, 0.0005)  # slight upward bias
        prices.append(prices[-1] + change)

    close = pd.Series(prices)
    high = close + np.random.uniform(0.0005, 0.0015, n)
    low = close - np.random.uniform(0.0005, 0.0015, n)
    open_ = close.shift(1).fillna(start_price)
    volume = pd.Series(np.random.uniform(100, 1000, n))

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_trending_down_df(n: int = 250, start_price: float = 1.0800) -> pd.DataFrame:
    """Generate a DataFrame with a clear downtrend."""
    np.random.seed(42)
    prices = [start_price]
    for _ in range(n - 1):
        change = np.random.normal(-0.0002, 0.0005)  # slight downward bias
        prices.append(prices[-1] + change)

    close = pd.Series(prices)
    high = close + np.random.uniform(0.0005, 0.0015, n)
    low = close - np.random.uniform(0.0005, 0.0015, n)
    open_ = close.shift(1).fillna(start_price)
    volume = pd.Series(np.random.uniform(100, 1000, n))

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_ranging_df(n: int = 250, center: float = 1.0800) -> pd.DataFrame:
    """Generate a DataFrame with sideways/ranging price action."""
    np.random.seed(42)
    prices = center + np.random.normal(0, 0.0010, n)
    close = pd.Series(prices)
    high = close + np.random.uniform(0.0005, 0.0015, n)
    low = close - np.random.uniform(0.0005, 0.0015, n)
    open_ = close.shift(1).fillna(center)
    volume = pd.Series(np.random.uniform(100, 1000, n))

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ── Indicator Engine Tests ────────────────────────────────────

class TestRSI:
    def test_rsi_in_range(self) -> None:
        df = make_trending_up_df()
        rsi = _compute_rsi(df["close"])
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_trending_up_above_50(self) -> None:
        df = make_trending_up_df()
        rsi = _compute_rsi(df["close"])
        assert rsi is not None
        assert rsi > 50

    def test_rsi_trending_down_below_50(self) -> None:
        df = make_trending_down_df()
        rsi = _compute_rsi(df["close"])
        assert rsi is not None
        assert rsi < 50

    def test_rsi_insufficient_data(self) -> None:
        short = pd.Series([1.0, 1.1, 1.2])
        rsi = _compute_rsi(short, period=14)
        # With < 14 data points, rolling mean returns NaN
        assert rsi is None or pd.isna(rsi)


class TestMACD:
    def test_macd_returns_all_fields(self) -> None:
        df = make_trending_up_df()
        result = _compute_macd(df["close"])
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        assert "cross" in result
        assert result["macd"] is not None
        assert result["signal"] is not None

    def test_macd_cross_detection(self) -> None:
        df = make_trending_up_df()
        result = _compute_macd(df["close"])
        assert result["cross"] in ("bullish", "bearish", "none")


class TestATR:
    def test_atr_positive(self) -> None:
        df = make_trending_up_df()
        atr = _compute_atr(df["high"], df["low"], df["close"])
        assert atr is not None
        assert atr > 0

    def test_atr_scales_with_volatility(self) -> None:
        # Low volatility
        df_low = make_ranging_df()
        atr_low = _compute_atr(df_low["high"], df_low["low"], df_low["close"])

        # High volatility (wider candles)
        df_high = make_ranging_df()
        df_high["high"] = df_high["close"] + 0.005
        df_high["low"] = df_high["close"] - 0.005
        atr_high = _compute_atr(df_high["high"], df_high["low"], df_high["close"])

        assert atr_low is not None and atr_high is not None
        assert atr_high > atr_low


class TestStochastic:
    def test_stochastic_in_range(self) -> None:
        df = make_trending_up_df()
        result = _compute_stochastic(df["high"], df["low"], df["close"])
        assert result["k"] is not None
        assert 0 <= result["k"] <= 100
        assert result["d"] is not None
        assert 0 <= result["d"] <= 100


class TestWilliamsR:
    def test_williams_in_range(self) -> None:
        df = make_trending_up_df()
        wr = _compute_williams_r(df["high"], df["low"], df["close"])
        assert wr is not None
        assert -100 <= wr <= 0


class TestCCI:
    def test_cci_returns_value(self) -> None:
        df = make_trending_up_df()
        cci = _compute_cci(df["high"], df["low"], df["close"])
        assert cci is not None


class TestFibonacci:
    def test_fib_levels_count(self) -> None:
        df = make_trending_up_df()
        fibs = _compute_fibonacci(df["high"], df["low"])
        assert len(fibs) == 7
        assert "0.382" in fibs
        assert "0.618" in fibs

    def test_fib_levels_ordered(self) -> None:
        df = make_trending_up_df()
        fibs = _compute_fibonacci(df["high"], df["low"])
        assert fibs["0.0"] > fibs["0.236"] > fibs["0.382"] > fibs["0.5"] > fibs["0.618"] > fibs["1.0"]


class TestCandlestickPatterns:
    def test_bullish_engulfing(self) -> None:
        # Create a bullish engulfing pattern
        open_ = pd.Series([1.08, 1.07, 1.06])
        high = pd.Series([1.085, 1.075, 1.085])
        low = pd.Series([1.075, 1.065, 1.055])
        close = pd.Series([1.075, 1.065, 1.08])

        patterns = _detect_candlestick_patterns(open_, high, low, close)
        assert "bullish_engulfing" in patterns

    def test_doji(self) -> None:
        open_ = pd.Series([1.08, 1.07, 1.0700])
        high = pd.Series([1.085, 1.075, 1.0750])
        low = pd.Series([1.075, 1.065, 1.0650])
        close = pd.Series([1.075, 1.065, 1.0701])  # Close ≈ Open

        patterns = _detect_candlestick_patterns(open_, high, low, close)
        assert "doji" in patterns


class TestClusterLevels:
    def test_clusters_nearby_levels(self) -> None:
        levels = [1.0800, 1.0801, 1.0802, 1.0900, 1.0901]
        clustered = _cluster_levels(levels, tolerance_pct=0.001)
        assert len(clustered) == 2  # Two clusters

    def test_empty_input(self) -> None:
        assert _cluster_levels([]) == []


# ── Full Indicator Stack Tests ────────────────────────────────

class TestComputeAll:
    def test_returns_snapshot(self) -> None:
        df = make_trending_up_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        assert isinstance(snapshot, IndicatorSnapshot)
        assert snapshot.instrument == "EUR/USD"
        assert snapshot.timeframe == "1h"
        assert snapshot.current_price > 0

    def test_trend_computed(self) -> None:
        df = make_trending_up_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        assert snapshot.trend.ema_20 is not None
        assert snapshot.trend.ema_50 is not None
        assert snapshot.trend.ema_200 is not None
        assert snapshot.trend.adx is not None

    def test_momentum_computed(self) -> None:
        df = make_trending_up_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        assert snapshot.momentum.rsi is not None
        assert snapshot.momentum.macd is not None

    def test_volatility_computed(self) -> None:
        df = make_trending_up_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        assert snapshot.volatility.atr is not None
        assert snapshot.volatility.bb_upper is not None
        assert snapshot.volatility.bb_lower is not None

    def test_insufficient_data_returns_empty(self) -> None:
        df = pd.DataFrame({
            "open": [1.08], "high": [1.085], "low": [1.075],
            "close": [1.08], "volume": [100]
        })
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        assert snapshot.current_price == 0.0


# ── Signal Scoring Tests ──────────────────────────────────────

class TestScoreTrend:
    def test_uptrend_scores_bullish(self) -> None:
        df = make_trending_up_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        score, direction, reasons = score_trend(snapshot)
        assert direction == "LONG"
        assert score > 0
        assert len(reasons) > 0

    def test_downtrend_scores_bearish(self) -> None:
        df = make_trending_down_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        score, direction, reasons = score_trend(snapshot)
        assert direction == "SHORT"
        assert score > 0


class TestTradeParams:
    def test_long_params(self) -> None:
        df = make_trending_up_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        params = compute_trade_params(snapshot, "LONG", "EUR/USD")
        assert params["entry"] > 0
        assert params["stop_loss"] < params["entry"]
        assert params["take_profit"] > params["entry"]

    def test_short_params(self) -> None:
        df = make_trending_down_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        params = compute_trade_params(snapshot, "SHORT", "EUR/USD")
        assert params["entry"] > 0
        assert params["stop_loss"] > params["entry"]
        assert params["take_profit"] < params["entry"]

    def test_risk_reward_minimum(self) -> None:
        df = make_trending_up_df()
        snapshot = compute_all_indicators(df, "EUR/USD", "1h")
        params = compute_trade_params(snapshot, "LONG", "EUR/USD")
        risk = params["entry"] - params["stop_loss"]
        reward = params["take_profit"] - params["entry"]
        rr = reward / risk if risk > 0 else 0
        assert rr >= 1.5  # Minimum R:R from risk rules


class TestCandleToDataframe:
    def test_converts_dict_candles(self) -> None:
        agent = TechnicalAnalystAgent()
        candles = [
            {"ts": "2026-04-01T14:00:00Z", "open": 1.08, "high": 1.085, "low": 1.075, "close": 1.082, "volume": 100},
            {"ts": "2026-04-01T15:00:00Z", "open": 1.082, "high": 1.09, "low": 1.08, "close": 1.088, "volume": 150},
        ]
        df = agent._candles_to_dataframe(candles)
        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns

    def test_empty_candles(self) -> None:
        agent = TechnicalAnalystAgent()
        df = agent._candles_to_dataframe([])
        assert len(df) == 0
