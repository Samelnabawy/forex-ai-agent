"""
Technical Indicators Computation Engine.

Pure computation layer — no database, no agent logic.
Takes OHLCV DataFrames, returns indicator values.

Full indicator stack (20+ indicators across 6 categories):

TREND:
  - EMA(20, 50, 200)
  - ADX(14)
  - Ichimoku Cloud (9, 26, 52)
  - Supertrend(10, 3)

MOMENTUM:
  - RSI(14)
  - MACD(12, 26, 9)
  - Stochastic(14, 3, 3)
  - Williams %R(14)
  - CCI(20)

VOLATILITY:
  - Bollinger Bands(20, 2)
  - ATR(14)
  - Keltner Channels(20, 1.5)

VOLUME:
  - VWAP
  - OBV
  - MFI(14)
  - Volume Profile (simplified)

STRUCTURE:
  - Support/Resistance (swing highs/lows)
  - Fibonacci Retracements
  - Pivot Points (daily, weekly)
  - Candlestick Patterns

MULTI-TIMEFRAME:
  - Higher TF trend alignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────

class TrendDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalType(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


# ── Data Containers ───────────────────────────────────────────

@dataclass
class TrendIndicators:
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    adx: float | None = None
    adx_plus_di: float | None = None
    adx_minus_di: float | None = None
    ema_alignment: str = ""          # "bullish", "bearish", "mixed"
    ichimoku_tenkan: float | None = None
    ichimoku_kijun: float | None = None
    ichimoku_senkou_a: float | None = None
    ichimoku_senkou_b: float | None = None
    ichimoku_chikou: float | None = None
    ichimoku_signal: str = ""        # "above_cloud", "below_cloud", "in_cloud"
    supertrend: float | None = None
    supertrend_direction: str = ""   # "bullish", "bearish"


@dataclass
class MomentumIndicators:
    rsi: float | None = None
    rsi_signal: str = ""             # "oversold", "overbought", "neutral"
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    macd_cross: str = ""             # "bullish", "bearish", "none"
    stoch_k: float | None = None
    stoch_d: float | None = None
    stoch_signal: str = ""           # "oversold", "overbought", "neutral"
    williams_r: float | None = None
    williams_signal: str = ""        # "oversold", "overbought", "neutral"
    cci: float | None = None
    cci_signal: str = ""             # "oversold", "overbought", "neutral"


@dataclass
class VolatilityIndicators:
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    bb_position: str = ""            # "above_upper", "below_lower", "middle"
    atr: float | None = None
    atr_pct: float | None = None     # ATR as % of price
    keltner_upper: float | None = None
    keltner_middle: float | None = None
    keltner_lower: float | None = None
    squeeze: bool = False            # BB inside Keltner = low vol squeeze


@dataclass
class VolumeIndicators:
    vwap: float | None = None
    price_vs_vwap: str = ""          # "above", "below"
    obv: float | None = None
    obv_trend: str = ""              # "rising", "falling", "flat"
    mfi: float | None = None
    mfi_signal: str = ""             # "oversold", "overbought", "neutral"


@dataclass
class StructureIndicators:
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    fib_levels: dict[str, float] = field(default_factory=dict)  # "0.236", "0.382", etc.
    daily_pivot: float | None = None
    daily_r1: float | None = None
    daily_r2: float | None = None
    daily_r3: float | None = None
    daily_s1: float | None = None
    daily_s2: float | None = None
    daily_s3: float | None = None
    weekly_pivot: float | None = None
    candlestick_patterns: list[str] = field(default_factory=list)  # detected patterns


@dataclass
class IndicatorSnapshot:
    """Complete indicator snapshot for one instrument at one timeframe."""
    instrument: str
    timeframe: str
    current_price: float
    trend: TrendIndicators = field(default_factory=TrendIndicators)
    momentum: MomentumIndicators = field(default_factory=MomentumIndicators)
    volatility: VolatilityIndicators = field(default_factory=VolatilityIndicators)
    volume: VolumeIndicators = field(default_factory=VolumeIndicators)
    structure: StructureIndicators = field(default_factory=StructureIndicators)


# ── Computation Functions ─────────────────────────────────────

def compute_trend(df: pd.DataFrame) -> TrendIndicators:
    """Compute all trend indicators from OHLCV DataFrame."""
    t = TrendIndicators()
    if len(df) < 200:
        return t

    close = df["close"]

    # EMAs
    t.ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    t.ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    t.ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

    # EMA alignment
    if t.ema_20 > t.ema_50 > t.ema_200:
        t.ema_alignment = "bullish"
    elif t.ema_20 < t.ema_50 < t.ema_200:
        t.ema_alignment = "bearish"
    else:
        t.ema_alignment = "mixed"

    # ADX
    adx_data = _compute_adx(df, period=14)
    t.adx = adx_data["adx"]
    t.adx_plus_di = adx_data["plus_di"]
    t.adx_minus_di = adx_data["minus_di"]

    # Ichimoku Cloud
    ichimoku = _compute_ichimoku(df)
    t.ichimoku_tenkan = ichimoku["tenkan"]
    t.ichimoku_kijun = ichimoku["kijun"]
    t.ichimoku_senkou_a = ichimoku["senkou_a"]
    t.ichimoku_senkou_b = ichimoku["senkou_b"]
    t.ichimoku_chikou = ichimoku["chikou"]

    price = close.iloc[-1]
    if t.ichimoku_senkou_a and t.ichimoku_senkou_b:
        cloud_top = max(t.ichimoku_senkou_a, t.ichimoku_senkou_b)
        cloud_bottom = min(t.ichimoku_senkou_a, t.ichimoku_senkou_b)
        if price > cloud_top:
            t.ichimoku_signal = "above_cloud"
        elif price < cloud_bottom:
            t.ichimoku_signal = "below_cloud"
        else:
            t.ichimoku_signal = "in_cloud"

    # Supertrend
    st = _compute_supertrend(df, period=10, multiplier=3.0)
    t.supertrend = st["value"]
    t.supertrend_direction = st["direction"]

    return t


def compute_momentum(df: pd.DataFrame) -> MomentumIndicators:
    """Compute all momentum indicators."""
    m = MomentumIndicators()
    if len(df) < 30:
        return m

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI(14)
    m.rsi = _compute_rsi(close, period=14)
    if m.rsi is not None:
        if m.rsi < 30:
            m.rsi_signal = "oversold"
        elif m.rsi > 70:
            m.rsi_signal = "overbought"
        else:
            m.rsi_signal = "neutral"

    # MACD(12, 26, 9)
    macd_data = _compute_macd(close, fast=12, slow=26, signal=9)
    m.macd = macd_data["macd"]
    m.macd_signal = macd_data["signal"]
    m.macd_histogram = macd_data["histogram"]
    m.macd_cross = macd_data["cross"]

    # Stochastic(14, 3, 3)
    stoch = _compute_stochastic(high, low, close, k_period=14, d_period=3, smooth=3)
    m.stoch_k = stoch["k"]
    m.stoch_d = stoch["d"]
    if m.stoch_k is not None:
        if m.stoch_k < 20:
            m.stoch_signal = "oversold"
        elif m.stoch_k > 80:
            m.stoch_signal = "overbought"
        else:
            m.stoch_signal = "neutral"

    # Williams %R(14)
    m.williams_r = _compute_williams_r(high, low, close, period=14)
    if m.williams_r is not None:
        if m.williams_r < -80:
            m.williams_signal = "oversold"
        elif m.williams_r > -20:
            m.williams_signal = "overbought"
        else:
            m.williams_signal = "neutral"

    # CCI(20)
    m.cci = _compute_cci(high, low, close, period=20)
    if m.cci is not None:
        if m.cci < -100:
            m.cci_signal = "oversold"
        elif m.cci > 100:
            m.cci_signal = "overbought"
        else:
            m.cci_signal = "neutral"

    return m


def compute_volatility(df: pd.DataFrame) -> VolatilityIndicators:
    """Compute all volatility indicators."""
    v = VolatilityIndicators()
    if len(df) < 20:
        return v

    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = close.iloc[-1]

    # Bollinger Bands(20, 2)
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    v.bb_upper = (sma + 2 * std).iloc[-1]
    v.bb_middle = sma.iloc[-1]
    v.bb_lower = (sma - 2 * std).iloc[-1]
    if v.bb_upper and v.bb_lower:
        v.bb_width = (v.bb_upper - v.bb_lower) / v.bb_middle if v.bb_middle else 0
        if price > v.bb_upper:
            v.bb_position = "above_upper"
        elif price < v.bb_lower:
            v.bb_position = "below_lower"
        else:
            v.bb_position = "middle"

    # ATR(14)
    v.atr = _compute_atr(high, low, close, period=14)
    if v.atr and price:
        v.atr_pct = (v.atr / price) * 100

    # Keltner Channels(20, 1.5)
    ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    atr_val = v.atr or 0
    v.keltner_upper = ema_20 + 1.5 * atr_val
    v.keltner_middle = ema_20
    v.keltner_lower = ema_20 - 1.5 * atr_val

    # Squeeze detection: BB inside Keltner
    if v.bb_upper and v.keltner_upper and v.bb_lower and v.keltner_lower:
        v.squeeze = v.bb_upper < v.keltner_upper and v.bb_lower > v.keltner_lower

    return v


def compute_volume(df: pd.DataFrame) -> VolumeIndicators:
    """Compute all volume indicators."""
    vol = VolumeIndicators()
    if len(df) < 20 or "volume" not in df.columns:
        return vol

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    price = close.iloc[-1]

    # VWAP (session-based approximation)
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    vwap_series = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    vol.vwap = vwap_series.iloc[-1] if not pd.isna(vwap_series.iloc[-1]) else None

    if vol.vwap:
        vol.price_vs_vwap = "above" if price > vol.vwap else "below"

    # OBV
    obv = _compute_obv(close, volume)
    vol.obv = obv["value"]
    vol.obv_trend = obv["trend"]

    # MFI(14)
    vol.mfi = _compute_mfi(high, low, close, volume, period=14)
    if vol.mfi is not None:
        if vol.mfi < 20:
            vol.mfi_signal = "oversold"
        elif vol.mfi > 80:
            vol.mfi_signal = "overbought"
        else:
            vol.mfi_signal = "neutral"

    return vol


def compute_structure(
    df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
    weekly_df: pd.DataFrame | None = None,
) -> StructureIndicators:
    """Compute support/resistance, fibs, pivots, and candlestick patterns."""
    s = StructureIndicators()
    if len(df) < 50:
        return s

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    price = close.iloc[-1]

    # Support/Resistance from swing highs/lows
    s.support_levels, s.resistance_levels = _find_sr_levels(high, low, close, lookback=50)

    # Nearest levels
    supports_below = [l for l in s.support_levels if l < price]
    resistances_above = [l for l in s.resistance_levels if l > price]
    s.nearest_support = max(supports_below) if supports_below else None
    s.nearest_resistance = min(resistances_above) if resistances_above else None

    # Fibonacci retracements (from recent swing high/low)
    s.fib_levels = _compute_fibonacci(high, low, lookback=100)

    # Pivot Points (daily)
    if daily_df is not None and len(daily_df) >= 2:
        prev = daily_df.iloc[-2]
        pivot = (prev["high"] + prev["low"] + prev["close"]) / 3
        s.daily_pivot = pivot
        s.daily_r1 = 2 * pivot - prev["low"]
        s.daily_s1 = 2 * pivot - prev["high"]
        s.daily_r2 = pivot + (prev["high"] - prev["low"])
        s.daily_s2 = pivot - (prev["high"] - prev["low"])
        s.daily_r3 = prev["high"] + 2 * (pivot - prev["low"])
        s.daily_s3 = prev["low"] - 2 * (prev["high"] - pivot)

    # Pivot Points (weekly)
    if weekly_df is not None and len(weekly_df) >= 2:
        prev_w = weekly_df.iloc[-2]
        s.weekly_pivot = (prev_w["high"] + prev_w["low"] + prev_w["close"]) / 3

    # Candlestick patterns
    s.candlestick_patterns = _detect_candlestick_patterns(open_, high, low, close)

    return s


def compute_all_indicators(
    df: pd.DataFrame,
    instrument: str,
    timeframe: str,
    daily_df: pd.DataFrame | None = None,
    weekly_df: pd.DataFrame | None = None,
) -> IndicatorSnapshot:
    """Compute the full indicator stack for one instrument/timeframe."""
    if len(df) < 5:
        return IndicatorSnapshot(
            instrument=instrument,
            timeframe=timeframe,
            current_price=0.0,
        )

    price = float(df["close"].iloc[-1])

    return IndicatorSnapshot(
        instrument=instrument,
        timeframe=timeframe,
        current_price=price,
        trend=compute_trend(df),
        momentum=compute_momentum(df),
        volatility=compute_volatility(df),
        volume=compute_volume(df),
        structure=compute_structure(df, daily_df, weekly_df),
    )


# ── Internal Computation Helpers ──────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> float | None:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, Any]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_val = macd_line.iloc[-1]
    signal_val = signal_line.iloc[-1]
    hist_val = histogram.iloc[-1]

    # Detect cross
    cross = "none"
    if len(histogram) >= 2:
        prev_hist = histogram.iloc[-2]
        if prev_hist <= 0 and hist_val > 0:
            cross = "bullish"
        elif prev_hist >= 0 and hist_val < 0:
            cross = "bearish"

    return {
        "macd": float(macd_val) if not pd.isna(macd_val) else None,
        "signal": float(signal_val) if not pd.isna(signal_val) else None,
        "histogram": float(hist_val) if not pd.isna(hist_val) else None,
        "cross": cross,
    }


def _compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
) -> dict[str, float | None]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return {
        "k": float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else None,
        "d": float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else None,
    }


def _compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float | None:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    val = wr.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _compute_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> float | None:
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (typical_price - sma) / (0.015 * mad).replace(0, np.nan)
    val = cci.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _compute_adx(df: pd.DataFrame, period: int = 14) -> dict[str, float | None]:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return {
        "adx": float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None,
        "plus_di": float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else None,
        "minus_di": float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else None,
    }


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float | None:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    val = atr.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _compute_ichimoku(df: pd.DataFrame) -> dict[str, float | None]:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Tenkan-sen (conversion line): (9-period high + 9-period low) / 2
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2

    # Kijun-sen (base line): (26-period high + 26-period low) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # Senkou Span A: (tenkan + kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)

    # Senkou Span B: (52-period high + 52-period low) / 2, shifted 26 periods ahead
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    # Chikou Span: current close, shifted 26 periods back
    chikou = close.shift(-26)

    return {
        "tenkan": float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else None,
        "kijun": float(kijun.iloc[-1]) if not pd.isna(kijun.iloc[-1]) else None,
        "senkou_a": float(senkou_a.iloc[-1]) if len(senkou_a.dropna()) > 0 and not pd.isna(senkou_a.iloc[-1]) else None,
        "senkou_b": float(senkou_b.iloc[-1]) if len(senkou_b.dropna()) > 0 and not pd.isna(senkou_b.iloc[-1]) else None,
        "chikou": float(close.iloc[-1]),  # Current close as chikou reference
    }


def _compute_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> dict[str, Any]:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    atr = _compute_atr(high, low, close, period) or 0
    hl2 = (high + low) / 2

    upper_band = hl2.iloc[-1] + multiplier * atr
    lower_band = hl2.iloc[-1] - multiplier * atr

    # Simple direction: if close > upper band → bullish
    price = close.iloc[-1]
    if price > upper_band:
        return {"value": lower_band, "direction": "bullish"}
    elif price < lower_band:
        return {"value": upper_band, "direction": "bearish"}
    else:
        # Use previous close for context
        prev_close = close.iloc[-2] if len(close) > 1 else price
        direction = "bullish" if price > prev_close else "bearish"
        return {"value": lower_band if direction == "bullish" else upper_band, "direction": direction}


def _compute_obv(close: pd.Series, volume: pd.Series) -> dict[str, Any]:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (volume * direction).cumsum()

    # Trend detection on OBV
    val = obv.iloc[-1]
    if len(obv) >= 10:
        obv_sma = obv.rolling(10).mean().iloc[-1]
        if val > obv_sma:
            trend = "rising"
        elif val < obv_sma:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "flat"

    return {"value": float(val), "trend": trend}


def _compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> float | None:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    tp_diff = typical_price.diff()

    pos_flow = money_flow.where(tp_diff > 0, 0.0).rolling(period).sum()
    neg_flow = money_flow.where(tp_diff <= 0, 0.0).rolling(period).sum()

    mfi = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))
    val = mfi.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _find_sr_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 50,
    tolerance_pct: float = 0.001,
) -> tuple[list[float], list[float]]:
    """Find support and resistance from swing highs/lows."""
    recent_high = high.tail(lookback)
    recent_low = low.tail(lookback)
    price = close.iloc[-1]

    # Swing highs: high > both neighbors
    swing_highs = []
    swing_lows = []

    for i in range(2, len(recent_high) - 2):
        h = recent_high.iloc[i]
        if h > recent_high.iloc[i - 1] and h > recent_high.iloc[i - 2] and \
           h > recent_high.iloc[i + 1] and h > recent_high.iloc[i + 2]:
            swing_highs.append(float(h))

        l = recent_low.iloc[i]
        if l < recent_low.iloc[i - 1] and l < recent_low.iloc[i - 2] and \
           l < recent_low.iloc[i + 1] and l < recent_low.iloc[i + 2]:
            swing_lows.append(float(l))

    # Cluster nearby levels
    supports = _cluster_levels(swing_lows, tolerance_pct)
    resistances = _cluster_levels(swing_highs, tolerance_pct)

    return sorted(supports), sorted(resistances)


def _cluster_levels(levels: list[float], tolerance_pct: float = 0.001) -> list[float]:
    """Cluster nearby price levels into single levels."""
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]

    for level in sorted_levels[1:]:
        if abs(level - clusters[-1][-1]) / clusters[-1][-1] < tolerance_pct:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    return [sum(c) / len(c) for c in clusters]


def _compute_fibonacci(
    high: pd.Series, low: pd.Series, lookback: int = 100
) -> dict[str, float]:
    """Compute Fibonacci retracement levels from recent swing."""
    recent_high = high.tail(lookback).max()
    recent_low = low.tail(lookback).min()
    diff = recent_high - recent_low

    return {
        "0.0": float(recent_high),
        "0.236": float(recent_high - 0.236 * diff),
        "0.382": float(recent_high - 0.382 * diff),
        "0.5": float(recent_high - 0.5 * diff),
        "0.618": float(recent_high - 0.618 * diff),
        "0.786": float(recent_high - 0.786 * diff),
        "1.0": float(recent_low),
    }


def _detect_candlestick_patterns(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> list[str]:
    """Detect common candlestick patterns on the last few candles."""
    patterns: list[str] = []
    if len(close) < 3:
        return patterns

    # Last 3 candles
    o1, o2, o3 = open_.iloc[-3], open_.iloc[-2], open_.iloc[-1]
    h1, h2, h3 = high.iloc[-3], high.iloc[-2], high.iloc[-1]
    l1, l2, l3 = low.iloc[-3], low.iloc[-2], low.iloc[-1]
    c1, c2, c3 = close.iloc[-3], close.iloc[-2], close.iloc[-1]

    body3 = abs(c3 - o3)
    range3 = h3 - l3
    if range3 == 0:
        return patterns

    body_ratio = body3 / range3

    # Doji
    if body_ratio < 0.1:
        patterns.append("doji")

    # Hammer (bullish reversal)
    lower_wick = min(o3, c3) - l3
    upper_wick = h3 - max(o3, c3)
    if lower_wick > 2 * body3 and upper_wick < body3 * 0.5:
        patterns.append("hammer")

    # Shooting star (bearish reversal)
    if upper_wick > 2 * body3 and lower_wick < body3 * 0.5:
        patterns.append("shooting_star")

    # Engulfing (bullish)
    if c2 < o2 and c3 > o3 and c3 > o2 and o3 < c2:
        patterns.append("bullish_engulfing")

    # Engulfing (bearish)
    if c2 > o2 and c3 < o3 and c3 < o2 and o3 > c2:
        patterns.append("bearish_engulfing")

    # Morning star (bullish)
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    if c1 < o1 and body2 < body1 * 0.3 and c3 > o3 and c3 > (o1 + c1) / 2:
        patterns.append("morning_star")

    # Evening star (bearish)
    if c1 > o1 and body2 < body1 * 0.3 and c3 < o3 and c3 < (o1 + c1) / 2:
        patterns.append("evening_star")

    # Three white soldiers
    if c1 > o1 and c2 > o2 and c3 > o3 and c2 > c1 and c3 > c2:
        patterns.append("three_white_soldiers")

    # Three black crows
    if c1 < o1 and c2 < o2 and c3 < o3 and c2 < c1 and c3 < c2:
        patterns.append("three_black_crows")

    return patterns
