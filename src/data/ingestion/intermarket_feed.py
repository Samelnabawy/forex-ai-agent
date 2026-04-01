"""
Intermarket Sentiment Feed.

Tracks non-forex market signals that predict currency moves:
  - Credit spreads (HY OAS) — smart money fear indicator
  - Copper/Gold ratio — real-time growth vs fear proxy
  - USD/CNH (offshore yuan) — leading indicator for AUD/NZD
  - Equity-FX correlation — structural regime detection

Sources: FRED (credit spreads), computed from existing price feeds
Used by: Sentiment Agent Layer 7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.data.storage.cache import get_latest_price

logger = logging.getLogger(__name__)

# FRED series for credit spreads
CREDIT_SERIES: dict[str, str] = {
    "HY_OAS": "BAMLH0A0HYM2",       # BofA High Yield OAS (daily)
    "IG_OAS": "BAMLC0A0CM",          # Investment Grade OAS
    "TED_SPREAD": "TEDRATE",          # TED spread (interbank stress)
}


@dataclass
class IntermarketSignals:
    """Intermarket sentiment signals."""
    timestamp: str = ""
    # Copper/Gold ratio (growth vs fear)
    copper_gold_ratio: float | None = None
    copper_gold_signal: str = "neutral"  # "risk_on", "risk_off", "neutral"
    # Credit spreads
    hy_oas: float | None = None
    credit_signal: str = "neutral"  # "stress", "normal", "complacent"
    credit_vix_divergence: bool = False  # Credit widening but VIX low = smart money nervous
    # Equity-FX
    spx_dxy_correlation: str = "normal"  # "normal" (inverse), "unusual" (both up = structural)
    # Overall intermarket risk score
    risk_score: float = 50.0  # 0 (extreme risk-off) to 100 (extreme risk-on)
    signals: list[str] = field(default_factory=list)


async def compute_intermarket_signals(
    vix: float | None = None,
    fred_data: dict[str, float | None] | None = None,
) -> IntermarketSignals:
    """
    Compute intermarket sentiment signals from available data.
    Uses cached price data from Redis where available.
    """
    result = IntermarketSignals(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    risk_points = 50.0  # Start neutral

    # 1. Copper/Gold ratio
    gold_data = await get_latest_price("XAU/USD")
    copper_price = None  # From commodity feed cache
    if fred_data:
        copper_price = fred_data.get("copper")

    if gold_data and copper_price:
        gold_price = float(gold_data.get("mid", 0))
        if gold_price > 0:
            ratio = copper_price / gold_price
            result.copper_gold_ratio = round(ratio, 6)

            # Interpretation: higher ratio = more risk appetite
            # Historical range roughly 0.003-0.007
            if ratio > 0.005:
                result.copper_gold_signal = "risk_on"
                risk_points += 15
                result.signals.append(f"Cu/Au ratio {ratio:.4f} — elevated, risk-on")
            elif ratio < 0.003:
                result.copper_gold_signal = "risk_off"
                risk_points -= 15
                result.signals.append(f"Cu/Au ratio {ratio:.4f} — depressed, risk-off")
            else:
                result.copper_gold_signal = "neutral"

    # 2. Credit spreads (from FRED data)
    if fred_data:
        hy_oas = fred_data.get("HY_OAS")
        if hy_oas is not None:
            result.hy_oas = hy_oas

            if hy_oas > 500:
                result.credit_signal = "stress"
                risk_points -= 25
                result.signals.append(f"HY OAS at {hy_oas}bps — credit stress")
            elif hy_oas > 400:
                result.credit_signal = "elevated"
                risk_points -= 10
                result.signals.append(f"HY OAS at {hy_oas}bps — elevated but not critical")
            elif hy_oas < 300:
                result.credit_signal = "complacent"
                risk_points += 10
                result.signals.append(f"HY OAS at {hy_oas}bps — tight, complacent")
            else:
                result.credit_signal = "normal"

            # Credit-VIX divergence: credit widening but VIX still low
            if vix is not None and hy_oas > 400 and vix < 18:
                result.credit_vix_divergence = True
                risk_points -= 15
                result.signals.append(
                    "DIVERGENCE: Credit spreads widening but VIX low — smart money nervous, retail complacent"
                )

    # 3. DXY-SPX relationship
    dxy_data = await get_latest_price("DXY")
    spx_data = await get_latest_price("US500")
    if dxy_data and spx_data:
        dxy_change = float(dxy_data.get("change_pct", 0))
        spx_change = float(spx_data.get("change_pct", 0))

        # Normally DXY and SPX are inversely correlated
        # Both rising simultaneously = unusual structural regime
        if dxy_change > 0.3 and spx_change > 0.3:
            result.spx_dxy_correlation = "unusual_both_up"
            result.signals.append("DXY and SPX both rising — unusual, structural shift possible")
        elif dxy_change < -0.3 and spx_change < -0.3:
            result.spx_dxy_correlation = "unusual_both_down"
            result.signals.append("DXY and SPX both falling — unusual, liquidity withdrawal?")

    # 4. VIX contribution
    if vix is not None:
        if vix > 30:
            risk_points -= 20
        elif vix > 25:
            risk_points -= 10
        elif vix < 14:
            risk_points += 10

    result.risk_score = round(max(0, min(100, risk_points)), 1)
    return result
