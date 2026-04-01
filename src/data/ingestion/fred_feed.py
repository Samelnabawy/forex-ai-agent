"""
FRED (Federal Reserve Economic Data) feed.

Fetches:
  - Bond yields (US 2Y, 10Y, 30Y + key foreign equivalents)
  - Yield curve shape (2Y-10Y spread)
  - Economic indicators (GDP, CPI, NFP, PMI, Retail Sales)
  - Interest rate differentials between economies
  - Money supply (M2)
  - Real interest rates (TIPS yields)

Source: FRED API (free, 120 requests/minute)
Used by: Macro Analyst Agent, Historical Brain
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# ── Series IDs ────────────────────────────────────────────────

# Bond Yields
YIELD_SERIES: dict[str, str] = {
    # US Treasury
    "US_2Y": "DGS2",
    "US_10Y": "DGS10",
    "US_30Y": "DGS30",
    # TIPS (real yields)
    "US_TIPS_5Y": "DFII5",
    "US_TIPS_10Y": "DFII10",
    # Fed Funds Rate
    "FED_FUNDS": "FEDFUNDS",
    # Policy rates (approximated via market rates where direct unavailable)
    "ECB_RATE": "ECBDFR",          # ECB deposit facility rate
    "BOE_RATE": "BOERUKM",         # BOE official bank rate
    "BOJ_RATE": "IRSTCI01JPM156N", # Japan short-term rate
}

# Economic Indicators
ECONOMIC_SERIES: dict[str, dict[str, str]] = {
    "US": {
        "GDP": "GDP",
        "CPI": "CPIAUCSL",
        "CPI_YOY": "CPIAUCNS",
        "CORE_CPI": "CPILFESL",
        "NFP": "PAYEMS",
        "UNEMPLOYMENT": "UNRATE",
        "PMI_MFG": "MANEMP",
        "RETAIL_SALES": "RSXFS",
        "M2": "M2SL",
        "CONSUMER_CONFIDENCE": "UMCSENT",
    },
    "EU": {
        "CPI_YOY": "CP0000EZ19M086NEST",
        "UNEMPLOYMENT": "LRHUTTTTEZM156S",
        "GDP": "CLVMNACSCAB1GQEA19",
    },
    "GB": {
        "CPI_YOY": "GBRCPIALLMINMEI",
        "UNEMPLOYMENT": "LMUNRRTTGBM156S",
        "GDP": "CLVMNACSCAB1GQUK",
    },
    "JP": {
        "CPI_YOY": "JPNCPIALLMINMEI",
        "GDP": "JPNRGDPEXP",
    },
    "AU": {
        "CPI_YOY": "AUSCPIALLQINMEI",
        "UNEMPLOYMENT": "LRUNTTTTAUM156S",
    },
    "CA": {
        "CPI_YOY": "CANCPIALLMINMEI",
        "UNEMPLOYMENT": "LRUNTTTTCAM156S",
    },
    "CN": {
        # China data — critical for AUD/NZD
        "PMI_MFG": "MPMICTCNM050S",
    },
}


class FREDFeed:
    """Fetches economic data from the FRED API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.fred_api_key.get_secret_value()
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, dict[str, Any]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _fetch_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch a single FRED series."""
        if not self._api_key:
            logger.warning("FRED API key not configured")
            return []

        if not observation_start:
            observation_start = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

        client = await self._get_client()
        try:
            resp = await client.get(
                f"{FRED_BASE_URL}/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": self._api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": limit,
                    "observation_start": observation_start,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("observations", [])
        except Exception as e:
            logger.error("FRED fetch failed", extra={"series": series_id, "error": str(e)})
            return []

    async def get_latest_value(self, series_id: str) -> float | None:
        """Get the most recent value for a FRED series."""
        observations = await self._fetch_series(series_id, limit=5)
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                try:
                    return float(val)
                except ValueError:
                    continue
        return None

    # ── Bond Yields ───────────────────────────────────────────

    async def get_bond_yields(self) -> dict[str, float | None]:
        """Fetch all tracked bond yields."""
        yields: dict[str, float | None] = {}
        for name, series_id in YIELD_SERIES.items():
            yields[name] = await self.get_latest_value(series_id)
        return yields

    async def get_yield_curve(self) -> dict[str, Any]:
        """Compute yield curve shape and key metrics."""
        yields = await self.get_bond_yields()

        us_2y = yields.get("US_2Y")
        us_10y = yields.get("US_10Y")
        us_30y = yields.get("US_30Y")

        result: dict[str, Any] = {
            "yields": yields,
            "spread_2y_10y": None,
            "spread_10y_30y": None,
            "curve_shape": "unknown",
            "inversion": False,
        }

        if us_2y is not None and us_10y is not None:
            spread = us_10y - us_2y
            result["spread_2y_10y"] = round(spread, 3)
            result["inversion"] = spread < 0

            if spread < -0.5:
                result["curve_shape"] = "deeply_inverted"
            elif spread < 0:
                result["curve_shape"] = "inverted"
            elif spread < 0.5:
                result["curve_shape"] = "flat"
            elif spread < 1.5:
                result["curve_shape"] = "normal"
            else:
                result["curve_shape"] = "steep"

        if us_10y is not None and us_30y is not None:
            result["spread_10y_30y"] = round(us_30y - us_10y, 3)

        return result

    # ── Rate Differentials ────────────────────────────────────

    async def get_rate_differentials(self) -> dict[str, dict[str, Any]]:
        """
        Compute interest rate differentials between USD and other currencies.
        This is the core driver of carry trades and medium-term FX trends.

        Positive differential = USD has higher rate = USD bullish (all else equal)
        """
        fed_rate = await self.get_latest_value("FEDFUNDS")
        ecb_rate = await self.get_latest_value("ECBDFR")
        boe_rate = await self.get_latest_value("BOERUKM")
        boj_rate = await self.get_latest_value("IRSTCI01JPM156N")

        differentials: dict[str, dict[str, Any]] = {}

        if fed_rate is not None:
            if ecb_rate is not None:
                diff = fed_rate - ecb_rate
                differentials["EUR/USD"] = {
                    "differential": round(diff, 3),
                    "usd_rate": fed_rate,
                    "counter_rate": ecb_rate,
                    "carry_direction": "USD" if diff > 0 else "EUR",
                    "magnitude": abs(diff),
                }

            if boe_rate is not None:
                diff = fed_rate - boe_rate
                differentials["GBP/USD"] = {
                    "differential": round(diff, 3),
                    "usd_rate": fed_rate,
                    "counter_rate": boe_rate,
                    "carry_direction": "USD" if diff > 0 else "GBP",
                    "magnitude": abs(diff),
                }

            if boj_rate is not None:
                diff = fed_rate - boj_rate
                differentials["USD/JPY"] = {
                    "differential": round(diff, 3),
                    "usd_rate": fed_rate,
                    "counter_rate": boj_rate,
                    "carry_direction": "USD" if diff > 0 else "JPY",
                    "magnitude": abs(diff),
                }

        return differentials

    # ── Real Interest Rates ───────────────────────────────────

    async def get_real_rates(self) -> dict[str, float | None]:
        """
        Fetch TIPS yields (real interest rates).
        Gold trades inversely to real rates — this is the primary XAU driver.
        """
        tips_5y = await self.get_latest_value("DFII5")
        tips_10y = await self.get_latest_value("DFII10")
        return {
            "US_REAL_5Y": tips_5y,
            "US_REAL_10Y": tips_10y,
        }

    # ── Economic Indicators ───────────────────────────────────

    async def get_economic_snapshot(self, country: str = "US") -> dict[str, float | None]:
        """Get latest values for all economic indicators of a country."""
        series_map = ECONOMIC_SERIES.get(country, {})
        snapshot: dict[str, float | None] = {}
        for name, series_id in series_map.items():
            snapshot[name] = await self.get_latest_value(series_id)
        return snapshot

    async def get_china_pmi(self) -> float | None:
        """Get China manufacturing PMI — critical for AUD/NZD."""
        return await self.get_latest_value("MPMICTCNM050S")

    # ── Full Macro Data Package ───────────────────────────────

    async def get_full_macro_data(self) -> dict[str, Any]:
        """
        Fetch everything the Macro Agent needs in one call.
        Bundles yields, curve, differentials, real rates, and key economic data.
        """
        yield_curve = await self.get_yield_curve()
        differentials = await self.get_rate_differentials()
        real_rates = await self.get_real_rates()

        # Key economic indicators
        us_data = await self.get_economic_snapshot("US")
        china_pmi = await self.get_china_pmi()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "yield_curve": yield_curve,
            "rate_differentials": differentials,
            "real_rates": real_rates,
            "economic_data": {
                "US": us_data,
                "CN_PMI": china_pmi,
            },
            "interpretation": {
                "curve_shape": yield_curve.get("curve_shape", "unknown"),
                "inverted": yield_curve.get("inversion", False),
                "gold_signal": self._interpret_gold_signal(real_rates),
            },
        }

    @staticmethod
    def _interpret_gold_signal(real_rates: dict[str, float | None]) -> str:
        """Interpret real rates for gold direction."""
        real_10y = real_rates.get("US_REAL_10Y")
        if real_10y is None:
            return "unknown"
        if real_10y < 0:
            return "bullish"  # negative real rates = gold bullish
        elif real_10y < 1.0:
            return "neutral"
        else:
            return "bearish"  # high real rates = gold bearish

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
