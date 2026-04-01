"""
Oil & Commodity data feeds.

Fetches:
  - EIA crude oil inventory (weekly, Wednesday) — WTI driver
  - OPEC meeting schedule — WTI/USD/CAD driver
  - Iron ore prices — AUD driver (Australia's #1 export)
  - Copper prices — Global growth proxy
  - Dairy prices — NZD driver (New Zealand's #1 export)

Sources: EIA API, FRED, Polygon
Used by: Macro Analyst, Correlation Agent, Sentiment Agent
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# EIA API for US oil data
EIA_BASE_URL = "https://api.eia.gov/v2"

# FRED series for commodities
COMMODITY_SERIES: dict[str, str] = {
    "IRON_ORE": "PIORECRUSDM",        # Iron ore price (monthly)
    "COPPER": "PCOPPUSDM",            # Copper price (monthly)
    "BRENT_OIL": "DCOILBRENTEU",      # Brent crude (daily)
    "WTI_OIL": "DCOILWTICO",          # WTI crude (daily)
    "NATURAL_GAS": "DHHNGSP",         # Henry Hub natural gas (daily)
    "GOLD_FIX": "GOLDAMGBD228NLBM",   # London gold fix (daily)
}

# OPEC meeting dates 2026 (approximate — announced annually)
OPEC_MEETINGS_2026: list[dict[str, str]] = [
    {"date": "2026-03-05", "type": "OPEC+ JMMC"},
    {"date": "2026-04-02", "type": "OPEC+ Full Ministerial"},
    {"date": "2026-06-04", "type": "OPEC+ JMMC"},
    {"date": "2026-07-02", "type": "OPEC+ Full Ministerial"},
    {"date": "2026-09-03", "type": "OPEC+ JMMC"},
    {"date": "2026-10-01", "type": "OPEC+ Full Ministerial"},
    {"date": "2026-12-03", "type": "OPEC+ Full Ministerial"},
]

# EIA petroleum report: every Wednesday at 10:30 AM ET (14:30 UTC)
EIA_REPORT_DAY = 2  # Wednesday = 2
EIA_REPORT_HOUR_UTC = 14
EIA_REPORT_MINUTE_UTC = 30


class OilCommodityFeed:
    """Fetches oil and commodity data."""

    def __init__(self) -> None:
        settings = get_settings()
        self._fred_key = settings.fred_api_key.get_secret_value()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _fetch_fred_latest(self, series_id: str) -> float | None:
        """Fetch latest value from FRED."""
        if not self._fred_key:
            return None

        client = await self._get_client()
        try:
            resp = await client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": self._fred_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                },
            )
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
            for obs in observations:
                val = obs.get("value", ".")
                if val != ".":
                    return float(val)
        except Exception as e:
            logger.error("FRED commodity fetch failed", extra={"series": series_id, "error": str(e)})
        return None

    async def get_oil_data(self) -> dict[str, Any]:
        """Get comprehensive oil market data."""
        wti = await self._fetch_fred_latest("DCOILWTICO")
        brent = await self._fetch_fred_latest("DCOILBRENTEU")

        # Brent-WTI spread (important for refinery margins)
        spread = None
        if wti and brent:
            spread = round(brent - wti, 2)

        # Check OPEC meetings
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        next_opec = None
        for meeting in OPEC_MEETINGS_2026:
            if meeting["date"] >= today:
                next_opec = meeting
                break

        return {
            "wti_price": wti,
            "brent_price": brent,
            "brent_wti_spread": spread,
            "next_opec_meeting": next_opec,
            "eia_report_today": self.is_eia_report_day(),
        }

    async def get_commodity_prices(self) -> dict[str, float | None]:
        """Get prices for commodities that drive currency pairs."""
        return {
            "iron_ore": await self._fetch_fred_latest("PIORECRUSDM"),
            "copper": await self._fetch_fred_latest("PCOPPUSDM"),
            "gold": await self._fetch_fred_latest("GOLDAMGBD228NLBM"),
            "natural_gas": await self._fetch_fred_latest("DHHNGSP"),
        }

    async def get_commodity_context(self) -> dict[str, Any]:
        """Full commodity context for Macro Agent."""
        oil = await self.get_oil_data()
        commodities = await self.get_commodity_prices()

        # Interpret for currency pairs
        interpretation: dict[str, str] = {}

        # Oil → USD/CAD
        if oil.get("wti_price"):
            wti = oil["wti_price"]
            if wti > 80:
                interpretation["USD/CAD"] = "Oil elevated — CAD supportive"
            elif wti < 60:
                interpretation["USD/CAD"] = "Oil weak — CAD pressure"
            else:
                interpretation["USD/CAD"] = "Oil neutral range"

        # Iron ore → AUD/USD
        iron = commodities.get("iron_ore")
        if iron:
            if iron > 120:
                interpretation["AUD/USD"] = "Iron ore elevated — AUD supportive"
            elif iron < 80:
                interpretation["AUD/USD"] = "Iron ore weak — AUD pressure"
            else:
                interpretation["AUD/USD"] = "Iron ore neutral range"

        # Copper → global growth proxy
        copper = commodities.get("copper")
        if copper:
            if copper > 10000:
                interpretation["GROWTH"] = "Copper high — risk-on signal"
            elif copper < 7000:
                interpretation["GROWTH"] = "Copper low — recession risk"

        return {
            "oil": oil,
            "commodities": commodities,
            "interpretation": interpretation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def is_eia_report_day(ts: datetime | None = None) -> bool:
        """Check if today is EIA petroleum report day (Wednesday)."""
        ts = ts or datetime.now(timezone.utc)
        return ts.weekday() == EIA_REPORT_DAY

    @staticmethod
    def is_near_eia_report(ts: datetime | None = None, window_minutes: int = 30) -> bool:
        """Check if we're near the EIA report time (Wednesday 14:30 UTC)."""
        ts = ts or datetime.now(timezone.utc)
        if ts.weekday() != EIA_REPORT_DAY:
            return False
        report_time = ts.replace(
            hour=EIA_REPORT_HOUR_UTC, minute=EIA_REPORT_MINUTE_UTC, second=0, microsecond=0
        )
        diff = abs((ts - report_time).total_seconds()) / 60
        return diff <= window_minutes

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
