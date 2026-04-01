"""
COT (Commitment of Traders) report feed.

Weekly data from CFTC showing institutional positioning.
Used by the Sentiment Agent to detect crowded trades and reversal risk.

Schedule: Published every Friday for the previous Tuesday's data.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# CFTC COT report URL (current year)
CFTC_BASE_URL = "https://www.cftc.gov/dea/newcot"

# Map CFTC contract names to our instruments
CONTRACT_MAP: dict[str, str] = {
    "EURO FX": "EUR/USD",
    "BRITISH POUND": "GBP/USD",
    "JAPANESE YEN": "USD/JPY",
    "SWISS FRANC": "USD/CHF",
    "AUSTRALIAN DOLLAR": "AUD/USD",
    "CANADIAN DOLLAR": "USD/CAD",
    "NEW ZEALAND DOLLAR": "NZD/USD",
    "GOLD": "XAU/USD",
    "CRUDE OIL, LIGHT SWEET": "WTI",
}


class COTFeed:
    """Fetches and parses CFTC Commitment of Traders reports."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def fetch_latest_cot(self) -> list[dict[str, Any]]:
        """
        Fetch the latest COT report.
        Returns parsed positioning data per instrument.
        """
        client = await self._get_client()

        try:
            # Futures-only report (most relevant for forex)
            resp = await client.get(f"{CFTC_BASE_URL}/fut_fin_txt.zip")
            resp.raise_for_status()

            # Parse the CSV from the zip
            import zipfile
            zip_buffer = io.BytesIO(resp.content)
            with zipfile.ZipFile(zip_buffer) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    content = f.read().decode("utf-8")

            return self._parse_cot_csv(content)

        except Exception as e:
            logger.error("COT fetch failed", extra={"error": str(e)})
            return []

    def _parse_cot_csv(self, content: str) -> list[dict[str, Any]]:
        """Parse CFTC CSV into structured positioning data."""
        reader = csv.DictReader(io.StringIO(content))
        results = []

        for row in reader:
            contract_name = row.get("Market_and_Exchange_Names", "").strip()

            # Match to our instruments
            instrument = None
            for cftc_name, our_symbol in CONTRACT_MAP.items():
                if cftc_name in contract_name.upper():
                    instrument = our_symbol
                    break

            if not instrument:
                continue

            try:
                # Non-commercial (speculators) positioning
                longs = int(row.get("NonComm_Positions_Long_All", 0))
                shorts = int(row.get("NonComm_Positions_Short_All", 0))
                net = longs - shorts

                # Changes from previous week
                change_long = int(row.get("Change_in_NonComm_Long_All", 0))
                change_short = int(row.get("Change_in_NonComm_Short_All", 0))

                # Open interest
                open_interest = int(row.get("Open_Interest_All", 0))

                results.append({
                    "instrument": instrument,
                    "report_date": row.get("As_of_Date_In_Form_YYMMDD", ""),
                    "non_commercial_long": longs,
                    "non_commercial_short": shorts,
                    "net_position": net,
                    "change_long": change_long,
                    "change_short": change_short,
                    "open_interest": open_interest,
                    "net_pct_of_oi": round(net / open_interest * 100, 2) if open_interest else 0,
                    "bias": "long" if net > 0 else "short" if net < 0 else "neutral",
                })

            except (ValueError, KeyError) as e:
                logger.debug("Skipping COT row", extra={"error": str(e)})
                continue

        logger.info("Parsed %d COT positions", len(results))
        return results

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
