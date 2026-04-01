"""
Capital.com REST API broker integration.

Handles authentication, order placement, position monitoring, and closure.
Supports demo and live environments via CAPITAL_ENVIRONMENT setting.

Capital.com API flow:
  1. POST /api/v1/session → CST + X-SECURITY-TOKEN (10-min TTL)
  2. POST /api/v1/positions → open trade with SL/TP
  3. GET  /api/v1/positions → monitor open trades
  4. DELETE /api/v1/positions/{dealId} → close trade
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Map internal symbols → Capital.com epic codes
EPIC_MAP: dict[str, str] = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
    "USD/CHF": "USDCHF",
    "AUD/USD": "AUDUSD",
    "USD/CAD": "USDCAD",
    "NZD/USD": "NZDUSD",
    "XAU/USD": "GOLD",
    "WTI": "OIL_CRUDE",
}

BASE_URLS = {
    "demo": "https://demo-api-capital.backend-capital.com",
    "live": "https://api-capital.backend-capital.com",
}


class CapitalComBroker:
    """
    Async Capital.com broker client.

    Manages session tokens with auto-refresh, provides methods to
    open/close/monitor positions, and fetch account state.
    """

    SESSION_TTL = 540  # Refresh tokens every 9 minutes (they expire at 10)

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.capital_api_key.get_secret_value()
        self._email = settings.capital_email
        self._password = settings.capital_password.get_secret_value()
        env = settings.capital_environment
        self._base_url = BASE_URLS.get(env, BASE_URLS["demo"])

        self._cst: str = ""
        self._security_token: str = ""
        self._session_ts: datetime | None = None
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {
            "X-CAP-API-KEY": self._api_key,
            "CST": self._cst,
            "X-SECURITY-TOKEN": self._security_token,
            "Content-Type": "application/json",
        }

    async def authenticate(self) -> bool:
        """Create a new session. Returns True on success."""
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self._base_url}/api/v1/session",
                headers={
                    "X-CAP-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "identifier": self._email,
                    "password": self._password,
                },
            )
            if resp.status_code != 200:
                logger.error("Auth failed", extra={"status": resp.status_code, "body": resp.text[:200]})
                return False

            self._cst = resp.headers.get("CST", "")
            self._security_token = resp.headers.get("X-SECURITY-TOKEN", "")
            self._session_ts = datetime.now(timezone.utc)
            logger.info("Capital.com session established")
            return True

        except Exception as e:
            logger.error("Auth exception", extra={"error": str(e)})
            return False

    async def _ensure_session(self) -> bool:
        """Re-authenticate if session is stale or missing."""
        if (
            not self._cst
            or not self._session_ts
            or (datetime.now(timezone.utc) - self._session_ts).total_seconds() > self.SESSION_TTL
        ):
            return await self.authenticate()
        return True

    async def _request(
        self, method: str, path: str, json: dict | None = None, params: dict | None = None
    ) -> httpx.Response | None:
        """Authenticated request with auto-refresh."""
        if not await self._ensure_session():
            return None

        client = await self._get_client()
        resp = await client.request(
            method,
            f"{self._base_url}{path}",
            headers=self._auth_headers(),
            json=json,
            params=params,
        )

        # Token expired mid-request — retry once
        if resp.status_code == 401:
            logger.warning("Session expired, re-authenticating")
            if await self.authenticate():
                resp = await client.request(
                    method,
                    f"{self._base_url}{path}",
                    headers=self._auth_headers(),
                    json=json,
                    params=params,
                )
        return resp

    # ── Account ──────────────────────────────────────────────

    async def get_account(self) -> dict[str, Any] | None:
        """Fetch account balance, equity, margin info."""
        resp = await self._request("GET", "/api/v1/accounts")
        if not resp or resp.status_code != 200:
            logger.error("Failed to fetch accounts", extra={"status": getattr(resp, "status_code", "?")})
            return None

        accounts = resp.json().get("accounts", [])
        if not accounts:
            return None

        acct = accounts[0]
        return {
            "account_id": acct.get("accountId"),
            "account_name": acct.get("accountName"),
            "balance": acct.get("balance", {}).get("balance", 0),
            "available": acct.get("balance", {}).get("available", 0),
            "deposit": acct.get("balance", {}).get("deposit", 0),
            "profit_loss": acct.get("balance", {}).get("profitLoss", 0),
            "currency": acct.get("currency"),
        }

    # ── Positions ────────────────────────────────────────────

    async def get_positions(self) -> list[dict[str, Any]]:
        """Fetch all open positions."""
        resp = await self._request("GET", "/api/v1/positions")
        if not resp or resp.status_code != 200:
            return []

        positions = []
        for p in resp.json().get("positions", []):
            pos = p.get("position", {})
            market = p.get("market", {})
            positions.append({
                "deal_id": pos.get("dealId"),
                "epic": market.get("epic"),
                "instrument": self._epic_to_symbol(market.get("epic", "")),
                "direction": pos.get("direction"),  # "BUY" or "SELL"
                "size": pos.get("size"),
                "open_level": pos.get("level"),
                "stop_level": pos.get("stopLevel"),
                "profit_level": pos.get("profitLevel"),
                "profit_loss": pos.get("profit"),
                "created": pos.get("createdDateUTC"),
                "currency": pos.get("currency"),
            })
        return positions

    async def open_position(
        self,
        instrument: str,
        direction: str,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop: bool = False,
    ) -> dict[str, Any] | None:
        """
        Open a new position.

        Args:
            instrument: Internal symbol (e.g. "EUR/USD")
            direction: "BUY" or "SELL"
            size: Position size in lots/units
            stop_loss: Absolute price level for SL
            take_profit: Absolute price level for TP
            trailing_stop: Enable trailing stop

        Returns:
            Deal confirmation dict or None on failure.
        """
        epic = EPIC_MAP.get(instrument)
        if not epic:
            logger.error("Unknown instrument", extra={"instrument": instrument})
            return None

        body: dict[str, Any] = {
            "epic": epic,
            "direction": direction.upper(),
            "size": size,
            "guaranteedStop": False,
            "trailingStop": trailing_stop,
        }

        if stop_loss is not None:
            body["stopLevel"] = round(stop_loss, 5)
        if take_profit is not None:
            body["profitLevel"] = round(take_profit, 5)

        logger.info(
            "Opening position",
            extra={"instrument": instrument, "direction": direction, "size": size, "sl": stop_loss, "tp": take_profit},
        )

        resp = await self._request("POST", "/api/v1/positions", json=body)
        if not resp:
            return None

        if resp.status_code == 200:
            data = resp.json()
            deal_ref = data.get("dealReference", "")
            logger.info("Position opened", extra={"deal_ref": deal_ref})

            # Confirm the deal
            confirm = await self._confirm_deal(deal_ref)

            # Fetch the actual position deal_id (differs from order deal_id)
            if confirm and confirm.get("status") == "ACCEPTED":
                await asyncio.sleep(0.3)
                positions = await self.get_positions()
                # Find the newest position matching this instrument
                for p in reversed(positions):
                    if p["instrument"] == instrument:
                        confirm["position_deal_id"] = p["deal_id"]
                        break

            return confirm

        logger.error(
            "Failed to open position",
            extra={"status": resp.status_code, "body": resp.text[:300]},
        )
        return None

    async def close_position(self, deal_id: str) -> dict[str, Any] | None:
        """Close an open position by deal ID."""
        logger.info("Closing position", extra={"deal_id": deal_id})

        resp = await self._request("DELETE", f"/api/v1/positions/{deal_id}")
        if not resp:
            return None

        if resp.status_code == 200:
            data = resp.json()
            deal_ref = data.get("dealReference", "")
            logger.info("Position closed", extra={"deal_ref": deal_ref})
            return await self._confirm_deal(deal_ref)

        logger.error(
            "Failed to close position",
            extra={"status": resp.status_code, "body": resp.text[:300]},
        )
        return None

    async def update_position(
        self,
        deal_id: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop: bool = False,
    ) -> dict[str, Any] | None:
        """Update SL/TP on an existing position."""
        body: dict[str, Any] = {"trailingStop": trailing_stop}
        if stop_loss is not None:
            body["stopLevel"] = round(stop_loss, 5)
        if take_profit is not None:
            body["profitLevel"] = round(take_profit, 5)

        resp = await self._request("PUT", f"/api/v1/positions/{deal_id}", json=body)
        if not resp:
            return None

        if resp.status_code == 200:
            data = resp.json()
            deal_ref = data.get("dealReference", "")
            return await self._confirm_deal(deal_ref)

        logger.error("Failed to update position", extra={"status": resp.status_code, "body": resp.text[:200]})
        return None

    # ── Market Info ──────────────────────────────────────────

    async def get_market_info(self, instrument: str) -> dict[str, Any] | None:
        """Get market details: min size, spread, trading hours."""
        epic = EPIC_MAP.get(instrument)
        if not epic:
            return None

        resp = await self._request("GET", f"/api/v1/markets/{epic}")
        if not resp or resp.status_code != 200:
            return None

        data = resp.json()
        snapshot = data.get("snapshot", {})
        rules = data.get("dealingRules", {})

        return {
            "epic": epic,
            "instrument": instrument,
            "bid": snapshot.get("bid"),
            "offer": snapshot.get("offer"),
            "spread": round(snapshot.get("offer", 0) - snapshot.get("bid", 0), 5),
            "status": snapshot.get("marketStatus"),
            "min_size": rules.get("minDealSize", {}).get("value"),
            "max_size": rules.get("maxDealSize", {}).get("value"),
            "min_stop_distance": rules.get("minNormalStopOrLimitDistance", {}).get("value"),
        }

    # ── Helpers ──────────────────────────────────────────────

    async def _confirm_deal(self, deal_reference: str) -> dict[str, Any] | None:
        """Poll deal confirmation endpoint."""
        if not deal_reference:
            return None

        await asyncio.sleep(0.5)  # Brief delay for processing
        resp = await self._request("GET", f"/api/v1/confirms/{deal_reference}")
        if not resp or resp.status_code != 200:
            return None

        data = resp.json()
        return {
            "deal_id": data.get("dealId"),
            "deal_reference": deal_reference,
            "status": data.get("dealStatus"),  # "ACCEPTED" or "REJECTED"
            "reason": data.get("reason", ""),
            "direction": data.get("direction"),
            "size": data.get("size"),
            "level": data.get("level"),
            "stop_level": data.get("stopLevel"),
            "profit_level": data.get("profitLevel"),
            "profit_loss": data.get("profit"),
        }

    @staticmethod
    def _epic_to_symbol(epic: str) -> str:
        """Reverse-map Capital.com epic to internal symbol."""
        for sym, ep in EPIC_MAP.items():
            if ep == epic:
                return sym
        return epic

    async def close(self) -> None:
        """Close HTTP client and session."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ── Module-level singleton ───────────────────────────────────

_broker: CapitalComBroker | None = None


def get_broker() -> CapitalComBroker:
    """Get or create the broker singleton."""
    global _broker
    if _broker is None:
        _broker = CapitalComBroker()
    return _broker
