"""
Session and Calendar Intelligence.

Knows what kind of day it is and adjusts system behavior accordingly.
Not all trading days are equal — NFP Friday, central bank meeting day,
month-end rebalancing, and London fix all have distinct characteristics.

Used by: Risk Manager, Portfolio Manager, all analysis agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class TradingSession(StrEnum):
    """Active trading sessions by timezone."""
    ASIAN = "asian"           # 00:00-08:00 UTC (Tokyo, Sydney, HK)
    LONDON = "london"         # 08:00-16:00 UTC
    NEW_YORK = "new_york"     # 13:00-21:00 UTC
    OVERLAP = "overlap"       # 13:00-16:00 UTC (highest liquidity)
    OFF_HOURS = "off_hours"   # 21:00-00:00 UTC


class DayType(StrEnum):
    """Special day classifications that affect trading behavior."""
    NORMAL = "normal"
    NFP_DAY = "nfp_day"                     # First Friday of month
    CENTRAL_BANK_DAY = "central_bank_day"   # Rate decision day
    MONTH_END = "month_end"                 # Last 3 trading days
    QUARTER_END = "quarter_end"             # Last 3 trading days of quarter
    HALF_DAY = "half_day"                   # Pre-holiday reduced hours
    HOLIDAY = "holiday"                     # Major market holiday


@dataclass
class SessionContext:
    """Complete session context for the current moment."""
    timestamp: datetime
    active_session: TradingSession
    day_type: DayType
    is_tradeable: bool              # within allowed session window
    london_fix_warning: bool        # within 15 min of 4PM GMT fix
    rollover_warning: bool          # within 15 min of 5PM EST rollover
    special_notes: list[str] = field(default_factory=list)
    upcoming_cb_meetings: list[dict[str, Any]] = field(default_factory=list)
    month_end_rebalancing: bool = False
    jpn_fiscal_yearend: bool = False   # March 25-31
    aus_fiscal_yearend: bool = False   # June 25-30


# ── Central Bank Meeting Schedule ─────────────────────────────

# Major central bank meeting dates for 2026 (approximate — update annually)
# These are hardcoded because they're published years in advance
CB_MEETINGS_2026: list[dict[str, Any]] = [
    # Fed (8 meetings)
    {"bank": "Fed", "currency": "USD", "date": "2026-01-28", "impact": "HIGH"},
    {"bank": "Fed", "currency": "USD", "date": "2026-03-18", "impact": "HIGH"},
    {"bank": "Fed", "currency": "USD", "date": "2026-05-06", "impact": "HIGH"},
    {"bank": "Fed", "currency": "USD", "date": "2026-06-17", "impact": "HIGH"},
    {"bank": "Fed", "currency": "USD", "date": "2026-07-29", "impact": "HIGH"},
    {"bank": "Fed", "currency": "USD", "date": "2026-09-16", "impact": "HIGH"},
    {"bank": "Fed", "currency": "USD", "date": "2026-11-04", "impact": "HIGH"},
    {"bank": "Fed", "currency": "USD", "date": "2026-12-16", "impact": "HIGH"},
    # ECB (8 meetings)
    {"bank": "ECB", "currency": "EUR", "date": "2026-01-22", "impact": "HIGH"},
    {"bank": "ECB", "currency": "EUR", "date": "2026-03-05", "impact": "HIGH"},
    {"bank": "ECB", "currency": "EUR", "date": "2026-04-16", "impact": "HIGH"},
    {"bank": "ECB", "currency": "EUR", "date": "2026-06-04", "impact": "HIGH"},
    {"bank": "ECB", "currency": "EUR", "date": "2026-07-16", "impact": "HIGH"},
    {"bank": "ECB", "currency": "EUR", "date": "2026-09-10", "impact": "HIGH"},
    {"bank": "ECB", "currency": "EUR", "date": "2026-10-22", "impact": "HIGH"},
    {"bank": "ECB", "currency": "EUR", "date": "2026-12-10", "impact": "HIGH"},
    # BOE (8 meetings)
    {"bank": "BOE", "currency": "GBP", "date": "2026-02-05", "impact": "HIGH"},
    {"bank": "BOE", "currency": "GBP", "date": "2026-03-19", "impact": "HIGH"},
    {"bank": "BOE", "currency": "GBP", "date": "2026-05-07", "impact": "HIGH"},
    {"bank": "BOE", "currency": "GBP", "date": "2026-06-18", "impact": "HIGH"},
    {"bank": "BOE", "currency": "GBP", "date": "2026-08-06", "impact": "HIGH"},
    {"bank": "BOE", "currency": "GBP", "date": "2026-09-17", "impact": "HIGH"},
    {"bank": "BOE", "currency": "GBP", "date": "2026-11-05", "impact": "HIGH"},
    {"bank": "BOE", "currency": "GBP", "date": "2026-12-17", "impact": "HIGH"},
    # BOJ (8 meetings)
    {"bank": "BOJ", "currency": "JPY", "date": "2026-01-22", "impact": "HIGH"},
    {"bank": "BOJ", "currency": "JPY", "date": "2026-03-13", "impact": "HIGH"},
    {"bank": "BOJ", "currency": "JPY", "date": "2026-04-28", "impact": "HIGH"},
    {"bank": "BOJ", "currency": "JPY", "date": "2026-06-16", "impact": "HIGH"},
    {"bank": "BOJ", "currency": "JPY", "date": "2026-07-15", "impact": "HIGH"},
    {"bank": "BOJ", "currency": "JPY", "date": "2026-09-17", "impact": "HIGH"},
    {"bank": "BOJ", "currency": "JPY", "date": "2026-10-29", "impact": "HIGH"},
    {"bank": "BOJ", "currency": "JPY", "date": "2026-12-18", "impact": "HIGH"},
    # SNB (4 meetings)
    {"bank": "SNB", "currency": "CHF", "date": "2026-03-19", "impact": "HIGH"},
    {"bank": "SNB", "currency": "CHF", "date": "2026-06-18", "impact": "HIGH"},
    {"bank": "SNB", "currency": "CHF", "date": "2026-09-24", "impact": "HIGH"},
    {"bank": "SNB", "currency": "CHF", "date": "2026-12-10", "impact": "HIGH"},
    # RBA (8 meetings)
    {"bank": "RBA", "currency": "AUD", "date": "2026-02-17", "impact": "HIGH"},
    {"bank": "RBA", "currency": "AUD", "date": "2026-04-07", "impact": "HIGH"},
    {"bank": "RBA", "currency": "AUD", "date": "2026-05-19", "impact": "HIGH"},
    {"bank": "RBA", "currency": "AUD", "date": "2026-07-07", "impact": "HIGH"},
    {"bank": "RBA", "currency": "AUD", "date": "2026-08-04", "impact": "HIGH"},
    {"bank": "RBA", "currency": "AUD", "date": "2026-09-01", "impact": "HIGH"},
    {"bank": "RBA", "currency": "AUD", "date": "2026-11-03", "impact": "HIGH"},
    {"bank": "RBA", "currency": "AUD", "date": "2026-12-01", "impact": "HIGH"},
    # RBNZ (7 meetings)
    {"bank": "RBNZ", "currency": "NZD", "date": "2026-02-19", "impact": "HIGH"},
    {"bank": "RBNZ", "currency": "NZD", "date": "2026-04-09", "impact": "HIGH"},
    {"bank": "RBNZ", "currency": "NZD", "date": "2026-05-27", "impact": "HIGH"},
    {"bank": "RBNZ", "currency": "NZD", "date": "2026-07-08", "impact": "HIGH"},
    {"bank": "RBNZ", "currency": "NZD", "date": "2026-08-12", "impact": "HIGH"},
    {"bank": "RBNZ", "currency": "NZD", "date": "2026-10-07", "impact": "HIGH"},
    {"bank": "RBNZ", "currency": "NZD", "date": "2026-11-25", "impact": "HIGH"},
]


# ── Session Detection ─────────────────────────────────────────

def get_active_session(ts: datetime | None = None) -> TradingSession:
    """Determine which trading session is currently active."""
    ts = ts or datetime.now(timezone.utc)
    hour = ts.hour

    if 13 <= hour < 16:
        return TradingSession.OVERLAP
    elif 8 <= hour < 16:
        return TradingSession.LONDON
    elif 13 <= hour < 21:
        return TradingSession.NEW_YORK
    elif 0 <= hour < 8:
        return TradingSession.ASIAN
    else:
        return TradingSession.OFF_HOURS


def is_nfp_day(ts: datetime | None = None) -> bool:
    """Check if today is NFP day (first Friday of the month)."""
    ts = ts or datetime.now(timezone.utc)
    if ts.weekday() != 4:  # Not Friday
        return False
    return ts.day <= 7  # First 7 days = first week


def is_central_bank_day(ts: datetime | None = None) -> tuple[bool, list[dict[str, Any]]]:
    """Check if today has a central bank meeting."""
    ts = ts or datetime.now(timezone.utc)
    today_str = ts.strftime("%Y-%m-%d")
    meetings = [m for m in CB_MEETINGS_2026 if m["date"] == today_str]
    return len(meetings) > 0, meetings


def is_month_end(ts: datetime | None = None, days_before: int = 3) -> bool:
    """Check if we're in the month-end rebalancing window (last N trading days)."""
    ts = ts or datetime.now(timezone.utc)
    next_month = (ts.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day = next_month - timedelta(days=1)
    days_until_end = (last_day - ts).days
    return days_until_end <= days_before


def is_quarter_end(ts: datetime | None = None) -> bool:
    """Check if we're in the quarter-end window."""
    ts = ts or datetime.now(timezone.utc)
    quarter_end_months = {3, 6, 9, 12}
    return ts.month in quarter_end_months and is_month_end(ts)


def is_london_fix_window(ts: datetime | None = None, window_minutes: int = 15) -> bool:
    """Check if we're within the London fix window (4:00 PM GMT)."""
    ts = ts or datetime.now(timezone.utc)
    fix_hour = 16  # 4 PM UTC
    fix_minute = 0
    fix_time = ts.replace(hour=fix_hour, minute=fix_minute, second=0, microsecond=0)
    diff = abs((ts - fix_time).total_seconds()) / 60
    return diff <= window_minutes


def is_rollover_window(ts: datetime | None = None, window_minutes: int = 15) -> bool:
    """Check if we're near the daily rollover (5:00 PM EST = 22:00 UTC)."""
    ts = ts or datetime.now(timezone.utc)
    rollover_hour = 22  # 10 PM UTC (5 PM EST)
    rollover_time = ts.replace(hour=rollover_hour, minute=0, second=0, microsecond=0)
    diff = abs((ts - rollover_time).total_seconds()) / 60
    return diff <= window_minutes


def is_jpn_fiscal_yearend(ts: datetime | None = None) -> bool:
    """Japanese fiscal year ends March 31. Major JPY repatriation flows."""
    ts = ts or datetime.now(timezone.utc)
    return ts.month == 3 and ts.day >= 25


def is_aus_fiscal_yearend(ts: datetime | None = None) -> bool:
    """Australian fiscal year ends June 30."""
    ts = ts or datetime.now(timezone.utc)
    return ts.month == 6 and ts.day >= 25


# ── Full Context Builder ──────────────────────────────────────

def get_session_context(ts: datetime | None = None) -> SessionContext:
    """Build complete session context for the current moment."""
    ts = ts or datetime.now(timezone.utc)

    session = get_active_session(ts)
    is_cb, cb_meetings = is_central_bank_day(ts)

    # Determine day type (priority order)
    if is_nfp_day(ts):
        day_type = DayType.NFP_DAY
    elif is_cb:
        day_type = DayType.CENTRAL_BANK_DAY
    elif is_quarter_end(ts):
        day_type = DayType.QUARTER_END
    elif is_month_end(ts):
        day_type = DayType.MONTH_END
    else:
        day_type = DayType.NORMAL

    # Tradeable = within London or NY session
    is_tradeable = session in (
        TradingSession.LONDON,
        TradingSession.NEW_YORK,
        TradingSession.OVERLAP,
    )

    # Build notes
    notes: list[str] = []
    if day_type == DayType.NFP_DAY:
        notes.append("NFP DAY — expect high volatility around 13:30 UTC")
    if day_type == DayType.CENTRAL_BANK_DAY:
        for m in cb_meetings:
            notes.append(f"{m['bank']} rate decision today — affects {m['currency']}")
    if is_month_end(ts):
        notes.append("Month-end rebalancing window — expect unusual flows")
    if is_london_fix_window(ts):
        notes.append("London fix window — avoid new positions")
    if is_rollover_window(ts):
        notes.append("Daily rollover window — swap costs apply")

    # Upcoming CB meetings (next 7 days)
    today_str = ts.strftime("%Y-%m-%d")
    week_from_now = (ts + timedelta(days=7)).strftime("%Y-%m-%d")
    upcoming = [
        m for m in CB_MEETINGS_2026
        if today_str <= m["date"] <= week_from_now
    ]

    return SessionContext(
        timestamp=ts,
        active_session=session,
        day_type=day_type,
        is_tradeable=is_tradeable,
        london_fix_warning=is_london_fix_window(ts),
        rollover_warning=is_rollover_window(ts),
        special_notes=notes,
        upcoming_cb_meetings=upcoming,
        month_end_rebalancing=is_month_end(ts),
        jpn_fiscal_yearend=is_jpn_fiscal_yearend(ts),
        aus_fiscal_yearend=is_aus_fiscal_yearend(ts),
    )
