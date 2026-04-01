"""
Geopolitical Risk Tracker.

Uses LLM to classify and score geopolitical events from news headlines.
Monitors:
  - Military conflicts (war escalation/de-escalation)
  - Sanctions and trade policy
  - Election uncertainty
  - Central bank intervention signals
  - Natural disasters (commodity supply disruption)

Output: Geopolitical risk score (0-100) + affected currencies/commodities.

Execution: Triggered by breaking news, plus periodic scan every 2 hours.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.agents.llm_client import get_llm_client

logger = logging.getLogger(__name__)

GEOPOLITICAL_SYSTEM_PROMPT = """You are a geopolitical risk analyst for a forex trading desk.
Your job is to assess geopolitical events and quantify their impact on currency markets.

Risk categories:
1. MILITARY — war, conflict escalation/de-escalation, military posturing
2. SANCTIONS — trade restrictions, asset freezes, economic warfare
3. POLITICAL — elections, government instability, policy shifts
4. INTERVENTION — central bank FX intervention (BOJ, SNB history)
5. SUPPLY_SHOCK — natural disasters, infrastructure failures affecting commodities

Impact mapping:
- Military escalation → safe havens (JPY, CHF, Gold UP; AUD, NZD DOWN)
- Sanctions on energy producers → Oil UP → USD/CAD DOWN, Gold UP
- Election uncertainty → volatility premium on that currency
- BOJ intervention threat → USD/JPY reversal risk when above 155-160
- SNB intervention → USD/CHF floor defense history

Score geopolitical risk 0-100:
  0-20: Calm, no significant risks
  20-40: Elevated awareness, monitor
  40-60: Active geopolitical tensions, adjust position sizes
  60-80: Crisis conditions, reduce exposure
  80-100: Extreme crisis, defensive positioning only

Always respond with valid JSON only."""

GEOPOLITICAL_USER_TEMPLATE = """Analyze these recent headlines for geopolitical risk to forex markets:

{headlines}

Current known conflicts/situations to monitor:
{known_situations}

Respond with:
{{
    "overall_risk_score": 0-100,
    "risk_level": "calm" | "elevated" | "high" | "crisis" | "extreme",
    "events": [
        {{
            "headline": "the headline",
            "category": "MILITARY|SANCTIONS|POLITICAL|INTERVENTION|SUPPLY_SHOCK",
            "severity": 0-100,
            "affected_currencies": ["USD", "JPY", ...],
            "affected_commodities": ["GOLD", "OIL", ...],
            "expected_direction": {{
                "safe_havens": "bid" | "offered" | "neutral",
                "risk_assets": "bid" | "offered" | "neutral"
            }},
            "duration": "hours" | "days" | "weeks",
            "reasoning": "brief explanation"
        }}
    ],
    "regime_impact": "none" | "risk_off_tilt" | "risk_off_shift" | "crisis_mode",
    "intervention_risk": {{
        "USD/JPY": "none" | "verbal" | "check_rate" | "imminent",
        "USD/CHF": "none" | "verbal" | "imminent"
    }},
    "recommended_adjustments": [
        "Reduce AUD exposure due to China tensions",
        "Widen stops on JPY pairs for intervention risk"
    ]
}}"""

# Known geopolitical situations to monitor (update periodically)
KNOWN_SITUATIONS: list[str] = [
    "Russia-Ukraine conflict: monitor for escalation/ceasefire signals",
    "Middle East tensions: Iran-Israel dynamics, Houthi shipping disruption",
    "US-China relations: Taiwan strait, tech sanctions, trade policy",
    "BOJ intervention: watch USD/JPY levels above 155 for verbal/actual intervention",
    "SNB policy: history of EUR/CHF floor defense",
    "Global elections: track upcoming elections in G10 countries",
]


class GeopoliticalTracker:
    """Tracks and scores geopolitical risks for forex markets."""

    def __init__(self) -> None:
        self._last_score: float = 0.0
        self._last_assessment: dict[str, Any] = {}
        self._last_update: datetime | None = None

    async def assess_risk(self, headlines: list[str]) -> dict[str, Any]:
        """
        Analyze headlines for geopolitical risk.
        Uses Claude Haiku for speed (this runs frequently).
        """
        if not headlines:
            return self._default_response()

        # Build prompt
        headlines_text = "\n".join(f"- {h}" for h in headlines[:20])
        situations_text = "\n".join(f"- {s}" for s in KNOWN_SITUATIONS)

        user_prompt = GEOPOLITICAL_USER_TEMPLATE.format(
            headlines=headlines_text,
            known_situations=situations_text,
        )

        # Call LLM (Haiku for speed)
        llm = get_llm_client()
        result = await llm.classify(
            system_prompt=GEOPOLITICAL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            agent_name="geopolitical_tracker",
            prompt_version="v1",
        )

        # Validate and store
        if not result.get("_fallback"):
            self._last_score = result.get("overall_risk_score", 0)
            self._last_assessment = result
            self._last_update = datetime.now(timezone.utc)

        return result

    @property
    def current_risk_score(self) -> float:
        return self._last_score

    @property
    def is_crisis(self) -> bool:
        return self._last_score >= 60

    @staticmethod
    def _default_response() -> dict[str, Any]:
        return {
            "overall_risk_score": 0,
            "risk_level": "calm",
            "events": [],
            "regime_impact": "none",
            "intervention_risk": {"USD/JPY": "none", "USD/CHF": "none"},
            "recommended_adjustments": [],
        }
