"""
Shared Claude API client for all LLM-powered agents.

Used by: Macro Analyst, Sentiment Agent, Bull Researcher, Bear Researcher.

Features:
  - Async Claude API calls via anthropic SDK
  - Model selection: Sonnet (reasoning) / Haiku (quick tasks)
  - Shared rate limiter (semaphore across all agents)
  - Retry with exponential backoff
  - Cost tracking + daily circuit breaker
  - Stale-data fallback (last known good output)
  - Prompt versioning for A/B testing
  - Structured JSON output enforcement
  - Token usage logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Cost Tracking ─────────────────────────────────────────────

# Approximate cost per 1M tokens (as of April 2026)
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}

# Daily cost cap — kills all LLM calls if breached
DAILY_COST_CAP_USD = 25.0


@dataclass
class UsageStats:
    """Track token usage and cost."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0
    daily_reset_date: str = ""
    calls_by_agent: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_name: str = "",
    ) -> float:
        """Record usage and return cost for this call."""
        costs = MODEL_COSTS.get(model, {"input": 5.0, "output": 15.0})
        cost = (input_tokens / 1_000_000 * costs["input"]) + \
               (output_tokens / 1_000_000 * costs["output"])

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
        self.total_cost_usd += cost

        # Daily tracking
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily_reset_date != today:
            self.daily_cost_usd = 0.0
            self.daily_reset_date = today
        self.daily_cost_usd += cost

        if agent_name:
            self.calls_by_agent[agent_name] = self.calls_by_agent.get(agent_name, 0) + 1

        return cost

    @property
    def is_over_daily_cap(self) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily_reset_date != today:
            return False
        return self.daily_cost_usd >= DAILY_COST_CAP_USD


# ── Fallback Cache ────────────────────────────────────────────

@dataclass
class CachedResponse:
    """Cached LLM response for fallback."""
    content: dict[str, Any]
    timestamp: datetime
    model: str
    agent_name: str
    prompt_version: str = ""

    @property
    def age_minutes(self) -> float:
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 60

    @property
    def is_stale(self) -> bool:
        """Stale after 2 hours."""
        return self.age_minutes > 120


# ── LLM Client ────────────────────────────────────────────────

class LLMClient:
    """
    Shared Claude API client.
    Singleton — create once, share across all agents.
    """

    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    MAX_CONCURRENT_CALLS = 3  # shared semaphore across all agents

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.anthropic_api_key.get_secret_value()
        if not api_key:
            logger.warning("Anthropic API key not configured — LLM calls will fail")

        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else None
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CALLS)
        self._usage = UsageStats()
        self._fallback_cache: dict[str, CachedResponse] = {}
        self._reasoning_model = settings.claude_reasoning_model
        self._fast_model = settings.claude_fast_model

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        prompt_version: str = "v1",
        require_json: bool = True,
    ) -> dict[str, Any]:
        """
        Send an analysis request to Claude.
        Returns parsed JSON response.

        Falls back to cached response if API fails.
        """
        model = model or self._reasoning_model
        cache_key = f"{agent_name}:{prompt_version}"

        # Check cost circuit breaker
        if self._usage.is_over_daily_cap:
            logger.error(
                "Daily cost cap breached — blocking LLM call",
                extra={
                    "daily_cost": self._usage.daily_cost_usd,
                    "cap": DAILY_COST_CAP_USD,
                    "agent": agent_name,
                },
            )
            return self._get_fallback(cache_key, reason="cost_cap_breached")

        if self._client is None:
            logger.warning("No API client — returning fallback")
            return self._get_fallback(cache_key, reason="no_api_key")

        # Enforce JSON output
        if require_json:
            json_instruction = (
                "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                "No markdown, no backticks, no explanation outside the JSON."
            )
            user_prompt = user_prompt + json_instruction

        # Call with rate limiting and retry
        async with self._semaphore:
            for attempt in range(self.MAX_RETRIES):
                try:
                    start = time.monotonic()

                    response = await self._client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )

                    elapsed_ms = int((time.monotonic() - start) * 1000)

                    # Extract text content
                    raw_text = ""
                    for block in response.content:
                        if block.type == "text":
                            raw_text += block.text

                    # Track usage
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    cost = self._usage.record(model, input_tokens, output_tokens, agent_name)

                    logger.info(
                        "LLM call complete",
                        extra={
                            "agent": agent_name,
                            "model": model,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cost_usd": round(cost, 4),
                            "elapsed_ms": elapsed_ms,
                            "prompt_version": prompt_version,
                        },
                    )

                    # Parse JSON
                    parsed = self._parse_json_response(raw_text)

                    # Cache successful response
                    self._fallback_cache[cache_key] = CachedResponse(
                        content=parsed,
                        timestamp=datetime.now(timezone.utc),
                        model=model,
                        agent_name=agent_name,
                        prompt_version=prompt_version,
                    )

                    return parsed

                except anthropic.RateLimitError:
                    wait = self.INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Rate limited, backing off",
                        extra={"attempt": attempt + 1, "wait_seconds": wait},
                    )
                    await asyncio.sleep(wait)

                except anthropic.APIConnectionError as e:
                    logger.error(
                        "API connection error",
                        extra={"error": str(e), "attempt": attempt + 1},
                    )
                    if attempt == self.MAX_RETRIES - 1:
                        self._usage.errors += 1
                        return self._get_fallback(cache_key, reason="connection_error")
                    await asyncio.sleep(self.INITIAL_BACKOFF * (2 ** attempt))

                except anthropic.APIStatusError as e:
                    logger.error(
                        "API status error",
                        extra={"status": e.status_code, "error": str(e)},
                    )
                    self._usage.errors += 1
                    if e.status_code >= 500:
                        # Server error — retry
                        await asyncio.sleep(self.INITIAL_BACKOFF * (2 ** attempt))
                    else:
                        # Client error — don't retry
                        return self._get_fallback(cache_key, reason=f"api_error_{e.status_code}")

                except Exception as e:
                    logger.error(
                        "Unexpected LLM error",
                        extra={"error": str(e), "attempt": attempt + 1},
                    )
                    self._usage.errors += 1
                    return self._get_fallback(cache_key, reason="unexpected_error")

        return self._get_fallback(cache_key, reason="exhausted_retries")

    async def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str = "",
        prompt_version: str = "v1",
    ) -> dict[str, Any]:
        """Quick classification using Haiku. For news sentiment, headline parsing, etc."""
        return await self.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            model=self._fast_model,
            max_tokens=1024,
            temperature=0.1,
            prompt_version=prompt_version,
        )

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse LLM JSON response",
                extra={"error": str(e), "raw_length": len(raw)},
            )
            # Return raw text wrapped in a dict
            return {"raw_response": raw, "parse_error": str(e)}

    def _get_fallback(self, cache_key: str, reason: str) -> dict[str, Any]:
        """Return cached response or empty fallback."""
        cached = self._fallback_cache.get(cache_key)
        if cached:
            logger.warning(
                "Using cached LLM response as fallback",
                extra={
                    "cache_key": cache_key,
                    "reason": reason,
                    "age_minutes": round(cached.age_minutes, 1),
                    "is_stale": cached.is_stale,
                },
            )
            result = cached.content.copy()
            result["_fallback"] = True
            result["_fallback_reason"] = reason
            result["_fallback_age_minutes"] = round(cached.age_minutes, 1)
            result["_is_stale"] = cached.is_stale
            return result

        logger.error(
            "No fallback available",
            extra={"cache_key": cache_key, "reason": reason},
        )
        return {
            "_fallback": True,
            "_fallback_reason": reason,
            "_no_cache": True,
        }

    @property
    def usage(self) -> dict[str, Any]:
        u = self._usage
        return {
            "total_calls": u.total_calls,
            "total_cost_usd": round(u.total_cost_usd, 4),
            "daily_cost_usd": round(u.daily_cost_usd, 4),
            "daily_cap_usd": DAILY_COST_CAP_USD,
            "total_input_tokens": u.total_input_tokens,
            "total_output_tokens": u.total_output_tokens,
            "errors": u.errors,
            "calls_by_agent": u.calls_by_agent,
        }


# ── Singleton ─────────────────────────────────────────────────

_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the shared LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
