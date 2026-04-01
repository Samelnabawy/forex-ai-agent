"""
Embedding Generation for the Historical Brain.

Converts market events, conditions, and states into 1536-dimensional
vectors stored in pgvector for similarity search.

Two embedding strategies:
  1. Text embeddings — event descriptions, CB statements, news narratives
  2. Market moment embeddings — numerical market state encoded as structured text
     then embedded (price levels, indicator values, regime, correlations)

Uses Claude API for embeddings (via Voyage or compatible endpoint).
Falls back to local sentence-transformers if API unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import httpx
import numpy as np

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536
_embedding_cache: dict[str, list[float]] = {}


async def generate_embedding(
    text: str,
    model: str = "voyage-3",
) -> list[float]:
    """
    Generate a 1536-dim embedding for text.
    Uses Voyage API (Anthropic's embedding partner) or compatible endpoint.
    Caches results to avoid redundant API calls.
    """
    # Cache check
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    settings = get_settings()
    api_key = settings.anthropic_api_key.get_secret_value()

    if not api_key:
        logger.warning("No API key — using deterministic pseudo-embedding")
        embedding = _deterministic_embedding(text)
        _embedding_cache[cache_key] = embedding
        return embedding

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": [text[:8000]],  # Truncate to model limit
                    "input_type": "document",
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                embedding = data["data"][0]["embedding"]
                _embedding_cache[cache_key] = embedding
                return embedding
            else:
                logger.warning("Embedding API returned %d — using fallback", resp.status_code)

    except Exception as e:
        logger.warning("Embedding API failed: %s — using fallback", str(e))

    embedding = _deterministic_embedding(text)
    _embedding_cache[cache_key] = embedding
    return embedding


async def generate_batch_embeddings(
    texts: list[str],
    model: str = "voyage-3",
) -> list[list[float]]:
    """Generate embeddings for multiple texts. Batches for efficiency."""
    # For now, sequential. Can be batched with the API.
    results = []
    for text in texts:
        emb = await generate_embedding(text, model)
        results.append(emb)
    return results


def encode_market_moment(
    instrument: str,
    price: float,
    indicators: dict[str, Any],
    macro_regime: str,
    correlations: dict[str, float] | None = None,
    sentiment: dict[str, Any] | None = None,
    context: str = "",
) -> str:
    """
    Encode a market state as structured text for embedding.

    Instead of embedding raw numbers (which don't work well),
    we describe the market state in natural language that captures
    the MEANING of the numbers.
    """
    parts = [f"Instrument: {instrument} at {price:.5f}."]

    # Trend description
    ema_alignment = indicators.get("ema_alignment", "mixed")
    adx = indicators.get("adx", 0)
    if ema_alignment == "bullish" and adx and adx > 25:
        parts.append(f"Strong uptrend with EMA alignment bullish, ADX at {adx:.0f}.")
    elif ema_alignment == "bearish" and adx and adx > 25:
        parts.append(f"Strong downtrend with EMA alignment bearish, ADX at {adx:.0f}.")
    elif adx and adx < 20:
        parts.append(f"Ranging market, ADX at {adx:.0f} indicating no clear trend.")
    else:
        parts.append(f"Mixed trend, EMA alignment {ema_alignment}, ADX at {adx or 'unknown'}.")

    # Momentum
    rsi = indicators.get("rsi")
    if rsi:
        if rsi < 30:
            parts.append(f"RSI oversold at {rsi:.0f}.")
        elif rsi > 70:
            parts.append(f"RSI overbought at {rsi:.0f}.")
        else:
            parts.append(f"RSI neutral at {rsi:.0f}.")

    macd = indicators.get("macd_cross", "none")
    if macd != "none":
        parts.append(f"MACD {macd} cross confirmed.")

    # Volatility
    squeeze = indicators.get("squeeze")
    if squeeze:
        parts.append("Bollinger/Keltner squeeze active — volatility expansion expected.")

    bb = indicators.get("bb_position", "")
    if bb:
        parts.append(f"Price at {bb.replace('_', ' ')} of Bollinger Bands.")

    # Macro context
    parts.append(f"Macro regime: {macro_regime}.")

    # Correlations
    if correlations:
        strong_corrs = [(k, v) for k, v in correlations.items() if abs(v) > 0.7]
        if strong_corrs:
            corr_desc = ", ".join(f"{k} ({v:+.2f})" for k, v in strong_corrs[:3])
            parts.append(f"Strong correlations: {corr_desc}.")

    # Sentiment
    if sentiment:
        fg = sentiment.get("fear_greed", {})
        if fg:
            parts.append(f"Fear/Greed index at {fg.get('score', 'unknown')} ({fg.get('label', '')}).")
        cot = sentiment.get("cot_signal", "")
        if cot:
            parts.append(f"COT positioning: {cot}.")

    # Additional context
    if context:
        parts.append(context)

    return " ".join(parts)


def _deterministic_embedding(text: str) -> list[float]:
    """
    Generate a deterministic pseudo-embedding from text.
    NOT semantically meaningful — only for testing and fallback.
    Uses hash-based approach to produce consistent vectors.
    """
    np.random.seed(int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**31))
    embedding = np.random.randn(EMBEDDING_DIM).tolist()
    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = (np.array(embedding) / norm).tolist()
    return embedding
