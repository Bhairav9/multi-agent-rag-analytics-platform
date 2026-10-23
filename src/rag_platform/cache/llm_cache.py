"""Exact-match LLM response cache backed by Redis."""

from __future__ import annotations

import hashlib
import json

from rag_platform.cache.redis_client import RedisClient
from rag_platform.config import get_settings


def _cache_key(query: str) -> str:
    normalized = query.strip().lower()
    return f"llm:cache:{hashlib.sha256(normalized.encode()).hexdigest()}"


async def get_cached(query: str, redis: RedisClient) -> dict | None:
    raw = await redis.get(_cache_key(query))
    if raw is None:
        return None
    return json.loads(raw)


async def set_cached(query: str, result: dict, redis: RedisClient) -> None:
    settings = get_settings()
    await redis.set(_cache_key(query), json.dumps(result), ex=settings.cache_ttl_seconds)
