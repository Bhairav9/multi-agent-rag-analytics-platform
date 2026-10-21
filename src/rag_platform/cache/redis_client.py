"""Async Redis client wrapper."""

from __future__ import annotations

from typing import Protocol

import redis.asyncio as aioredis

from rag_platform.config import get_settings

class RedisClient(Protocol):
    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> None: ...
    async def rpush(self, key: str, value: str) -> None: ...
    async def lrange(self, key: str, start: int, end: int) -> list[bytes]: ...


def get_redis_client() -> aioredis.Redis:
    """Return a connected async Redis client."""
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)
