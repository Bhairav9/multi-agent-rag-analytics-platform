"""Unit tests for Redis cache layer (hermetic — FakeRedis)."""

from __future__ import annotations

import pytest

from rag_platform.cache.conversation import load_history, save_turn
from rag_platform.cache.llm_cache import get_cached, set_cached


class FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    async def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self._lists.get(key, [])
        if end == -1:
            return items[start:]
        return items[start : end + 1]


@pytest.mark.asyncio
async def test_cache_miss_returns_none() -> None:
    redis = FakeRedis()
    result = await get_cached("What is Apple's revenue?", redis)
    assert result is None


@pytest.mark.asyncio
async def test_cache_hit_returns_stored_result() -> None:
    redis = FakeRedis()
    payload = {"final_answer": "Apple's revenue was $391B.", "citations": []}
    await set_cached("What is Apple's revenue?", payload, redis)
    result = await get_cached("What is Apple's revenue?", redis)
    assert result is not None
    assert result["final_answer"] == "Apple's revenue was $391B."


@pytest.mark.asyncio
async def test_cache_normalizes_query() -> None:
    redis = FakeRedis()
    payload = {"final_answer": "Apple's revenue was $391B.", "citations": []}
    await set_cached("  What is Apple's revenue?  ", payload, redis)
    result = await get_cached("what is apple's revenue?", redis)
    assert result is not None


@pytest.mark.asyncio
async def test_conversation_history_empty_session() -> None:
    redis = FakeRedis()
    history = await load_history("new-session", redis)
    assert history == []


@pytest.mark.asyncio
async def test_conversation_history_save_and_load() -> None:
    redis = FakeRedis()
    await save_turn("session-1", "What is Apple's revenue?", "It was $391B.", redis)
    history = await load_history("session-1", redis)
    assert len(history) == 2
    assert history[0].content == "What is Apple's revenue?"
    assert history[1].content == "It was $391B."


@pytest.mark.asyncio
async def test_conversation_history_multiple_turns() -> None:
    redis = FakeRedis()
    await save_turn("session-2", "Question 1", "Answer 1", redis)
    await save_turn("session-2", "Question 2", "Answer 2", redis)
    history = await load_history("session-2", redis)
    assert len(history) == 4
    assert history[2].content == "Question 2"
    assert history[3].content == "Answer 2"


@pytest.mark.asyncio
async def test_sessions_are_isolated() -> None:
    redis = FakeRedis()
    await save_turn("session-A", "Q for A", "A for A", redis)
    await save_turn("session-B", "Q for B", "A for B", redis)
    history_a = await load_history("session-A", redis)
    history_b = await load_history("session-B", redis)
    assert len(history_a) == 2
    assert len(history_b) == 2
    assert history_a[0].content == "Q for A"
    assert history_b[0].content == "Q for B"
