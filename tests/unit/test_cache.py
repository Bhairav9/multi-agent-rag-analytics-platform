"""Unit tests for the Redis-backed cache layer (hermetic — no Redis needed)."""

from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock

from rag_platform.cache.conversation import load_history, save_turn
from rag_platform.cache.llm_cache import _cache_key, get_cached, set_cached


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_redis(*, get_return=None, lrange_return=None) -> AsyncMock:
    redis = AsyncMock()
    redis.get.return_value = get_return
    redis.lrange.return_value = lrange_return or []
    return redis


# ---------------------------------------------------------------------------
# llm_cache tests
# ---------------------------------------------------------------------------

def test_cache_key_is_deterministic() -> None:
    assert _cache_key("Hello World") == _cache_key("hello world")
    assert _cache_key("  hello  ") == _cache_key("hello")


def test_cache_key_differs_for_different_queries() -> None:
    assert _cache_key("apple revenue") != _cache_key("microsoft revenue")


@pytest.mark.asyncio
async def test_get_cached_returns_none_on_miss() -> None:
    redis = _mock_redis(get_return=None)
    result = await get_cached("some query", redis)
    assert result is None


@pytest.mark.asyncio
async def test_get_cached_deserializes_hit() -> None:
    payload = {"final_answer": "Apple's revenue was $391B.", "citations": ["acc-123"]}
    redis = _mock_redis(get_return=json.dumps(payload))
    result = await get_cached("apple revenue", redis)
    assert result == payload


@pytest.mark.asyncio
async def test_set_cached_serializes_and_stores(monkeypatch) -> None:
    from rag_platform.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("CACHE_TTL_SECONDS", "3600")
    get_settings.cache_clear()

    redis = _mock_redis()
    payload = {"final_answer": "stub", "citations": []}
    await set_cached("apple revenue", payload, redis)

    redis.set.assert_awaited_once()
    call_args = redis.set.call_args
    stored = json.loads(call_args.args[1])
    assert stored == payload

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# conversation cache tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_history_empty_returns_empty_list() -> None:
    redis = _mock_redis(lrange_return=[])
    messages = await load_history("session-1", redis)
    assert messages == []


@pytest.mark.asyncio
async def test_load_history_deserializes_messages() -> None:
    items = [
        json.dumps({"role": "human", "content": "What is Apple's revenue?"}),
        json.dumps({"role": "ai", "content": "Apple's revenue was $391B."}),
    ]
    redis = _mock_redis(lrange_return=items)
    messages = await load_history("session-1", redis)

    assert len(messages) == 2
    from langchain_core.messages import HumanMessage, AIMessage
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "What is Apple's revenue?"
    assert messages[1].content == "Apple's revenue was $391B."


@pytest.mark.asyncio
async def test_save_turn_pushes_two_messages() -> None:
    redis = _mock_redis()
    await save_turn("session-1", "What is Apple's revenue?", "It was $391B.", redis)

    assert redis.rpush.await_count == 2
    calls = redis.rpush.call_args_list
    human_msg = json.loads(calls[0].args[1])
    ai_msg = json.loads(calls[1].args[1])
    assert human_msg == {"role": "human", "content": "What is Apple's revenue?"}
    assert ai_msg == {"role": "ai", "content": "It was $391B."}


@pytest.mark.asyncio
async def test_save_then_load_roundtrip() -> None:
    stored: list[str] = []

    redis = AsyncMock()
    redis.rpush.side_effect = lambda key, val: stored.append(val) or None
    redis.lrange.return_value = stored[-20:]

    await save_turn("s", "query", "answer", redis)
    redis.lrange.return_value = stored
    messages = await load_history("s", redis)

    assert len(messages) == 2
    assert messages[0].content == "query"
    assert messages[1].content == "answer"
