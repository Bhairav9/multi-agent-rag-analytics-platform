"""Multi-turn conversation history stored in Redis lists."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from rag_platform.cache.redis_client import RedisClient

_MAX_TURNS = 10
_TTL = 3600


def _history_key(session_id: str) -> str:
    return f"session:{session_id}:history"


async def load_history(session_id: str, redis: RedisClient) -> list[BaseMessage]:
    raw_items = await redis.lrange(_history_key(session_id), -_MAX_TURNS * 2, -1)
    messages: list[BaseMessage] = []
    for item in raw_items:
        data = json.loads(item)
        if data["role"] == "human":
            messages.append(HumanMessage(content=data["content"]))
        else:
            messages.append(AIMessage(content=data["content"]))
    return messages


async def save_turn(session_id: str, query: str, answer: str, redis: RedisClient) -> None:
    key = _history_key(session_id)
    await redis.rpush(key, json.dumps({"role": "human", "content": query}))
    await redis.rpush(key, json.dumps({"role": "ai", "content": answer}))
    # Reset TTL on every write so idle sessions expire after 1 hour.
    await redis.expire(key, _TTL)
