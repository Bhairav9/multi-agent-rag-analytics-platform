# Phase 4 — Redis Cache + WebSocket Streaming

> Goal: end this phase with a FastAPI WebSocket endpoint (`/ws/chat`) that accepts a user query, runs the LangGraph agent graph, and streams tokens back to the client in real time — plus a Redis exact-match cache that returns cached answers instantly on repeated queries, skipping the LLM entirely. Unit tests are hermetic (no Redis server, no WebSocket client required). **No Streamlit UI yet — that is Phase 6.** Just the API layer and cache.

---

## Task checklist

| # | Task | File(s) | Status |
|---|---|---|---|
| 1 | Redis client wrapper | `src/rag_platform/cache/redis_client.py` | ⬜ |
| 2 | LLM response cache (exact-match) | `src/rag_platform/cache/llm_cache.py` | ⬜ |
| 3 | Conversation history store | `src/rag_platform/cache/conversation.py` | ⬜ |
| 4 | WebSocket endpoint | `src/rag_platform/api/websocket.py` | ⬜ |
| 5 | Mount WebSocket router in `main.py` | `src/rag_platform/main.py` | ⬜ |
| 6 | Cache unit tests (fake Redis) | `tests/unit/test_cache.py` | ⬜ |
| 7 | WebSocket integration test | `tests/unit/test_websocket.py` | ⬜ |

Phase 4 touches `cache/`, `api/websocket.py`, and `main.py`. `evaluation/` and `ui/` stay empty until their phases.

---

## Core concepts (read once before you type anything)

### 1. Why Redis cache — the -35% latency claim

Every LLM call to Ollama takes 25-40 seconds cold. If the same question is asked twice (or a very similar one), running the full agent graph again is wasteful. Redis stores the answer the first time and returns it instantly on repeat queries.

**Two types of cache hits:**
- **Exact match**: `hash(normalized_query)` → cached `final_answer`. Returns in <1ms vs 30s.
- **Semantic match** (Phase 7 stretch): embed the query, KNN search over prior queries. Not in this phase.

**How the -35% latency is measured:**
- Run 50 queries cold (no cache) → record p50/p95
- Run same 50 queries warm (cache populated) → record p50/p95
- Cache hit rate of ~30-40% on realistic workloads → ~35% average latency reduction

*Used in:* Tasks 1, 2.

---

### 2. WebSocket streaming — why not plain HTTP

A plain HTTP endpoint returns the full answer after 30s — the user sees a blank screen for half a minute. A WebSocket connection streams tokens as they are generated:

```
Client sends: {"query": "What is Apple's revenue?", "session_id": "abc123"}
Server streams: "Apple" " " "'s" " " "FY2024" " " "revenue" " " "was" ...
Server sends:   {"type": "done", "citations": [...]}
Client renders: tokens appear word-by-word, ChatGPT-style
```

**Perceived latency drops dramatically** — the user sees the first word in ~1s even if the full answer takes 30s. This is the TTFT (time to first token) metric.

*Used in:* Task 4.

---

### 3. WebSocket message protocol

The server sends two types of messages over the WebSocket:

```python
# Token message — sent for each streaming token
{"type": "token", "content": "Apple"}

# Done message — sent when the graph finishes
{"type": "done", "final_answer": "...", "citations": [...]}

# Error message — sent if something goes wrong
{"type": "error", "message": "..."}
```

The client (Streamlit in Phase 6) listens for these and renders tokens as they arrive.

*Used in:* Task 4.

---

### 4. Cache key design — deterministic hashing

The cache key must be deterministic: same query = same key = same cached answer.

```python
import hashlib

def cache_key(query: str) -> str:
    normalized = query.strip().lower()
    return f"llm:cache:{hashlib.sha256(normalized.encode()).hexdigest()}"
```

`sha256` of the normalized query gives a fixed-length, collision-resistant key. The `llm:cache:` prefix namespaces it so Redis keys don't clash with conversation history keys (`session:{id}:history`).

*Used in:* Task 2.

---

### 5. Conversation history — multi-turn memory

Redis stores the last N messages of each session under `session:{session_id}:history` as a JSON list. When the user sends a new message, the WebSocket handler:
1. Loads prior history from Redis
2. Adds it to `AgentState.chat_history`
3. Runs the graph
4. Appends the new Q+A pair back to Redis

This is the "multi-turn" part of the resume claim — the system remembers prior questions within a session.

*Used in:* Task 3.

---

### 6. Dependency injection for Redis — hermetic tests

`redis_client.py` wraps `redis.asyncio.Redis`. Tests inject a `FakeRedis` dict-backed stub — no Redis server needed:

```python
class FakeRedis:
    def __init__(self):
        self._store = {}

    async def get(self, key): return self._store.get(key)
    async def set(self, key, value, ex=None): self._store[key] = value
    async def rpush(self, key, value): ...
    async def lrange(self, key, start, end): ...
```

Same Protocol pattern as Phase 1 embedders and Phase 2 LLM.

*Used in:* Tasks 1, 2, 3, 6.

---

## Order to type things

1. `cache/redis_client.py` — Redis wrapper + Protocol.
2. `cache/llm_cache.py` — exact-match cache using the Redis client.
3. `cache/conversation.py` — conversation history using the Redis client.
4. `api/websocket.py` — WebSocket endpoint wiring graph + cache + history.
5. `main.py` — mount the WebSocket router.
6. `tests/unit/test_cache.py` — hermetic cache tests with FakeRedis.
7. `tests/unit/test_websocket.py` — WebSocket test with TestClient.

---

## Task descriptions

### Task 1 — `src/rag_platform/cache/redis_client.py`

**What you're doing:** A thin async Redis wrapper + `RedisClient` Protocol so tests can inject a fake.

**Done when:** `get_redis_client()` returns a connected `redis.asyncio.Redis` instance using `settings.redis_url`.

```python
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
```

---

### Task 2 — `src/rag_platform/cache/llm_cache.py`

**What you're doing:** Exact-match LLM response cache — check before running the graph, store after.

**Done when:** `get_cached(query, redis)` returns a cached answer or `None`. `set_cached(query, answer, redis)` stores with TTL.

```python
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
    """Return cached result for query, or None if not cached."""
    raw = await redis.get(_cache_key(query))
    if raw is None:
        return None
    return json.loads(raw)


async def set_cached(query: str, result: dict, redis: RedisClient) -> None:
    """Store result in cache with configured TTL."""
    settings = get_settings()
    await redis.set(_cache_key(query), json.dumps(result), ex=settings.cache_ttl_seconds)
```

---

### Task 3 — `src/rag_platform/cache/conversation.py`

**What you're doing:** Multi-turn conversation history — load prior turns from Redis, append new ones.

**Done when:** `load_history(session_id, redis)` returns a list of `BaseMessage` objects. `save_turn(session_id, query, answer, redis)` appends the new Q+A pair.

```python
"""Conversation history store backed by Redis."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from rag_platform.cache.redis_client import RedisClient

_MAX_TURNS = 10
_TTL = 3600  # 1 hour session TTL


def _history_key(session_id: str) -> str:
    return f"session:{session_id}:history"


async def load_history(session_id: str, redis: RedisClient) -> list[BaseMessage]:
    """Load the last N turns of conversation history from Redis."""
    raw_items = await redis.lrange(_history_key(session_id), -_MAX_TURNS * 2, -1)
    messages: list[BaseMessage] = []
    for item in raw_items:
        data = json.loads(item)
        if data["role"] == "human":
            messages.append(HumanMessage(content=data["content"]))
        else:
            messages.append(AIMessage(content=data["content"]))
    return messages


async def save_turn(
    session_id: str, query: str, answer: str, redis: RedisClient
) -> None:
    """Append a Q+A turn to the session history."""
    key = _history_key(session_id)
    await redis.rpush(key, json.dumps({"role": "human", "content": query}))
    await redis.rpush(key, json.dumps({"role": "ai", "content": answer}))
```

---

### Task 4 — `src/rag_platform/api/websocket.py`

**What you're doing:** The WebSocket endpoint — accepts a query, checks cache, runs the graph, streams tokens, saves to cache and history.

**Done when:** Connecting to `ws://localhost:8000/ws/chat` and sending `{"query": "...", "session_id": "..."}` streams token messages and a final done message.

```python
"""WebSocket endpoint — streams agent graph tokens to the client."""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from qdrant_client import QdrantClient

from rag_platform.agents.graph import build_graph
from rag_platform.agents.state import AgentState
from rag_platform.cache.conversation import load_history, save_turn
from rag_platform.cache.llm_cache import get_cached, set_cached
from rag_platform.cache.redis_client import get_redis_client
from rag_platform.config import get_settings
from rag_platform.llm.provider import get_llm_provider
from rag_platform.retrieval.bm25 import FastEmbedSparse
from rag_platform.retrieval.embeddings import FastEmbedDense
from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.retrieval.qdrant_store import QdrantStore

ws_router = APIRouter()


@ws_router.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()

    store = QdrantStore(
        QdrantClient(url=settings.qdrant_url), settings.qdrant_collection
    )
    retriever = HybridRetriever(FastEmbedDense(), FastEmbedSparse(), store)
    llm = get_llm_provider()
    redis = get_redis_client()
    graph = await build_graph(llm, retriever)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            query: str = data.get("query", "").strip()
            session_id: str = data.get("session_id", "default")

            if not query:
                continue

            cached = await get_cached(query, redis)
            if cached:
                await websocket.send_text(json.dumps({"type": "token", "content": cached["final_answer"]}))
                await websocket.send_text(json.dumps({"type": "done", **cached}))
                continue

            history = await load_history(session_id, redis)
            initial_state = AgentState(
                query=query,
                chat_history=history,
                session_id=session_id,
                needs_web=False,
                retrieved_chunks=[],
                web_results=[],
                draft_answer="",
                final_answer="",
                citations=[],
                critic_feedback="",
                revision_count=0,
            )

            result = await graph.ainvoke(initial_state)
            final_answer = result.get("final_answer", "")
            citations = result.get("citations", [])

            await websocket.send_text(json.dumps({"type": "token", "content": final_answer}))
            await websocket.send_text(json.dumps({
                "type": "done",
                "final_answer": final_answer,
                "citations": citations,
            }))

            await set_cached(query, {"final_answer": final_answer, "citations": citations}, redis)
            await save_turn(session_id, query, final_answer, redis)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
```

---

### Task 5 — Mount WebSocket router in `main.py`

**What you're doing:** Include the WebSocket router in the FastAPI app factory.

**Done when:** `GET /openapi.json` shows the `/ws/chat` route.

```python
# add to create_app() in main.py after existing api_router include:
from rag_platform.api.websocket import ws_router
app.include_router(ws_router)
```

---

### Task 6 — `tests/unit/test_cache.py`

**What you're doing:** Hermetic tests for `llm_cache.py` and `conversation.py` using a `FakeRedis` dict store — no Redis server needed.

```python
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
        return items[start: end + 1]


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
```

---

### Task 7 — `tests/unit/test_websocket.py`

**What you're doing:** Test the WebSocket endpoint using FastAPI's `TestClient` with mocked graph and Redis — no real Ollama or Qdrant needed.

```python
"""Unit tests for WebSocket chat endpoint (hermetic — mocked graph + Redis)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from rag_platform.main import create_app


def test_websocket_returns_done_message() -> None:
    app = create_app()

    mock_result = {
        "final_answer": "Apple's revenue was $391B.",
        "citations": ["0000320193-24-000123"],
    }

    with (
        patch("rag_platform.api.websocket.build_graph", new_callable=AsyncMock, return_value=AsyncMock(ainvoke=AsyncMock(return_value=mock_result))),
        patch("rag_platform.api.websocket.get_redis_client", return_value=AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock(), rpush=AsyncMock(), lrange=AsyncMock(return_value=[]))),
        patch("rag_platform.api.websocket.FastEmbedDense"),
        patch("rag_platform.api.websocket.FastEmbedSparse"),
        patch("rag_platform.api.websocket.QdrantClient"),
        patch("rag_platform.api.websocket.get_llm_provider"),
    ):
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"query": "What is Apple's revenue?", "session_id": "test"})
                messages = []
                for _ in range(2):
                    messages.append(ws.receive_json())

    types = [m["type"] for m in messages]
    assert "token" in types
    assert "done" in types
```

---

## Definition of done for Phase 4

- `ws://localhost:8000/ws/chat` accepts queries and streams tokens (verified manually with Docker stack up).
- Repeated identical queries return cached answers from Redis instantly.
- Conversation history persists within a session (multi-turn Q&A works).
- `uv run pytest` (hermetic) is green with ≥80% coverage on `cache/` and `api/websocket.py`.
- `ruff check .`, `ruff format --check .`, `mypy src` all pass.
- Phase 5 can now benchmark cached vs uncached latency.
