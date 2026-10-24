"""Unit tests for the /ws/chat WebSocket endpoint (hermetic — fake graph + redis)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag_platform.agents.state import AgentState
from rag_platform.main import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain(ws, *, until: str) -> tuple[list[dict], dict]:
    """Receive messages, skipping 'status' frames, until the target type arrives.

    Returns (skipped_status_msgs, target_msg).
    """
    skipped: list[dict] = []
    while True:
        msg = json.loads(ws.receive_text())
        if msg["type"] == "status":
            skipped.append(msg)
            continue
        if msg["type"] == until:
            return skipped, msg
        # Unexpected message type — return it so the caller can assert.
        return skipped, msg


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeGraph:
    async def ainvoke(self, state: AgentState) -> dict:
        return {
            **state,
            "final_answer": "Apple's FY2024 revenue was $391 billion.",
            "citations": ["0000320193-24-000123"],
        }

    async def astream(self, state: AgentState, stream_mode: str = "values"):
        yield {
            **state,
            "final_answer": "Apple's FY2024 revenue was $391 billion.",
            "citations": ["0000320193-24-000123"],
        }


class _FakeRedis:
    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        pass

    async def rpush(self, key: str, value: str) -> None:
        pass

    async def lrange(self, key: str, start: int, end: int) -> list:
        return []


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def ws_client() -> TestClient:
    """TestClient with fake graph + redis pre-loaded into app.state."""
    app = create_app()
    app.state.graph = _FakeGraph()
    app.state.redis = _FakeRedis()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ws_chat_returns_token_then_done(ws_client: TestClient) -> None:
    with ws_client.websocket_connect("/ws/chat?api_key=change-me") as ws:
        ws.send_text(json.dumps({
            "query": "What was Apple's total revenue in FY2024?",
            "session_id": "test-session",
        }))
        _, token_msg = _drain(ws, until="token")
        _, done_msg = _drain(ws, until="done")

    assert token_msg["type"] == "token"
    assert "391" in token_msg["content"]
    assert done_msg["type"] == "done"
    assert done_msg["final_answer"] == "Apple's FY2024 revenue was $391 billion."
    assert done_msg["citations"] == ["0000320193-24-000123"]


def test_ws_chat_empty_query_is_ignored(ws_client: TestClient) -> None:
    """An empty query must not trigger the graph; the loop waits for the next message."""
    with ws_client.websocket_connect("/ws/chat?api_key=change-me") as ws:
        ws.send_text(json.dumps({"query": "  ", "session_id": "test"}))
        # Send a real query so the loop doesn't block forever.
        ws.send_text(json.dumps({"query": "What was Apple's revenue?", "session_id": "test"}))
        _, token_msg = _drain(ws, until="token")
        _, done_msg = _drain(ws, until="done")

    assert token_msg["type"] == "token"
    assert done_msg["type"] == "done"


def test_ws_chat_cached_response_served_directly(ws_client: TestClient) -> None:
    """Second identical query hits the Redis cache and skips the graph."""
    cached_answer = "Cached: Apple revenue $391B."

    class _CachingRedis(_FakeRedis):
        def __init__(self) -> None:
            self._store: dict = {}  # instance variable — no shared state between tests

        async def get(self, key: str):
            return self._store.get(key)

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            self._store[key] = value

    ws_client.app.state.redis = _CachingRedis()

    # First request — populates cache (drain all messages including status frames).
    with ws_client.websocket_connect("/ws/chat?api_key=change-me") as ws:
        ws.send_text(json.dumps({"query": "Apple revenue?", "session_id": "s1"}))
        _drain(ws, until="done")

    # Manually inject a known cached value.
    import hashlib
    key = f"llm:cache:{hashlib.sha256('apple revenue?'.encode()).hexdigest()}"
    ws_client.app.state.redis._store[key] = json.dumps(
        {"final_answer": cached_answer, "citations": []}
    )

    # Second request — must come from cache (may receive a status frame first).
    with ws_client.websocket_connect("/ws/chat?api_key=change-me") as ws:
        ws.send_text(json.dumps({"query": "Apple revenue?", "session_id": "s2"}))
        _, token_msg = _drain(ws, until="token")
        _, done_msg = _drain(ws, until="done")

    assert token_msg["content"] == cached_answer
    assert done_msg["final_answer"] == cached_answer
