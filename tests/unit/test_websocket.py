"""Unit tests for WebSocket chat endpoint (hermetic — mocked graph + Redis)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from rag_platform.main import create_app

_MOCK_RESULT = {
    "final_answer": "Apple's FY2024 revenue was $391B.",
    "citations": ["0000320193-24-000123"],
}

_FAKE_REDIS = MagicMock(
    get=AsyncMock(return_value=None),
    set=AsyncMock(),
    rpush=AsyncMock(),
    lrange=AsyncMock(return_value=[]),
)


def _make_patches():
    return [
        patch(
            "rag_platform.api.websocket.build_graph",
            new_callable=AsyncMock,
            return_value=AsyncMock(ainvoke=AsyncMock(return_value=_MOCK_RESULT)),
        ),
        patch("rag_platform.api.websocket.get_redis_client", return_value=_FAKE_REDIS),
        patch("rag_platform.api.websocket.FastEmbedDense"),
        patch("rag_platform.api.websocket.FastEmbedSparse"),
        patch("rag_platform.api.websocket.QdrantClient"),
        patch("rag_platform.api.websocket.get_llm_provider"),
    ]


def test_websocket_returns_token_and_done() -> None:
    app = create_app()
    patches = _make_patches()
    for p in patches:
        p.start()

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"query": "What is Apple's revenue?", "session_id": "test"})
                messages = [ws.receive_json(), ws.receive_json()]
    finally:
        for p in patches:
            p.stop()

    types = [m["type"] for m in messages]
    assert "token" in types
    assert "done" in types


def test_websocket_done_message_contains_answer() -> None:
    app = create_app()
    patches = _make_patches()
    for p in patches:
        p.start()

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"query": "What is Apple's revenue?", "session_id": "test"})
                messages = [ws.receive_json(), ws.receive_json()]
    finally:
        for p in patches:
            p.stop()

    done = next(m for m in messages if m["type"] == "done")
    assert "391B" in done["final_answer"]
    assert done["citations"] == ["0000320193-24-000123"]


def test_websocket_empty_query_is_skipped() -> None:
    """Empty query should not invoke the graph — send a real query after to confirm connection still works."""
    app = create_app()
    patches = _make_patches()
    for p in patches:
        p.start()

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"query": "", "session_id": "test"})
                ws.send_json({"query": "What is Apple's revenue?", "session_id": "test"})
                messages = [ws.receive_json(), ws.receive_json()]
    finally:
        for p in patches:
            p.stop()

    types = [m["type"] for m in messages]
    assert "token" in types
    assert "done" in types
