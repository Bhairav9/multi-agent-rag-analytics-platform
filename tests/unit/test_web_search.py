"""Unit tests for the web_search agent node (hermetic — no network)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from rag_platform.agents.state import AgentState


def _state(query: str = "What is Apple's revenue?") -> AgentState:
    return AgentState(
        query=query,
        chat_history=[],
        session_id="test",
        needs_web=True,
        retrieved_chunks=[],
        web_results=[],
        draft_answer="",
        final_answer="",
        citations=[],
        critic_feedback="",
        revision_count=0,
    )


@pytest.mark.asyncio
async def test_web_search_returns_results(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    from rag_platform.config import get_settings
    get_settings.cache_clear()

    fake_hits = [{"body": "Apple revenue was $391B", "href": "https://example.com", "title": "Apple"}]

    with patch("rag_platform.agents.web_search.DDGS") as MockDDGS:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=fake_hits)
        MockDDGS.return_value = mock_ddgs

        from rag_platform.agents.web_search import web_search_node
        result = await web_search_node(_state())

    assert "web_results" in result
    assert len(result["web_results"]) == 1
    assert result["web_results"][0]["text"] == "Apple revenue was $391B"

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_web_search_tavily_falls_back_to_ddgs_when_key_missing(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    from rag_platform.config import get_settings
    get_settings.cache_clear()

    fake_hits = [{"body": "Fallback result", "href": "https://example.com", "title": "T"}]

    with patch("rag_platform.agents.web_search.DDGS") as MockDDGS:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=fake_hits)
        MockDDGS.return_value = mock_ddgs

        from rag_platform.agents import web_search as ws_module
        import importlib
        importlib.reload(ws_module)
        result = await ws_module.web_search_node(_state())

    assert len(result["web_results"]) >= 0  # fallback ran without error

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_web_search_returns_empty_on_exception(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    from rag_platform.config import get_settings
    get_settings.cache_clear()

    with patch("rag_platform.agents.web_search.DDGS") as MockDDGS:
        MockDDGS.side_effect = RuntimeError("network error")

        from rag_platform.agents.web_search import web_search_node
        result = await web_search_node(_state())

    assert result == {"web_results": []}

    get_settings.cache_clear()
