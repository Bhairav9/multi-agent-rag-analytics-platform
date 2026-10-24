"""Unit tests for the LangGraph agent graph (hermetic — fake LLM + fake retriever)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from rag_platform.agents.graph import build_graph
from rag_platform.agents.state import AgentState
from rag_platform.retrieval.hybrid import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 10, ticker: str | None = None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="Apple's FY2024 revenue was $391 billion.",
                score=0.95,
                metadata={"ticker": "AAPL", "item": "7", "accession": "0000320193-24-000123"},
            )
        ]


class FakeLLM:
    model = "fake"

    async def ainvoke_messages(self, messages) -> str:
        return '{"needs_web": false}'

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        if "fact-checker" in prompt:
            yield "APPROVED"
        else:
            yield "Apple's FY2024 revenue was $391 billion [1]."


def _initial_state(query: str = "What was Apple's revenue in FY2024?") -> AgentState:
    return AgentState(
        query=query,
        chat_history=[],
        session_id="test-session",
        needs_web=False,
        retrieved_chunks=[],
        web_results=[],
        draft_answer="",
        final_answer="",
        citations=[],
        critic_feedback="",
        revision_count=0,
    )


@pytest.mark.asyncio
async def test_graph_returns_final_answer() -> None:
    graph = await build_graph(FakeLLM(), FakeRetriever())
    result = await graph.ainvoke(_initial_state())

    assert result["final_answer"] != ""
    assert "391" in result["final_answer"]


@pytest.mark.asyncio
async def test_graph_critic_approves_without_revision() -> None:
    graph = await build_graph(FakeLLM(), FakeRetriever())
    result = await graph.ainvoke(_initial_state())

    assert result["revision_count"] == 0
    assert result["critic_feedback"] == ""


@pytest.mark.asyncio
async def test_graph_retrieved_chunks_populated() -> None:
    graph = await build_graph(FakeLLM(), FakeRetriever())
    result = await graph.ainvoke(_initial_state())

    assert len(result["retrieved_chunks"]) >= 1
    assert result["retrieved_chunks"][0]["ticker"] == "AAPL"
