"""Shared state schema for the LangGraph agent graph."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage

MAX_REVISIONS = 2


class AgentState(TypedDict):
    query: str
    chat_history: list[BaseMessage]
    session_id: str
    needs_web: bool
    retrieved_chunks: list[dict]
    web_results: list[dict]
    draft_answer: str
    final_answer: str
    citations: list[str]
    critic_feedback: str
    revision_count: int