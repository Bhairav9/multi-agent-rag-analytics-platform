"""Supervisor node — routes query to retrieval-only or retrieval + web search."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rag_platform.agents.state import AgentState
from rag_platform.llm.provider import LLMProvider

_SYSTEM = (
    "You are a router for a financial document Q&A system. "
    "Decide if the user's question requires a live web search "
    "(e.g. recent news, current prices, events after your training cutoff) "
    "or if the internal SEC filing corpus is sufficient. "
    "Respond with JSON only."
)


class SupervisorDecision(BaseModel):
    needs_web: bool
    reasoning: str


async def make_supervisor_node(llm: LLMProvider):
    """Return a supervisor node bound to the given LLM."""
    structured_llm = llm._llm.with_structured_output(SupervisorDecision)

    async def supervisor_node(state: AgentState) -> dict:
        messages = [HumanMessage(content=f"{_SYSTEM}\n\nUser question: {state['query']}")]
        decision: SupervisorDecision = await structured_llm.ainvoke(messages)
        return {"needs_web": decision.needs_web}

    return supervisor_node


