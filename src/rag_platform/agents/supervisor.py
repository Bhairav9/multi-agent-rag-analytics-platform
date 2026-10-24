"""Supervisor node — routes query to retrieval-only or retrieval + web search."""

from __future__ import annotations

import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from rag_platform.agents.state import AgentState
from rag_platform.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a router for a financial Q&A system that has a local corpus of "
    "SEC 10-K and 10-Q filings for AAPL, MSFT, GOOGL, TSLA, and JPM. "
    "Answer needs_web=true ONLY for questions about real-time stock prices, "
    "breaking news from the last few days, or companies NOT in that list. "
    "For ANY question about revenue, earnings, profit, expenses, guidance, "
    "balance sheet, cash flow, or other financial metrics of those five companies, "
    "always answer needs_web=false because the filings already contain the data. "
    'Reply with exactly this JSON and nothing else: {"needs_web": true} or {"needs_web": false}'
)


async def make_supervisor_node(llm: LLMProvider):
    """Return a supervisor node bound to the given LLM."""

    async def supervisor_node(state: AgentState) -> dict:
        logger.info("Supervisor: routing query", extra={"query": state["query"][:120]})
        t0 = time.perf_counter()
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=state["query"]),
        ]
        text = await llm.ainvoke_messages(messages)
        match = re.search(r'"needs_web"\s*:\s*(true|false)', text)
        needs_web = match.group(1) == "true" if match else False
        logger.info(
            "Supervisor: decision",
            extra={
                "needs_web": needs_web,
                "duration_ms": round((time.perf_counter() - t0) * 1000),
            },
        )
        return {"needs_web": needs_web}

    return supervisor_node
