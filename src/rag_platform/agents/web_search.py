"""Web-Search agent node — DuckDuckGo (free, no API key)."""

from __future__ import annotations

from duckduckgo_search import DDGS

from rag_platform.agents.state import AgentState

_MAX_RESULTS = 5


async def web_search_node(state: AgentState) -> dict:
    results = []

    with DDGS() as ddgs:
        for hit in ddgs.text(state["query"], max_results=_MAX_RESULTS):
            results.append({
                "text": hit.get("body", ""),
                "source_url": hit.get("href", ""),
                "title": hit.get("title", ""),
            })

    return {"web_results": results}

