"""Web-Search agent node — DuckDuckGo (default) or Tavily."""

from __future__ import annotations

import asyncio
import logging
import time

from duckduckgo_search import DDGS

from rag_platform.agents.state import AgentState
from rag_platform.config import get_settings

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5
_TIMEOUT = 15  # seconds; container may have no outbound internet


def _search_duckduckgo(query: str) -> list[dict]:
    results = []
    with DDGS(timeout=_TIMEOUT) as ddgs:
        for hit in ddgs.text(query, max_results=_MAX_RESULTS):
            results.append({
                "text": hit.get("body", ""),
                "body": hit.get("body", ""),
                "source_url": hit.get("href", ""),
                "title": hit.get("title", ""),
            })
    return results


def _search_tavily(query: str, api_key: str) -> list[dict]:
    from tavily import TavilyClient  # optional dep
    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=_MAX_RESULTS)
    return [
        {
            "text": r.get("content", ""),
            "body": r.get("content", ""),
            "source_url": r.get("url", ""),
            "title": r.get("title", ""),
        }
        for r in response.get("results", [])
    ]


async def web_search_node(state: AgentState) -> dict:
    settings = get_settings()
    provider = settings.web_search_provider

    if provider == "tavily":
        if not settings.tavily_api_key:
            logger.warning(
                "web_search_provider=tavily but TAVILY_API_KEY is empty — falling back to DuckDuckGo"
            )
            provider = "duckduckgo"

    logger.info(
        "Web search: starting",
        extra={"query": state["query"][:120], "provider": provider, "timeout_s": _TIMEOUT},
    )
    t0 = time.perf_counter()

    def _search() -> list[dict]:
        if provider == "tavily":
            return _search_tavily(state["query"], settings.tavily_api_key)
        return _search_duckduckgo(state["query"])

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_search), timeout=_TIMEOUT + 5
        )
        logger.info(
            "Web search: complete",
            extra={
                "results": len(results),
                "provider": provider,
                "duration_ms": round((time.perf_counter() - t0) * 1000),
            },
        )
    except Exception as exc:
        logger.warning(
            "Web search: failed/timeout",
            extra={
                "error": str(exc),
                "provider": provider,
                "duration_ms": round((time.perf_counter() - t0) * 1000),
            },
        )
        results = []

    return {"web_results": results}

