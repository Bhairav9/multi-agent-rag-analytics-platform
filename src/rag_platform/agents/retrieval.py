"""Retrieval agent node — hybrid Qdrant search."""

from __future__ import annotations

import logging
import re
import time

from rag_platform.agents.state import AgentState
from rag_platform.config import get_settings
from rag_platform.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# Maps name/alias -> canonical ticker stored in Qdrant payload.
_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL",
    "tesla": "TSLA", "tsla": "TSLA",
    "jpmorgan": "JPM", "jpm": "JPM",
}


def _detect_ticker(query: str) -> str | None:
    """Return the single ticker the query is about, or None if ambiguous/unknown."""
    q = query.lower()
    # Phrase check first so "jp morgan" (two words) maps correctly.
    if "jp morgan" in q:
        return "JPM"
    words = re.findall(r"[a-z]+", q)
    found = {_TICKER_MAP[w] for w in words if w in _TICKER_MAP}
    # Only filter when exactly one company is identified.
    return found.pop() if len(found) == 1 else None


def make_retrieval_node(retriever: HybridRetriever):
    """Return a retrieval node bound to the given retriever."""

    async def retrieval_node(state: AgentState) -> dict:
        settings = get_settings()
        query = state["query"]
        ticker = _detect_ticker(query)
        logger.info(
            "Retrieval: searching corpus",
            extra={"query": query[:120], "top_k": settings.retrieval_top_k, "ticker_filter": ticker},
        )
        t0 = time.perf_counter()
        results = retriever.retrieve(query, top_k=settings.retrieval_top_k, ticker=ticker)
        chunks = [{"text": r.text, "score": r.score, **r.metadata} for r in results]
        logger.info(
            "Retrieval: complete",
            extra={
                "chunks_returned": len(chunks),
                "duration_ms": round((time.perf_counter() - t0) * 1000),
            },
        )
        return {"retrieved_chunks": chunks}

    return retrieval_node