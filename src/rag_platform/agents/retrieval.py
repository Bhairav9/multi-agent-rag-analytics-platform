"""Retrieval agent node — hybrid Qdrant search."""

from __future__ import annotations

from rag_platform.agents.state import AgentState
from rag_platform.config import get_settings
from rag_platform.retrieval.hybrid import HybridRetriever


def make_retrieval_node(retriever: HybridRetriever):
    """Return a retrieval node bound to the given retriever."""

    async def retrieval_node(state: AgentState) -> dict:
        settings = get_settings()
        results = retriever.retrieve(state["query"], top_k=settings.retrieval_top_k)
        return {
            "retrieved_chunks": [
                {"text": r.text, "score": r.score, **r.metadata} for r in results
            ]
        }

    return retrieval_node