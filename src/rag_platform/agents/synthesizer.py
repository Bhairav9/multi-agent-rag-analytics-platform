"""Synthesizer node — generates a grounded answer with citations."""

from __future__ import annotations

import logging
import time

from rag_platform.agents.state import AgentState
from rag_platform.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_PROMPT = """You are a financial analyst assistant. Answer the user's question using ONLY the provided sources.
Include inline citations like [1], [2] referring to the source index.
If the sources do not contain enough information, say so — do not invent facts.

Question: {query}

Sources:
{sources}

{revision_note}

Answer:"""


def make_synthesizer_node(llm: LLMProvider):
    """Return a synthesizer node bound to the given LLM."""

    async def synthesizer_node(state: AgentState) -> dict:
        all_chunks = state.get("retrieved_chunks", []) + state.get("web_results", [])
        revision = state.get("revision_count", 0)
        logger.info(
            "Synthesizer: generating answer",
            extra={
                "query": state["query"][:120],
                "sources": len(all_chunks),
                "revision": revision,
            },
        )
        t0 = time.perf_counter()

        sources = "\n\n".join(
            f"[{i + 1}] {c['text']}" for i, c in enumerate(all_chunks)
        )
        citations = [
            ref for c in all_chunks
            if (ref := c.get("source_url") or c.get("accession", ""))
        ]

        revision_note = ""
        if state.get("critic_feedback"):
            revision_note = f"Previous answer was rejected. Feedback: {state['critic_feedback']}\nPlease revise."

        prompt = _PROMPT.format(
            query=state["query"],
            sources=sources,
            revision_note=revision_note,
        )

        answer = ""
        async for token in llm.astream(prompt):
            answer += token

        logger.info(
            "Synthesizer: answer ready",
            extra={
                "answer_len": len(answer),
                "duration_ms": round((time.perf_counter() - t0) * 1000),
            },
        )
        answer = answer.strip()
        return {"draft_answer": answer, "final_answer": answer, "citations": citations}

    return synthesizer_node
