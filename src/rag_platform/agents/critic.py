"""Critic agent node — groundedness check with self-correction loop."""

from __future__ import annotations

import logging
import time

from rag_platform.agents.state import MAX_REVISIONS, AgentState
from rag_platform.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_PROMPT = """You are a strict fact-checker for a financial Q&A system.

Draft answer:
{draft}

Source chunks:
{chunks}

Instructions:
- If every factual claim in the draft is supported by the source chunks, respond with exactly: APPROVED
- If any claim is unsupported or contradicts the sources, respond with: REVISE: <specific feedback>

Be strict. Numbers, dates, and company names must match the sources exactly."""


def make_critic_node(llm: LLMProvider):
    """Return a critic node bound to the given LLM."""

    async def critic_node(state: AgentState) -> dict:
        logger.info(
            "Critic: evaluating draft",
            extra={"draft_len": len(state.get("draft_answer", ""))},
        )
        t0 = time.perf_counter()
        qdrant_chunks = [c["text"] for c in state["retrieved_chunks"]]
        web_chunks = [r.get("body", r.get("text", "")) for r in state.get("web_results", [])]
        source_texts = qdrant_chunks or web_chunks  # fall back to web results for web-routed queries
        chunks_text = "\n---\n".join(source_texts) if source_texts else "(no source chunks available)"
        prompt = _PROMPT.format(draft=state["draft_answer"], chunks=chunks_text)

        response = ""
        async for token in llm.astream(prompt):
            response += token

        response = response.strip()
        duration_ms = round((time.perf_counter() - t0) * 1000)

        if response.startswith("APPROVED"):
            logger.info("Critic: APPROVED", extra={"duration_ms": duration_ms})
            return {"final_answer": state["draft_answer"], "critic_feedback": ""}

        feedback = response.removeprefix("REVISE:").strip()
        new_count = state.get("revision_count", 0) + 1
        logger.info(
            "Critic: REVISE",
            extra={
                "feedback": feedback[:200],
                "revision_count": new_count,
                "duration_ms": duration_ms,
            },
        )
        result: dict = {"critic_feedback": feedback, "revision_count": new_count}
        if new_count >= MAX_REVISIONS:
            # No more retries — surface the best draft we have.
            logger.info("Critic: max revisions reached, accepting draft")
            result["final_answer"] = state["draft_answer"]
        else:
            # Clear final_answer so _route_critic routes back to synthesizer,
            # not to END (LangGraph merges state — previous value persists otherwise).
            result["final_answer"] = ""
        return result

    return critic_node
