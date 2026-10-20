"""Critic agent node — groundedness check with self-correction loop."""

from __future__ import annotations

from rag_platform.agents.state import AgentState
from rag_platform.llm.provider import LLMProvider

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
        chunks_text = "\n---\n".join(c["text"] for c in state["retrieved_chunks"])
        prompt = _PROMPT.format(draft=state["draft_answer"], chunks=chunks_text)

        response = ""
        async for token in llm.astream(prompt):
            response += token

        response = response.strip()

        if response.startswith("APPROVED"):
            return {"final_answer": state["draft_answer"], "critic_feedback": ""}

        feedback = response.removeprefix("REVISE:").strip()
        return {
            "critic_feedback": feedback,
            "revision_count": state.get("revision_count", 0) + 1,
        }

    return critic_node
