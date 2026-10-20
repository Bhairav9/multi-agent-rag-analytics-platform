"""Synthesizer node — generates a grounded answer with citations."""

from __future__ import annotations

from rag_platform.agents.state import AgentState
from rag_platform.llm.provider import LLMProvider

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
        sources = "\n\n".join(
            f"[{i + 1}] {c['text']}" for i, c in enumerate(all_chunks)
        )
        citations = [c.get("source_url", c.get("accession", "")) for c in all_chunks]

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

        return {"draft_answer": answer.strip(), "citations": citations}

    return synthesizer_node
