# Phase 3 — Multi-Agent LangGraph

> Goal: end this phase with a working `build_graph()` function that returns a compiled LangGraph `StateGraph` — with five nodes (Supervisor, Retrieval, WebSearch, Critic, Synthesizer) wired together by conditional edges including a Critic self-correction loop (max 2 iterations). Calling `await graph.ainvoke({"query": "...", ...})` returns a `final_answer` with inline citations. Unit tests are hermetic (no Qdrant, no Ollama, no network). **No WebSocket streaming yet — that is Phase 4.** Just the agent graph.

---

## Task checklist

| # | Task | File(s) | Status |
|---|---|---|---|
| 1 | `AgentState` TypedDict | `src/rag_platform/agents/state.py` | ⬜ |
| 2 | Supervisor node | `src/rag_platform/agents/supervisor.py` | ⬜ |
| 3 | Retrieval agent node | `src/rag_platform/agents/retrieval.py` | ⬜ |
| 4 | Web-Search agent node | `src/rag_platform/agents/web_search.py` | ⬜ |
| 5 | Critic agent node | `src/rag_platform/agents/critic.py` | ⬜ |
| 6 | Synthesizer node | `src/rag_platform/agents/synthesizer.py` | ⬜ |
| 7 | Graph assembly + conditional edges | `src/rag_platform/agents/graph.py` | ⬜ |
| 8 | Agent unit tests (fake LLM + fake retriever) | `tests/unit/test_agents.py` | ⬜ |

Phase 3 only touches `agents/`. `cache/`, `evaluation/`, `ui/` stay empty until their phases.

---

## Core concepts (read once before you type anything)

### 1. What LangGraph is — and why it replaces a plain LangChain chain

A plain LangChain chain is a **linear pipeline**: A → B → C. That is enough for a single-shot RAG (retrieve → generate). But this platform needs:
- A **conditional branch** (invoke web search only if the Supervisor decides it is needed)
- A **loop** (Critic can send the answer back to Synthesizer for revision, up to 2 times)
- **Shared state** that every node can read and write

LangGraph models this as a **directed graph** where:
- **Nodes** are async functions `(state: AgentState) -> dict` — they read the current state, do work, and return a partial update.
- **Edges** are transitions between nodes — either unconditional (`A → B always`) or conditional (`A → B or A → C depending on state`).
- **State** is a `TypedDict` that accumulates updates from every node as the graph runs.

This is the core resume claim: *"multi-agent orchestration with LangGraph"* — a graph is fundamentally different from a chain, and interviewers know it.

*Used in:* Task 7.

---

### 2. `AgentState` — the shared blackboard

Every node reads from and writes to the same `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    query: str                        # the user's question
    chat_history: list[BaseMessage]   # prior turns
    session_id: str                   # for Redis in Phase 4
    needs_web: bool                   # Supervisor decision
    retrieved_chunks: list[dict]      # from Retrieval node
    web_results: list[dict]           # from WebSearch node
    draft_answer: str                 # from Synthesizer
    final_answer: str                 # from Critic (when approved)
    citations: list[str]              # source references
    critic_feedback: str              # Critic's revision note
    revision_count: int               # capped at MAX_REVISIONS = 2
```

A node returns only the keys it updates — LangGraph merges the partial dict into the full state. So `retrieval_node` returns `{"retrieved_chunks": [...]}` and leaves everything else untouched.

*Used in:* Task 1.

---

### 3. The five nodes and what each does

```
START
  │
  ▼
Supervisor ──────────────────────────┐
  │                                  │ (if needs_web=True)
  ▼                                  ▼
Retrieval                        WebSearch
  │                                  │
  └──────────────┬───────────────────┘
                 ▼
            Synthesizer ◄────────────┐
                 │                   │ (if revision_count < 2
                 ▼                   │  and critic_feedback exists)
               Critic ───────────────┘
                 │
                 ▼ (approved or revision_count == 2)
                END
```

| Node | Input from state | Output to state | What it does |
|---|---|---|---|
| **Supervisor** | `query`, `chat_history` | `needs_web` | LLM decides if web search is needed |
| **Retrieval** | `query` | `retrieved_chunks` | Calls `HybridRetriever.retrieve()` |
| **WebSearch** | `query` | `web_results` | DuckDuckGo search (only if `needs_web`) |
| **Synthesizer** | `retrieved_chunks`, `web_results`, `critic_feedback` | `draft_answer`, `citations` | LLM writes the answer |
| **Critic** | `draft_answer`, `retrieved_chunks` | `final_answer`, `critic_feedback`, `revision_count` | LLM checks groundedness; approves or requests revision |

*Used in:* Tasks 2–6.

---

### 4. Conditional edges — the routing logic

LangGraph supports two kinds of edges:

```python
# Unconditional — always goes from A to B
graph.add_edge("supervisor", "retrieval")

# Conditional — a router function decides the next node
graph.add_conditional_edges(
    "supervisor",
    route_after_supervisor,          # function returns a node name
    {"retrieval": "retrieval", "web_search": "web_search"},
)
```

The router function reads the state and returns the name of the next node:

```python
def route_after_supervisor(state: AgentState) -> str:
    return "web_search" if state["needs_web"] else "retrieval"
```

The Critic loop works the same way:

```python
def route_after_critic(state: AgentState) -> str:
    if state["critic_feedback"] and state["revision_count"] < MAX_REVISIONS:
        return "synthesizer"   # revise
    return END                 # done
```

*Used in:* Task 7.

---

### 5. Supervisor uses structured output — not free-form text

The Supervisor LLM call returns a structured Pydantic object, not a raw string:

```python
class SupervisorDecision(BaseModel):
    needs_web: bool
    reasoning: str  # for observability
```

`ChatOllama` supports `.with_structured_output(SupervisorDecision)` — it prompts the model to return JSON matching the schema, then deserializes it. This is safer than parsing free-form LLM output and makes the Supervisor's decision explicit and testable.

*Used in:* Task 2.

---

### 6. Critic prompt engineering — the groundedness gate

The Critic is the key to the ≥94% groundedness claim. Its prompt:

```
You are a Critic reviewing an AI-generated answer about SEC filings.

Answer to review:
{draft_answer}

Source chunks used:
{retrieved_chunks}

Your job:
1. Check every factual claim in the answer against the source chunks.
2. If every claim is supported → respond with: APPROVED
3. If any claim is unsupported → respond with: REVISE: <specific feedback>

Be strict. If a number or date cannot be found in the chunks, flag it.
```

If the Critic says `APPROVED` → the draft becomes the `final_answer`.
If the Critic says `REVISE: ...` → the feedback is stored in `critic_feedback` and the Synthesizer is asked to revise (max 2 times).

*Used in:* Task 5.

---

### 7. Hermetic tests — fake everything

Phase 3 agents talk to:
- `HybridRetriever` (Qdrant + fastembed)
- `LLMProvider` (Ollama)
- DuckDuckGo (web)

None of these should be live in unit tests. The fake versions:

```python
class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        return [RetrievedChunk(text="Apple revenue was $391B.", score=0.9, metadata={"ticker": "AAPL"})]

class FakeLLM:
    model = "fake"
    async def astream(self, prompt: str) -> AsyncIterator[str]:
        yield "The answer is 42."
```

Tests assert: correct state keys updated, critic loop terminates, `final_answer` is non-empty.

*Used in:* Task 8.

---

## Order to type things

1. `state.py` — define `AgentState` first; every other file imports it.
2. `supervisor.py` — needs `LLMProvider`, `AgentState`.
3. `retrieval.py` — needs `HybridRetriever`, `AgentState`.
4. `web_search.py` — needs `AgentState`, DuckDuckGo.
5. `synthesizer.py` — needs `LLMProvider`, `AgentState`.
6. `critic.py` — needs `LLMProvider`, `AgentState`.
7. `graph.py` — assembles all nodes + edges into a compiled graph.
8. `tests/unit/test_agents.py` — fake everything, test graph end-to-end.

---

## Task descriptions

### Task 1 — `src/rag_platform/agents/state.py`

**What you're doing:** Define the shared `AgentState` TypedDict that flows through every graph node.

**Why the project needs it:** LangGraph requires a single state type passed to every node. All inter-agent communication happens through state mutations — no direct calls between agents.

**Done when:** `from rag_platform.agents.state import AgentState` works and all fields type-check cleanly.

```python
"""Shared state schema for the LangGraph agent graph."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage

MAX_REVISIONS = 2


class AgentState(TypedDict):
    query: str
    chat_history: list[BaseMessage]
    session_id: str
    needs_web: bool
    retrieved_chunks: list[dict]
    web_results: list[dict]
    draft_answer: str
    final_answer: str
    citations: list[str]
    critic_feedback: str
    revision_count: int
```

---

### Task 2 — `src/rag_platform/agents/supervisor.py`

**What you're doing:** The entry node — asks the LLM whether web search is needed, returns `needs_web`.

**Done when:** `supervisor_node(state)` returns `{"needs_web": True/False}` based on the query.

```python
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
```

---

### Task 3 — `src/rag_platform/agents/retrieval.py`

**What you're doing:** The Retrieval node — calls `HybridRetriever.retrieve()` and stores results in state.

**Done when:** `retrieval_node(state)` returns `{"retrieved_chunks": [...]}` with dicts containing `text`, `score`, `metadata`.

```python
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
```

---

### Task 4 — `src/rag_platform/agents/web_search.py`

**What you're doing:** The WebSearch node — runs a DuckDuckGo search and stores results in state. Only invoked when `needs_web=True`.

**Done when:** `web_search_node(state)` returns `{"web_results": [...]}` with text snippets.

```python
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
```

---

### Task 5 — `src/rag_platform/agents/critic.py`

**What you're doing:** The Critic node — checks the draft answer for unsupported claims. Approves or requests a revision (max `MAX_REVISIONS` times).

**Done when:** `critic_node(state)` sets `final_answer` when approved, or sets `critic_feedback` + increments `revision_count` when requesting revision.

```python
"""Critic agent node — groundedness check with self-correction loop."""

from __future__ import annotations

from rag_platform.agents.state import MAX_REVISIONS, AgentState
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
```

---

### Task 6 — `src/rag_platform/agents/synthesizer.py`

**What you're doing:** The Synthesizer node — generates the final grounded answer with citations from retrieved chunks and optional web results.

**Done when:** `synthesizer_node(state)` returns `{"draft_answer": "...", "citations": [...]}`.

```python
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
```

---

### Task 7 — `src/rag_platform/agents/graph.py`

**What you're doing:** Assemble all nodes into a compiled LangGraph `StateGraph` with conditional edges and the Critic self-correction loop.

**Done when:** `build_graph(llm, retriever)` returns a compiled graph that accepts `AgentState` and produces `final_answer`.

```python
"""LangGraph agent graph — assembles all nodes and edges."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag_platform.agents.critic import make_critic_node
from rag_platform.agents.retrieval import make_retrieval_node
from rag_platform.agents.state import MAX_REVISIONS, AgentState
from rag_platform.agents.supervisor import make_supervisor_node
from rag_platform.agents.synthesizer import make_synthesizer_node
from rag_platform.agents.web_search import web_search_node
from rag_platform.llm.provider import LLMProvider
from rag_platform.retrieval.hybrid import HybridRetriever


def _route_after_supervisor(state: AgentState) -> str:
    return "web_search" if state.get("needs_web") else "retrieval"


def _route_after_critic(state: AgentState) -> str:
    if state.get("critic_feedback") and state.get("revision_count", 0) < MAX_REVISIONS:
        return "synthesizer"
    return END


async def build_graph(llm: LLMProvider, retriever: HybridRetriever):
    """Build and compile the multi-agent LangGraph graph."""
    supervisor = await make_supervisor_node(llm)
    retrieval = make_retrieval_node(retriever)
    synthesizer = make_synthesizer_node(llm)
    critic = make_critic_node(llm)

    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("retrieval", retrieval)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("critic", critic)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"retrieval": "retrieval", "web_search": "web_search"},
    )
    graph.add_edge("web_search", "retrieval")
    graph.add_edge("retrieval", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"synthesizer": "synthesizer", END: END},
    )

    return graph.compile()
```

---

### Task 8 — `tests/unit/test_agents.py`

**What you're doing:** End-to-end graph test using fake LLM + fake retriever — no Ollama, no Qdrant, no network.

**Done when:** `await graph.ainvoke(initial_state)` returns a state with non-empty `final_answer` and the Critic loop terminates correctly.

```python
"""Unit tests for the LangGraph agent graph (hermetic — fake LLM + fake retriever)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from rag_platform.agents.graph import build_graph
from rag_platform.agents.state import AgentState
from rag_platform.retrieval.hybrid import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="Apple's FY2024 revenue was $391 billion.",
                score=0.95,
                metadata={"ticker": "AAPL", "item": "7", "accession": "0000320193-24-000123"},
            )
        ]


class FakeLLM:
    model = "fake"

    class _FakeChatModel:
        def with_structured_output(self, schema):
            async def ainvoke(messages):
                return schema(needs_web=False, reasoning="corpus is sufficient")
            return type("StructuredLLM", (), {"ainvoke": staticmethod(ainvoke)})()

    _llm = _FakeChatModel()

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        if "fact-checker" in prompt or "Critic" in prompt:
            yield "APPROVED"
        else:
            yield "Apple's FY2024 revenue was $391 billion [1]."


@pytest.mark.asyncio
async def test_graph_returns_final_answer() -> None:
    graph = await build_graph(FakeLLM(), FakeRetriever())
    initial_state: AgentState = {
        "query": "What was Apple's revenue in FY2024?",
        "chat_history": [],
        "session_id": "test-session",
        "needs_web": False,
        "retrieved_chunks": [],
        "web_results": [],
        "draft_answer": "",
        "final_answer": "",
        "citations": [],
        "critic_feedback": "",
        "revision_count": 0,
    }
    result = await graph.ainvoke(initial_state)

    assert result["final_answer"] != ""
    assert "391" in result["final_answer"]


@pytest.mark.asyncio
async def test_graph_critic_approves_without_revision() -> None:
    graph = await build_graph(FakeLLM(), FakeRetriever())
    initial_state: AgentState = {
        "query": "What was Apple's revenue?",
        "chat_history": [],
        "session_id": "test-session",
        "needs_web": False,
        "retrieved_chunks": [],
        "web_results": [],
        "draft_answer": "",
        "final_answer": "",
        "citations": [],
        "critic_feedback": "",
        "revision_count": 0,
    }
    result = await graph.ainvoke(initial_state)

    assert result["revision_count"] == 0
    assert result["critic_feedback"] == ""
```

---

## Definition of done for Phase 3

- `await build_graph(llm, retriever)` returns a compiled `StateGraph`.
- `await graph.ainvoke(state)` with a real query (Docker stack up, Ollama running) returns a non-empty `final_answer` with citations.
- Critic self-correction loop terminates — either on `APPROVED` or after `MAX_REVISIONS = 2` iterations.
- `uv run pytest` (hermetic) is green with ≥80% coverage on `agents/`.
- `ruff check .`, `ruff format --check .`, `mypy src` all pass.
- Phase 4 can now call `build_graph()` and stream tokens via WebSocket.
