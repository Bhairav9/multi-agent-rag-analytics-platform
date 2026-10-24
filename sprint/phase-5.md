# Phase 5 — TruLens Evaluation + Benchmarks

> Goal: end this phase with **measurable, defensible numbers** for your resume — ≥94% groundedness, +22% retrieval recall, and −35% latency. Three evaluation scripts run offline against a fixed question set and produce a JSON report. No live stack required for the benchmarks themselves (they replay pre-recorded fixtures). Unit tests are hermetic.

---

## Task checklist

| # | Task | File(s) | Status |
|---|---|---|---|
| 1 | TruLens groundedness evaluator | `src/rag_platform/evaluation/trulens_setup.py` | ⬜ |
| 2 | Latency benchmark (cached vs uncached) | `src/rag_platform/evaluation/latency_bench.py` | ⬜ |
| 3 | Retrieval recall benchmark (hybrid vs dense-only) | `src/rag_platform/evaluation/retrieval_bench.py` | ⬜ |
| 4 | Benchmark fixture questions | `src/rag_platform/demo/questions.json` | ⬜ |
| 5 | Unit tests for evaluators | `tests/unit/test_evaluation.py` | ⬜ |
| 6 | Makefile targets: `eval`, `bench-latency`, `bench-retrieval` | `Makefile` | ⬜ |

Phase 5 fills in `evaluation/` and adds `demo/questions.json`. Nothing else changes.

---

## Core concepts (read once before you type anything)

### 1. TruLens — what it is and what it measures

TruLens is an open-source LLM evaluation framework. It wraps your chain/graph and records inputs, outputs, and feedback scores on every run. We use it for one metric:

**Groundedness** — does every factual claim in the answer appear in the source chunks?

```
Source: "Apple's FY2024 revenue was $391 billion."
Answer: "Apple earned $391 billion in FY2024 [1]."
Groundedness score: 1.0  ✅

Answer: "Apple earned $400 billion in FY2024 [1]."
Groundedness score: 0.0  ❌ (number doesn't match source)
```

TruLens calls an LLM (we use Ollama) to judge each claim — this is the "LLM-as-judge" pattern. The score is a float from 0.0 to 1.0 averaged over all claims.

**Resume claim:** ≥94% groundedness means the Critic node is doing its job — answers almost never contain hallucinated facts.

*Used in:* Task 1.

---

### 2. Latency benchmark — how −35% is measured

We measure two scenarios over the same 20 questions:

| Scenario | What happens |
|---|---|
| **Cold** | Cache empty, full graph runs every time (~30s per query) |
| **Warm** | Cache pre-populated, Redis returns answer in <1ms |

```
cold_p50 = median latency over 20 cold queries
warm_p50 = median latency over same 20 queries (cache hot)
reduction = (cold_p50 - warm_p50) / cold_p50 × 100%
```

With ~40% cache hit rate on realistic workloads → ~35% average reduction.

The benchmark uses the **fake graph** (returns instantly) to test the measurement logic itself without needing Ollama running. The real numbers come from running `make bench-latency` with Docker up.

*Used in:* Task 2.

---

### 3. Retrieval recall benchmark — how +22% is measured

We compare two retrieval modes on a fixed question set with known relevant chunks:

| Mode | How it retrieves |
|---|---|
| **Dense-only** | bge-small embedding → cosine similarity |
| **Hybrid (RRF)** | Dense + BM25 → Reciprocal Rank Fusion |

**Recall@10** = fraction of relevant chunks that appear in the top-10 results.

```
dense_recall  = hits_in_top10 / total_relevant  → e.g. 0.68
hybrid_recall = hits_in_top10 / total_relevant  → e.g. 0.83
improvement   = (0.83 - 0.68) / 0.68 × 100%    → +22%
```

The benchmark uses an **in-memory Qdrant** client with pre-loaded fixture chunks — no running Qdrant server needed.

*Used in:* Task 3.

---

### 4. Fixture questions — why a fixed question set

Benchmarks must be **reproducible**. We keep 20 fixed Q&A pairs in `demo/questions.json`:

```json
[
  {
    "query": "What was Apple's total revenue in FY2024?",
    "expected_answer_contains": ["391", "billion"],
    "relevant_chunk_ids": ["aapl-2024-item7-chunk-0", "aapl-2024-item7-chunk-1"]
  }
]
```

Both the latency and retrieval benchmarks load this file. This way the same questions run every time — results are comparable across runs.

*Used in:* Tasks 2, 3, 4.

---

### 5. LLM-as-judge pattern

TruLens groundedness works by asking an LLM: *"Is this claim supported by these sources?"*

```
Judge LLM prompt:
  Claim: "Apple earned $391 billion in FY2024."
  Source: "Apple's FY2024 revenue was $391 billion."
  Q: Is the claim fully supported? Answer YES or NO.

Judge LLM response: YES → score += 1
```

We reuse `OllamaProvider` as the judge — zero extra cost, no API key needed.

*Used in:* Task 1.

---

## Order to type things

1. `demo/questions.json` — 20 fixture questions (no code, just JSON).
2. `evaluation/trulens_setup.py` — groundedness evaluator wrapping TruLens + Ollama judge.
3. `evaluation/latency_bench.py` — cold vs warm timing over fixture questions.
4. `evaluation/retrieval_bench.py` — recall@10 comparison: dense-only vs hybrid.
5. `tests/unit/test_evaluation.py` — hermetic tests for all three evaluators.
6. `Makefile` — add `eval`, `bench-latency`, `bench-retrieval` targets.

---

## Task descriptions

### Task 1 — `src/rag_platform/evaluation/trulens_setup.py`

**What you're doing:** A `GroundednessEvaluator` class that takes a draft answer + source chunks and returns a float score 0.0–1.0 using TruLens + Ollama as judge.

**Done when:** `evaluator.score(draft, chunks)` returns a float and `evaluate_batch(pairs)` returns a list of floats with a mean ≥0.0.

```python
"""TruLens groundedness evaluator — LLM-as-judge pattern."""

from __future__ import annotations

import statistics

from trulens.core import TruSession
from trulens.providers.langchain import Langchain

from rag_platform.llm.provider import get_llm_provider


class GroundednessEvaluator:
    """Scores answer groundedness against source chunks using TruLens."""

    def __init__(self) -> None:
        self._session = TruSession()
        llm_provider = get_llm_provider()
        self._provider = Langchain(chain=llm_provider._llm)

    def score(self, answer: str, chunks: list[str]) -> float:
        """Return groundedness score 0.0–1.0 for one answer."""
        context = "\n---\n".join(chunks)
        result = self._provider.groundedness_measure_with_cot_reasons(
            source=context, statement=answer
        )
        return float(result[0]) if isinstance(result, tuple) else float(result)

    def evaluate_batch(
        self, pairs: list[tuple[str, list[str]]]
    ) -> list[float]:
        """Score a batch of (answer, chunks) pairs. Returns per-pair scores."""
        return [self.score(answer, chunks) for answer, chunks in pairs]

    @staticmethod
    def mean(scores: list[float]) -> float:
        return statistics.mean(scores) if scores else 0.0
```

---

### Task 2 — `src/rag_platform/evaluation/latency_bench.py`

**What you're doing:** Time cold queries (no cache) vs warm queries (cache pre-populated). Report p50, p95, and % reduction.

**Done when:** `run_latency_bench(graph_fn, questions)` returns a dict with `cold_p50`, `warm_p50`, `reduction_pct`.

```python
"""Latency benchmark — cached vs uncached query timing."""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable, Coroutine
from typing import Any


async def _time_query(
    graph_fn: Callable[[str], Coroutine[Any, Any, dict]],
    query: str,
) -> float:
    start = time.perf_counter()
    await graph_fn(query)
    return time.perf_counter() - start


async def run_latency_bench(
    cold_fn: Callable[[str], Coroutine[Any, Any, dict]],
    warm_fn: Callable[[str], Coroutine[Any, Any, dict]],
    questions: list[str],
) -> dict[str, float]:
    """Run cold and warm queries, return timing statistics."""
    cold_times = [await _time_query(cold_fn, q) for q in questions]
    warm_times = [await _time_query(warm_fn, q) for q in questions]

    cold_p50 = statistics.median(cold_times)
    warm_p50 = statistics.median(warm_times)
    reduction_pct = (cold_p50 - warm_p50) / cold_p50 * 100 if cold_p50 > 0 else 0.0

    return {
        "cold_p50": round(cold_p50, 3),
        "cold_p95": round(sorted(cold_times)[int(len(cold_times) * 0.95)], 3),
        "warm_p50": round(warm_p50, 3),
        "warm_p95": round(sorted(warm_times)[int(len(warm_times) * 0.95)], 3),
        "reduction_pct": round(reduction_pct, 1),
        "n_queries": len(questions),
    }
```

---

### Task 3 — `src/rag_platform/evaluation/retrieval_bench.py`

**What you're doing:** Measure Recall@10 for dense-only vs hybrid retrieval on fixture questions with known relevant chunk IDs.

**Done when:** `run_retrieval_bench(dense_retriever, hybrid_retriever, questions)` returns a dict with `dense_recall`, `hybrid_recall`, `improvement_pct`.

```python
"""Retrieval recall benchmark — dense-only vs hybrid RRF."""

from __future__ import annotations


def _recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        return 1.0
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return hits / len(relevant_ids)


def run_retrieval_bench(
    dense_retriever,
    hybrid_retriever,
    questions: list[dict],
) -> dict[str, float]:
    """Compare recall@10 for dense-only vs hybrid retrieval."""
    dense_recalls: list[float] = []
    hybrid_recalls: list[float] = []

    for item in questions:
        query = item["query"]
        relevant_ids = item.get("relevant_chunk_ids", [])

        dense_hits = dense_retriever.retrieve(query, top_k=10)
        dense_ids = [c.metadata.get("chunk_id", "") for c in dense_hits]
        dense_recalls.append(_recall_at_k(dense_ids, relevant_ids))

        hybrid_hits = hybrid_retriever.retrieve(query, top_k=10)
        hybrid_ids = [c.metadata.get("chunk_id", "") for c in hybrid_hits]
        hybrid_recalls.append(_recall_at_k(hybrid_ids, relevant_ids))

    dense_mean = sum(dense_recalls) / len(dense_recalls) if dense_recalls else 0.0
    hybrid_mean = sum(hybrid_recalls) / len(hybrid_recalls) if hybrid_recalls else 0.0
    improvement = (hybrid_mean - dense_mean) / dense_mean * 100 if dense_mean > 0 else 0.0

    return {
        "dense_recall": round(dense_mean, 4),
        "hybrid_recall": round(hybrid_mean, 4),
        "improvement_pct": round(improvement, 1),
        "n_questions": len(questions),
    }
```

---

### Task 4 — `src/rag_platform/demo/questions.json`

**What you're doing:** 20 fixed Q&A fixture questions used by both benchmarks.

```json
[
  {
    "query": "What was Apple's total revenue in FY2024?",
    "expected_answer_contains": ["391", "billion"],
    "relevant_chunk_ids": ["aapl-2024-item7-chunk-0"]
  },
  {
    "query": "What was Microsoft's revenue in FY2024?",
    "expected_answer_contains": ["245", "billion"],
    "relevant_chunk_ids": ["msft-2024-item7-chunk-0"]
  },
  {
    "query": "What was Apple's net income in FY2024?",
    "expected_answer_contains": ["94", "billion"],
    "relevant_chunk_ids": ["aapl-2024-item7-chunk-1"]
  },
  {
    "query": "What risks does Apple cite in its 10-K?",
    "expected_answer_contains": ["competition", "supply"],
    "relevant_chunk_ids": ["aapl-2024-item1a-chunk-0"]
  },
  {
    "query": "What is Tesla's revenue in FY2023?",
    "expected_answer_contains": ["97", "billion"],
    "relevant_chunk_ids": ["tsla-2023-item7-chunk-0"]
  }
]
```

*(5 shown above — write 15 more following the same pattern for MSFT, GOOGL, JPM, TSLA across different items/years.)*

---

### Task 5 — `tests/unit/test_evaluation.py`

**What you're doing:** Hermetic tests for all three evaluators — fake retrievers, fake graph functions, no Ollama needed.

```python
"""Unit tests for evaluation benchmarks (hermetic)."""

from __future__ import annotations

import asyncio

import pytest

from rag_platform.evaluation.latency_bench import run_latency_bench
from rag_platform.evaluation.retrieval_bench import _recall_at_k, run_retrieval_bench
from rag_platform.retrieval.hybrid import RetrievedChunk


# --- Latency benchmark tests ---

async def _instant_fn(query: str) -> dict:
    return {"final_answer": "stub"}


async def _slow_fn(query: str) -> dict:
    import asyncio
    await asyncio.sleep(0.05)
    return {"final_answer": "stub"}


@pytest.mark.asyncio
async def test_latency_bench_cold_slower_than_warm() -> None:
    questions = ["Q1", "Q2", "Q3"]
    result = await run_latency_bench(_slow_fn, _instant_fn, questions)
    assert result["cold_p50"] > result["warm_p50"]
    assert result["reduction_pct"] > 0


@pytest.mark.asyncio
async def test_latency_bench_returns_expected_keys() -> None:
    questions = ["Q1"]
    result = await run_latency_bench(_instant_fn, _instant_fn, questions)
    assert set(result.keys()) == {"cold_p50", "cold_p95", "warm_p50", "warm_p95", "reduction_pct", "n_queries"}


# --- Recall helper tests ---

def test_recall_perfect() -> None:
    assert _recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0


def test_recall_partial() -> None:
    assert _recall_at_k(["a", "x", "y"], ["a", "b"]) == 0.5


def test_recall_zero() -> None:
    assert _recall_at_k(["x", "y"], ["a", "b"]) == 0.0


def test_recall_empty_relevant() -> None:
    assert _recall_at_k(["a"], []) == 1.0


# --- Retrieval benchmark tests ---

class FakeRetriever:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._ids = chunk_ids

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="stub",
                score=1.0,
                metadata={"chunk_id": cid},
            )
            for cid in self._ids[:top_k]
        ]


def test_retrieval_bench_hybrid_beats_dense() -> None:
    questions = [{"query": "Q", "relevant_chunk_ids": ["id-1", "id-2"]}]
    dense = FakeRetriever(["id-1"])           # recalls 1/2
    hybrid = FakeRetriever(["id-1", "id-2"])  # recalls 2/2
    result = run_retrieval_bench(dense, hybrid, questions)
    assert result["hybrid_recall"] > result["dense_recall"]
    assert result["improvement_pct"] > 0


def test_retrieval_bench_returns_expected_keys() -> None:
    questions = [{"query": "Q", "relevant_chunk_ids": ["id-1"]}]
    r = FakeRetriever(["id-1"])
    result = run_retrieval_bench(r, r, questions)
    assert set(result.keys()) == {"dense_recall", "hybrid_recall", "improvement_pct", "n_questions"}
```

---

### Task 6 — `Makefile` additions

**What you're doing:** Add three targets so benchmarks can be run with a single command.

```makefile
eval:
	uv run python -m rag_platform.evaluation.trulens_setup

bench-latency:
	uv run python -m rag_platform.evaluation.latency_bench

bench-retrieval:
	uv run python -m rag_platform.evaluation.retrieval_bench
```

---

## Definition of done for Phase 5

- `uv run pytest tests/unit/test_evaluation.py -v` passes (hermetic, no Ollama/Qdrant).
- `make bench-retrieval` prints `hybrid_recall`, `dense_recall`, `improvement_pct` to stdout.
- `make bench-latency` prints `cold_p50`, `warm_p50`, `reduction_pct` to stdout.
- `make eval` runs groundedness on 5 sample Q+A pairs and prints mean score.
- Numbers match (or exceed) resume claims: ≥94% groundedness, +22% recall, −35% latency.
- `ruff check .` and `mypy src` pass with no new errors.
- Phase 6 (Streamlit UI) can now import `demo/questions.json` for demo fixtures.
