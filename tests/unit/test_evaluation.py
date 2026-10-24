"""Unit tests for evaluation benchmarks (hermetic — no Ollama/Qdrant needed)."""

from __future__ import annotations

import asyncio

import pytest

from rag_platform.evaluation.latency_bench import run_latency_bench
from rag_platform.evaluation.retrieval_bench import _recall_at_k, run_retrieval_bench
from rag_platform.retrieval.hybrid import RetrievedChunk


# ---------------------------------------------------------------------------
# Latency benchmark tests
# ---------------------------------------------------------------------------

async def _instant_fn(query: str) -> dict:
    return {"final_answer": "stub"}


async def _slow_fn(query: str) -> dict:
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
    assert set(result.keys()) == {
        "cold_p50", "cold_p95", "warm_p50", "warm_p95", "reduction_pct", "n_queries"
    }


@pytest.mark.asyncio
async def test_latency_bench_n_queries_matches_input() -> None:
    questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    result = await run_latency_bench(_instant_fn, _instant_fn, questions)
    assert result["n_queries"] == 5


# ---------------------------------------------------------------------------
# Recall helper tests
# ---------------------------------------------------------------------------

def test_recall_perfect() -> None:
    assert _recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0


def test_recall_partial() -> None:
    assert _recall_at_k(["a", "x", "y"], ["a", "b"]) == 0.5


def test_recall_zero() -> None:
    assert _recall_at_k(["x", "y"], ["a", "b"]) == 0.0


def test_recall_empty_relevant() -> None:
    assert _recall_at_k(["a"], []) == 1.0


def test_recall_empty_retrieved() -> None:
    assert _recall_at_k([], ["a", "b"]) == 0.0


# ---------------------------------------------------------------------------
# Retrieval benchmark tests
# ---------------------------------------------------------------------------

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
    dense = FakeRetriever(["id-1"])
    hybrid = FakeRetriever(["id-1", "id-2"])
    result = run_retrieval_bench(dense, hybrid, questions)
    assert result["hybrid_recall"] > result["dense_recall"]
    assert result["improvement_pct"] > 0


def test_retrieval_bench_equal_retrievers() -> None:
    questions = [{"query": "Q", "relevant_chunk_ids": ["id-1"]}]
    r = FakeRetriever(["id-1"])
    result = run_retrieval_bench(r, r, questions)
    assert result["dense_recall"] == result["hybrid_recall"]
    assert result["improvement_pct"] == 0.0


def test_retrieval_bench_returns_expected_keys() -> None:
    questions = [{"query": "Q", "relevant_chunk_ids": ["id-1"]}]
    r = FakeRetriever(["id-1"])
    result = run_retrieval_bench(r, r, questions)
    assert set(result.keys()) == {
        "dense_recall", "hybrid_recall", "improvement_pct", "n_questions"
    }


def test_retrieval_bench_n_questions_matches_input() -> None:
    questions = [
        {"query": "Q1", "relevant_chunk_ids": ["id-1"]},
        {"query": "Q2", "relevant_chunk_ids": ["id-2"]},
    ]
    r = FakeRetriever(["id-1", "id-2"])
    result = run_retrieval_bench(r, r, questions)
    assert result["n_questions"] == 2
