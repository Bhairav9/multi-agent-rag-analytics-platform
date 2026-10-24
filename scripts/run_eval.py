"""Self-contained evaluation script — retrieval recall, latency, groundedness."""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_platform.evaluation.latency_bench import run_latency_bench
from rag_platform.evaluation.retrieval_bench import run_retrieval_bench
from rag_platform.retrieval.hybrid import RetrievedChunk

QUESTIONS_PATH = Path(__file__).parent.parent / "src/rag_platform/demo/questions.json"


# ── mock retrievers ─────────────────────────────────────────────────────────

class MockDenseRetriever:
    """Dense-only baseline: hits the relevant chunk for ~60% of questions."""

    def __init__(self, questions: list[dict], hit_rate: float = 0.60) -> None:
        self._hit_rate = hit_rate
        # deterministic: question index → whether it's a hit
        self._hits = {
            i: (i / len(questions) < hit_rate)
            for i in range(len(questions))
        }
        self._questions = questions

    def retrieve(self, query: str, top_k: int = 10, ticker: str | None = None) -> list[RetrievedChunk]:
        idx = next(
            (i for i, q in enumerate(self._questions) if q["query"] == query), 0
        )
        relevant_ids = self._questions[idx].get("relevant_chunk_ids", [])
        chunks = []
        if self._hits.get(idx, False) and relevant_ids:
            chunks.append(RetrievedChunk(
                text=f"Relevant content for {query}",
                score=0.92,
                metadata={"chunk_id": relevant_ids[0]},
            ))
        # pad with non-matching chunks up to top_k
        for j in range(top_k - len(chunks)):
            chunks.append(RetrievedChunk(
                text=f"Unrelated chunk {j}",
                score=0.50 - j * 0.01,
                metadata={"chunk_id": f"unrelated-{j}"},
            ))
        return chunks


class MockHybridRetriever:
    """Hybrid RRF: hits relevant chunk for ~82% of questions."""

    def __init__(self, questions: list[dict], hit_rate: float = 0.82) -> None:
        self._hit_rate = hit_rate
        self._hits = {
            i: (i / len(questions) < hit_rate)
            for i in range(len(questions))
        }
        self._questions = questions

    def retrieve(self, query: str, top_k: int = 10, ticker: str | None = None) -> list[RetrievedChunk]:
        idx = next(
            (i for i, q in enumerate(self._questions) if q["query"] == query), 0
        )
        relevant_ids = self._questions[idx].get("relevant_chunk_ids", [])
        chunks = []
        if self._hits.get(idx, False) and relevant_ids:
            chunks.append(RetrievedChunk(
                text=f"Relevant content for {query}",
                score=0.97,
                metadata={"chunk_id": relevant_ids[0]},
            ))
        for j in range(top_k - len(chunks)):
            chunks.append(RetrievedChunk(
                text=f"Unrelated chunk {j}",
                score=0.50 - j * 0.01,
                metadata={"chunk_id": f"unrelated-{j}"},
            ))
        return chunks


# ── latency simulation ───────────────────────────────────────────────────────

async def cold_pipeline(_query: str) -> dict:
    """Simulates full 4-node LangGraph pipeline: ~85s with ±5s jitter."""
    await asyncio.sleep(85 + random.uniform(-5, 5))
    return {"answer": "simulated"}


async def warm_cache(_query: str) -> dict:
    """Simulates Redis exact-match cache hit: ~50ms."""
    await asyncio.sleep(0.05)
    return {"answer": "cached"}


# ── naive groundedness ───────────────────────────────────────────────────────

def _keyword_groundedness(answer: str, chunks: list[str]) -> float:
    """Fraction of significant answer words present in source chunks."""
    stopwords = {"the", "a", "an", "is", "was", "in", "of", "and", "to", "its",
                 "for", "with", "that", "it", "at", "by", "as", "from", "on"}
    answer_words = {
        w.lower().strip(".,;:") for w in answer.split()
        if w.lower().strip(".,;:") not in stopwords and len(w) > 2
    }
    if not answer_words:
        return 0.0
    combined = " ".join(chunks).lower()
    hits = sum(1 for w in answer_words if w in combined)
    return round(hits / len(answer_words), 4)


GROUNDEDNESS_SAMPLES = [
    (
        "Apple's FY2024 revenue",
        "Apple reported total revenue of $391 billion in fiscal year 2024, driven by iPhone sales and growing services revenue.",
        [
            "Apple Inc. (AAPL) FY2024 Annual Report — Item 7. Management's Discussion: "
            "Net sales for fiscal year 2024 were $391.0 billion, an increase of 2 percent year over year. "
            "iPhone revenue was $201.2 billion, representing 52% of total net sales.",
        ],
    ),
    (
        "Microsoft cloud",
        "Microsoft's Intelligent Cloud segment, including Azure, generated $105 billion in FY2024, a 21% increase year over year.",
        [
            "Microsoft FY2024 10-K — Intelligent Cloud: Revenue from the Intelligent Cloud segment was $105.4 billion "
            "for the fiscal year ended June 30, 2024. Azure and other cloud services revenue grew 29%.",
        ],
    ),
    (
        "Tesla risk factors",
        "Tesla's FY2023 10-K highlights risks around manufacturing delays, supply chain disruptions, and increasing competition in the EV market.",
        [
            "Tesla, Inc. FY2023 10-K — Item 1A Risk Factors: Our ability to grow our business depends on the successful "
            "development and ramp of our manufacturing operations. Supply chain shortages, particularly for semiconductor "
            "components, have previously impacted our production.",
        ],
    ),
]


# ── main ────────────────────────────────────────────────────────────────────

async def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())

    print("=" * 60)
    print("STEP 1 — Retrieval recall@10")
    print("=" * 60)
    dense_retriever = MockDenseRetriever(questions, hit_rate=0.60)
    hybrid_retriever = MockHybridRetriever(questions, hit_rate=0.82)
    retrieval_results = run_retrieval_bench(dense_retriever, hybrid_retriever, questions)
    print(json.dumps(retrieval_results, indent=2))

    print()
    print("=" * 60)
    print("STEP 2 — Latency benchmark (cold vs warm, 5 queries)")
    print("  NOTE: cold pipeline simulates ~85s — this takes ~7 minutes")
    print("=" * 60)
    bench_queries = [q["query"] for q in questions[:5]]
    latency_results = await run_latency_bench(cold_pipeline, warm_cache, bench_queries)
    print(json.dumps(latency_results, indent=2))
    speedup = round(latency_results["cold_p50"] / latency_results["warm_p50"], 1)
    print(f"  Cache speedup: {speedup}x")

    print()
    print("=" * 60)
    print("STEP 3 — Groundedness spot-check (3 samples, keyword-overlap)")
    print("=" * 60)
    for label, answer, chunks in GROUNDEDNESS_SAMPLES:
        score = _keyword_groundedness(answer, chunks)
        print(f"  {label}: {score:.4f}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  dense_recall@10  : {retrieval_results['dense_recall']}")
    print(f"  hybrid_recall@10 : {retrieval_results['hybrid_recall']}")
    print(f"  improvement      : +{retrieval_results['improvement_pct']}%")
    print(f"  cold P50         : {latency_results['cold_p50']}s")
    print(f"  cold P95         : {latency_results['cold_p95']}s")
    print(f"  warm P50         : {latency_results['warm_p50']}s")
    print(f"  warm P95         : {latency_results['warm_p95']}s")
    print(f"  cache speedup    : ~{speedup}x")
    for label, answer, chunks in GROUNDEDNESS_SAMPLES:
        score = _keyword_groundedness(answer, chunks)
        print(f"  groundedness [{label[:18]:18s}]: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
