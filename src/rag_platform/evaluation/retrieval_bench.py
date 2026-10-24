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

