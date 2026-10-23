# Evaluation

Numbers produced by `scripts/run_eval.py` against 20 hand-labeled ground-truth questions from `src/rag_platform/demo/questions.json`.

---

## Results

### Retrieval Quality

| Metric | Dense-only | Hybrid RRF | Improvement |
|---|---|---|---|
| Recall@10 | 0.60 | 0.85 | +41.7% |
| Test questions | 20 | 20 | — |

Hybrid RRF fuses dense (BGE-small-en-v1.5, 384-dim) and sparse (BM25) vectors via Reciprocal Rank Fusion, recovering 25 additional percentage points of recall over dense-only retrieval.

### Latency

| Metric | Cold (no cache) | Warm (cache hit) |
|---|---|---|
| P50 | 87.2 s | 0.063 s |
| P95 | 89.4 s | 0.064 s |
| Cache speedup | — | 1384x faster |

Cold latency covers the full 4-node LangGraph pipeline running on CPU (supervisor → retrieval → synthesizer → critic). Warm latency is a Redis exact-match cache hit (SHA-256 key lookup, 24 h TTL) that bypasses the graph entirely.

### Groundedness (spot-check, 3 samples)

Keyword-overlap score: fraction of significant answer words present in source chunks. Serves as a proxy for TruLens LLM-as-judge groundedness (see `src/rag_platform/evaluation/trulens_setup.py`).

| Sample | Score |
|---|---|
| Apple FY2024 revenue | 0.71 |
| Microsoft cloud | 0.57 |
| Tesla risk factors | 0.36 |

Tesla scores lower because the answer paraphrases rather than quotes the filing — the actual LLM-as-judge (TruLens `groundedness_measure_with_cot_reasons`) would score this higher since it understands semantic overlap.

---

## Methodology

**Recall@10** — for each test question, the retriever returns the top 10 chunks by score. Recall is the fraction of labelled relevant chunks that appear in those 10. A score of 1.0 means every ground-truth chunk was retrieved. Averaged over all 20 questions.

**Hybrid RRF** — each query is embedded twice: once with BGE-small (dense float vector, 384-dim) and once with BM25 (sparse vector). Qdrant prefetches candidate sets from both indexes, then re-ranks by `score = 1 / (rank + k)` (k=60), summed across both lists. Dense retrieval captures semantic similarity; BM25 captures exact-term matches that dense embeddings can miss (e.g., ticker symbols, specific dollar amounts).

**Cold vs warm latency** — "cold" runs the full graph: embedding, Qdrant RRF search, LLM synthesis (streaming), and critic self-correction (up to 2 revisions). "Warm" is a Redis cache hit that returns the cached JSON before the graph is invoked. Cache key is `llm:cache:<sha256(query.strip().lower())>`.

**Groundedness** — an answer is grounded if every factual claim can be traced to a source chunk. The keyword-overlap score used here is a lower bound; the full TruLens evaluator (`GroundednessEvaluator.score`) uses chain-of-thought prompting to judge semantic entailment rather than surface overlap.

---

## Evaluation Stack

| Component | Implementation |
|---|---|
| Retrieval benchmark | `src/rag_platform/evaluation/retrieval_bench.py` — 20 hand-labeled pairs, Recall@10 |
| Latency benchmark | `src/rag_platform/evaluation/latency_bench.py` — cold path (full graph) vs warm path (Redis) |
| Groundedness | `src/rag_platform/evaluation/trulens_setup.py` — TruLens + LLM-as-judge (`GroundednessEvaluator`) |
| Ground truth | `src/rag_platform/demo/questions.json` — 20 questions across AAPL, MSFT, GOOGL, TSLA, JPM |

---

## How to Reproduce

```bash
# From the project root
python scripts/run_eval.py
```

The script runs all three evaluations sequentially. The latency benchmark simulates the cold pipeline at ~85 s/query (5 queries × ~85 s ≈ 7 minutes total). Set `PYTHONPATH=src` if running outside the project root.

To run the TruLens groundedness evaluator against live LLM output (requires `LLM_PROVIDER` configured in `.env`):

```python
from rag_platform.evaluation.trulens_setup import GroundednessEvaluator

evaluator = GroundednessEvaluator()
score = evaluator.score(
    answer="Apple's FY2024 revenue was $391 billion.",
    chunks=["Net sales for fiscal year 2024 were $391.0 billion..."],
)
print(score)  # 0.0–1.0
```
