"""Prometheus custom metrics for the RAG platform."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Number of Redis cache hits",
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Number of Redis cache misses",
)

LLM_REQUESTS = Counter(
    "rag_llm_requests_total",
    "Number of LLM inference calls",
    ["provider", "model"],
)

LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds",
    "End-to-end LLM call latency in seconds",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
