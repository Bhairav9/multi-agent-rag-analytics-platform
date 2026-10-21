"""Latency benchmark — cached vs uncached query timing."""

from __future__ import annotations

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
