"""
Prometheus metrics + in-process latency tracking for benchmarking.
Exposes /metrics endpoint for Prometheus scraping.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, Generator, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ─── Metric definitions ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "synapseos_requests_total",
    "Total inference requests",
    ["intent", "status"],
)

REQUEST_LATENCY = Histogram(
    "synapseos_request_latency_seconds",
    "End-to-end request latency",
    ["intent"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

TTFT_HISTOGRAM = Histogram(
    "synapseos_time_to_first_token_seconds",
    "Time to first token",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

TOKEN_THROUGHPUT = Summary(
    "synapseos_tokens_per_second",
    "Generation throughput tokens/second",
)

MEMORY_HITS = Counter(
    "synapseos_memory_cache_hits_total",
    "Cache hits per memory level",
    ["level"],
)

RAG_RETRIEVED = Counter(
    "synapseos_rag_chunks_retrieved_total",
    "RAG chunks retrieved",
)

FUSION_ITEMS = Histogram(
    "synapseos_fusion_context_items",
    "Number of items entering fusion",
    buckets=[1, 2, 4, 8, 16, 32],
)

ACTIVE_SESSIONS = Gauge(
    "synapseos_active_sessions",
    "Currently active sessions",
)

LLAMA_ERRORS = Counter(
    "synapseos_llama_errors_total",
    "llama.cpp backend errors",
    ["error_type"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

@contextmanager
def track_latency(histogram: Histogram, labels: Optional[Dict[str, str]] = None) -> Generator:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if labels:
            histogram.labels(**labels).observe(elapsed)
        else:
            histogram.observe(elapsed)


@asynccontextmanager
async def track_latency_async(
    histogram: Histogram,
    labels: Optional[Dict[str, str]] = None,
) -> AsyncGenerator:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if labels:
            histogram.labels(**labels).observe(elapsed)
        else:
            histogram.observe(elapsed)


class LatencyTracker:
    """Simple wall-clock tracker for multi-phase latency measurement."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._checkpoints: Dict[str, float] = {}

    def checkpoint(self, name: str) -> float:
        t = time.perf_counter()
        self._checkpoints[name] = (t - self._start) * 1000
        return self._checkpoints[name]

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def report(self) -> Dict[str, float]:
        return {**self._checkpoints, "total_ms": self.elapsed_ms()}


def get_metrics_output() -> bytes:
    return generate_latest()


METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST
