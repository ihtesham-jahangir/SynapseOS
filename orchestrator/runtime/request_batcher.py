"""
Inference Request Batcher — v2.1

Controls concurrent access to llama.cpp inference slots so Python-level
concurrency matches the server's --parallel N capacity.

Design:
  ┌──────────────────────────────────────────────────────────────┐
  │  InferenceEngine.generate() / stream()                       │
  │     │                                                        │
  │     ▼  async with batcher.acquire():                        │
  │  InferenceBatcher ── Semaphore(n_parallel) ──► llama.cpp    │
  │                                                              │
  │  Optional time-window batching (batch_timeout_ms > 0):      │
  │    Requests arriving within the window sleep briefly before  │
  │    competing for slots, naturally grouping under bursty load │
  │    and maximising slot utilisation.                          │
  └──────────────────────────────────────────────────────────────┘

Settings (via .env):
  LLAMA_N_PARALLEL=2        # match --parallel 2 in start_llama.sh
  LLAMA_BATCH_TIMEOUT_MS=50 # optional: enable time-window grouping
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.metrics import (
    INFERENCE_QUEUE_DEPTH,
    INFERENCE_IN_FLIGHT,
    INFERENCE_WAIT_MS,
)

log = get_logger(__name__)


class InferenceBatcher:
    """
    Semaphore-based concurrency gate for llama.cpp inference calls.

    When n_parallel=1 (default), serialises all generation calls — safe for
    a single-slot llama.cpp server.  Set to N to allow N simultaneous calls
    (requires --parallel N in start_llama.sh).

    With batch_timeout_ms > 0, requests sleep briefly before acquiring the
    semaphore so that bursts arriving within the window compete together,
    maximising slot saturation rather than stacking sequentially.
    """

    def __init__(
        self,
        n_parallel: int = 1,
        batch_timeout_ms: float = 0.0,
    ) -> None:
        self._n_parallel = max(1, n_parallel)
        self._timeout_s = max(0.0, batch_timeout_ms) / 1000.0
        self._sem = asyncio.Semaphore(self._n_parallel)
        self._queue_depth = 0
        self._in_flight = 0

        log.info(
            "InferenceBatcher ready",
            n_parallel=self._n_parallel,
            batch_timeout_ms=batch_timeout_ms,
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[None, None]:
        """
        Async context manager: acquire an inference slot.
        Tracks queue depth and in-flight count in Prometheus gauges.
        """
        self._queue_depth += 1
        INFERENCE_QUEUE_DEPTH.set(self._queue_depth)

        t_start = time.perf_counter()

        # Optional: sleep for batch_timeout_ms so concurrent requests group
        if self._timeout_s > 0:
            await asyncio.sleep(self._timeout_s)

        await self._sem.acquire()
        wait_ms = (time.perf_counter() - t_start) * 1000

        self._queue_depth -= 1
        self._in_flight += 1
        INFERENCE_QUEUE_DEPTH.set(self._queue_depth)
        INFERENCE_IN_FLIGHT.set(self._in_flight)
        INFERENCE_WAIT_MS.observe(wait_ms)

        log.debug(
            "Inference slot acquired",
            wait_ms=f"{wait_ms:.1f}",
            in_flight=self._in_flight,
            queued=self._queue_depth,
        )

        try:
            yield
        finally:
            self._in_flight -= 1
            self._sem.release()
            INFERENCE_IN_FLIGHT.set(self._in_flight)

    @property
    def n_parallel(self) -> int:
        return self._n_parallel

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queue_depth(self) -> int:
        return self._queue_depth
