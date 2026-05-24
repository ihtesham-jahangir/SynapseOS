"""
Async L2 Compression Queue — v2.1

Replaces the ad-hoc asyncio.create_task() pattern with a proper queue-backed
background worker so L1→L2 summarization is truly async and rate-limited.

Design:
  ┌─────────────────────────────────────────────────────────────┐
  │  MemoryManager.record_turn()                                │
  │    │ L1 overflows → submit job (non-blocking)               │
  │    ▼                                                        │
  │  CompressionQueue  ─── asyncio.Queue (maxsize=32) ──►      │
  │                                   single background worker  │
  │                                      │                     │
  │                                      ▼                     │
  │                        ContextCompressor.compress_turns()   │
  │                        L2SummaryCache.store()               │
  └─────────────────────────────────────────────────────────────┘

Guarantees:
  - At most one active LLM summarization call at a time per session
  - Per-session deduplication: submitting for session X while X is already
    queued or in-flight is a no-op
  - Backpressure: QueueFull drops jobs gracefully (L2 miss on next retrieval)
  - Ordered shutdown: stop() drains the queue before returning
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Set

from orchestrator.core.types import Message
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.metrics import COMPRESSION_QUEUE_DEPTH

log = get_logger(__name__)

_SENTINEL = None  # signals worker to exit


class CompressionQueue:
    """
    Single-worker async queue for L1→L2 conversation summarization.

    Call ``start()`` at application startup and ``stop()`` at shutdown.
    MemoryManager submits jobs via ``submit()``.
    """

    def __init__(
        self,
        compressor,
        l2,
        maxsize: int = 32,
    ) -> None:
        self._compressor = compressor
        self._l2 = l2
        self._maxsize = maxsize

        self._queue: Optional[asyncio.Queue] = None
        self._queued: Set[str] = set()      # sessions currently queued
        self._in_flight: Optional[str] = None  # session currently being compressed
        self._worker_task: Optional[asyncio.Task] = None
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._started:
            return
        # Create queue here so it binds to the running event loop, not __init__'s loop
        self._queue = asyncio.Queue(maxsize=self._maxsize)
        self._queued.clear()
        self._worker_task = asyncio.create_task(self._worker(), name="compression-worker")
        self._started = True
        log.info("Compression worker started", maxsize=self._maxsize)

    async def stop(self) -> None:
        if not self._started or self._queue is None:
            return
        await self._queue.put(_SENTINEL)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
        self._started = False
        log.info("Compression worker stopped")

    # ── Public interface ───────────────────────────────────────────────────────

    async def submit(
        self,
        session_id: str,
        turns: List[Message],
        turn_start: int = 0,
        turn_end: Optional[int] = None,
    ) -> None:
        """
        Non-blocking submission. Drops the job if:
          - the session is already queued or in-flight (deduplication)
          - the queue is full (backpressure)
        """
        if session_id in self._queued or session_id == self._in_flight:
            log.debug(
                "Compression already pending — skipping duplicate",
                session=session_id,
            )
            return

        if self._queue is None:
            log.debug("Compression queue not started — dropping job", session=session_id)
            return

        job = (session_id, list(turns), turn_start, turn_end or len(turns))
        try:
            self._queue.put_nowait(job)
            self._queued.add(session_id)
            COMPRESSION_QUEUE_DEPTH.set(self._queue.qsize())
            log.debug(
                "Compression job queued",
                session=session_id,
                turns=len(turns),
                queue_depth=self._queue.qsize(),
            )
        except asyncio.QueueFull:
            log.warning(
                "Compression queue full — dropping job",
                session=session_id,
                maxsize=self._maxsize,
            )

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    # ── Background worker ──────────────────────────────────────────────────────

    async def _worker(self) -> None:
        """
        Processes compression jobs one at a time from the queue.
        Runs until a sentinel (None) is received.
        """
        log.debug("Compression worker loop running")
        while True:
            job = await self._queue.get()
            COMPRESSION_QUEUE_DEPTH.set(self._queue.qsize())

            if job is _SENTINEL:
                self._queue.task_done()
                break

            session_id, turns, turn_start, turn_end = job
            self._queued.discard(session_id)
            self._in_flight = session_id

            try:
                summary_item = await self._compressor.compress_turns(
                    turns=turns,
                    session_id=session_id,
                    turn_start=turn_start,
                    turn_end=turn_end,
                )
                await self._l2.store(summary_item)
                log.info(
                    "L1→L2 compression complete",
                    session=session_id,
                    turns_compressed=turn_end - turn_start,
                    summary_tokens=summary_item.token_count,
                )
            except Exception as exc:
                log.error(
                    "Compression job failed",
                    session=session_id,
                    error=str(exc),
                )
            finally:
                self._in_flight = None
                self._queue.task_done()
