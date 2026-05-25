"""
Unified Memory Manager – orchestrates all four cache tiers.

The manager presents a single interface to the inference engine.
It decides which tiers to query, merges results, and triggers
compression when L1 overflows.
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from orchestrator.core.types import (
    MemoryItem,
    MemoryLevel,
    MemoryResult,
    Message,
    MessageRole,
)
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.async_utils import gather_with_fallback
from orchestrator.utils.metrics import MEMORY_HITS, MEMORY_RETRIEVAL_LATENCY_MS

from .l1_cache import L1ConversationCache
from .l2_cache import L2SummaryCache
from .l3_cache import L3VectorMemory
from .l4_cache import L4KnowledgeBase
from .compressor import ContextCompressor
from .compression_queue import CompressionQueue
from orchestrator.utils.fact_extractor import extract_facts, should_extract

log = get_logger(__name__)


class MemoryManager:
    """
    Hierarchical memory access with automatic tier management.

    Read path:
        L1 (always) → L2 (if session has history) → L3 (semantic) → L4 (knowledge)
        Results are returned as a unified MemoryResult for the fusion engine.

    Write path:
        New turns → L1
        When L1 overflows → compress to L2
        Important facts → L3 (via extract_and_store)

    Priority: L1 > L4 > L2 > L3
    (Knowledge base and hot context get highest fusion priority)
    """

    def __init__(
        self,
        l1: L1ConversationCache,
        l2: L2SummaryCache,
        l3: L3VectorMemory,
        l4: L4KnowledgeBase,
        compressor: Optional[ContextCompressor] = None,
        compression_queue: Optional[CompressionQueue] = None,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._compressor = compressor
        self._compression_queue = compression_queue
        self._cfg = get_settings().memory
        self._compression_locks: dict = {}  # fallback when no queue is wired

    async def record_turn(
        self,
        session_id: str,
        user_message: Message,
        assistant_message: Message,
    ) -> None:
        """
        Store a completed conversation turn.
        Triggers L1→L2 compression if L1 token budget is near exhausted.
        """
        await self._l1.push_turn(session_id, user_message)
        await self._l1.push_turn(session_id, assistant_message)

        # Auto-extract personal facts from user message → L3 semantic memory
        if should_extract(user_message.content):
            facts = extract_facts(user_message.content)
            for fact_text, importance in facts:
                asyncio.create_task(self.store_fact(session_id, fact_text, importance))

        # Check if compression is needed
        current_tokens = self._l1.session_token_count(session_id)
        l1_capacity = self._l1._max_tokens
        if current_tokens > l1_capacity * 0.85:
            if self._compression_queue is not None:
                # v2.1: queue-based async compression (rate-limited, deduplicated)
                turns = await self._l1.get_turns(session_id)
                split = len(turns) // 2
                await self._compression_queue.submit(
                    session_id=session_id,
                    turns=turns[:split],
                    turn_start=0,
                    turn_end=split,
                )
            else:
                # fallback: ad-hoc task (no rate limiting)
                asyncio.create_task(self._compress_session(session_id))

    async def _compress_session(self, session_id: str) -> None:
        """Background compression of L1 overflow into L2 summary."""
        if self._compressor is None:
            return

        # Prevent concurrent compressions for same session
        lock = self._compression_locks.setdefault(session_id, asyncio.Lock())
        if lock.locked():
            return

        async with lock:
            turns = await self._l1.get_turns(session_id)
            if len(turns) < self._cfg.l2_summary_every_n_turns:
                return

            # Compress the older half of the conversation
            split = len(turns) // 2
            to_compress = turns[:split]

            try:
                summary_item = await self._compressor.compress_turns(
                    turns=to_compress,
                    session_id=session_id,
                    turn_start=0,
                    turn_end=split,
                )
                await self._l2.store(summary_item)
                await self._l1.pop_turns(session_id, split)
                log.info(
                    "L1→L2 compression completed",
                    session=session_id,
                    turns_compressed=split,
                    summary_tokens=summary_item.token_count,
                )
            except Exception as exc:
                log.error("Compression failed", session=session_id, error=str(exc))

    async def store_fact(
        self,
        session_id: str,
        fact: str,
        importance: float = 0.7,
    ) -> None:
        """Explicitly store an important fact into L3 vector memory."""
        from orchestrator.core.types import MemoryItem
        import uuid
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=fact,
            level=MemoryLevel.L3,
            session_id=session_id,
            importance_score=importance,
        )
        await self._l3.store(item)

    async def retrieve_all(
        self,
        query: str,
        session_id: str,
    ) -> MemoryResult:
        """
        Query all active tiers in parallel and return merged MemoryResult.
        Each tier's results are labeled with their MemoryLevel for
        the fusion engine to weight appropriately.
        """
        t0 = time.perf_counter()

        # Parallel retrieval from all tiers (with per-tier latency tracking)
        async def _timed(coro, level: str):
            t = time.perf_counter()
            result = await coro
            MEMORY_RETRIEVAL_LATENCY_MS.labels(level=level).observe(
                (time.perf_counter() - t) * 1000
            )
            return result

        l1_items, l2_items, l3_items, l4_items = await gather_with_fallback(
            _timed(self._l1.retrieve(query, session_id, limit=self._cfg.l1_max_turns), "l1"),
            _timed(self._l2.retrieve(query, session_id, limit=3), "l2"),
            _timed(self._l3.retrieve(query, session_id, limit=self._cfg.l3_max_results), "l3"),
            _timed(self._l4.retrieve(query, session_id, limit=self._cfg.l4_max_results), "l4"),
            fallbacks=[[], [], [], []],
        )

        # Track cache hits for metrics
        if l1_items:
            MEMORY_HITS.labels(level="l1").inc(len(l1_items))
        if l2_items:
            MEMORY_HITS.labels(level="l2").inc(len(l2_items))
        if l3_items:
            MEMORY_HITS.labels(level="l3").inc(len(l3_items))
        if l4_items:
            MEMORY_HITS.labels(level="l4").inc(len(l4_items))

        all_items: List[MemoryItem] = (
            list(l1_items or [])
            + list(l4_items or [])  # L4 before L2/L3 for priority
            + list(l2_items or [])
            + list(l3_items or [])
        )

        total_tokens = sum(item.token_count for item in all_items)
        levels_queried = [MemoryLevel.L1, MemoryLevel.L2, MemoryLevel.L3, MemoryLevel.L4]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        log.debug(
            "Memory retrieval complete",
            session=session_id,
            l1=len(l1_items or []),
            l2=len(l2_items or []),
            l3=len(l3_items or []),
            l4=len(l4_items or []),
            total_tokens=total_tokens,
            elapsed_ms=f"{elapsed_ms:.1f}",
        )

        return MemoryResult(
            items=all_items,
            total_tokens=total_tokens,
            levels_queried=levels_queried,
            query_time_ms=elapsed_ms,
        )

    async def get_conversation(self, session_id: str) -> List[Message]:
        """Return raw L1 conversation turns for context building."""
        return await self._l1.get_turns(session_id)

    def list_sessions(self) -> list:
        """Return summary info for every session currently in L1."""
        sessions = []
        for sid in self._l1.active_sessions():
            sessions.append({
                "session_id": sid,
                "turns": len(self._l1._sessions.get(sid, [])),
                "tokens": self._l1.session_token_count(sid),
            })
        return sessions

    async def clear_session(self, session_id: str) -> None:
        await asyncio.gather(
            self._l1.clear_session(session_id),
            self._l2.clear_session(session_id),
            self._l3.clear_session(session_id),
            self._l4.clear_session(session_id),
        )
        log.info("Session memory cleared", session=session_id)
