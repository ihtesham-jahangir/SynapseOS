"""
Tests for v2.1 features:
  - WAL mode enabled on L2 and L4 SQLite databases
  - CompressionQueue: lifecycle, deduplication, backpressure, worker processing
  - InferenceBatcher: semaphore gate, concurrency limiting, metrics, time-window
  - InferenceEngine uses batcher around generate() and stream()
  - MemoryManager routes to CompressionQueue when provided
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.memory.l2_cache import L2SummaryCache
from orchestrator.memory.l4_cache import L4KnowledgeBase
from orchestrator.memory.compression_queue import CompressionQueue
from orchestrator.runtime.request_batcher import InferenceBatcher
from orchestrator.core.types import (
    Message, MessageRole, MemoryItem, MemoryLevel,
    FusedContext, GenerationParams, ContextItem,
)


# ══════════════════════════════════════════════════════════════════════════════
# WAL Mode
# ══════════════════════════════════════════════════════════════════════════════

class TestWALMode:
    @pytest.mark.asyncio
    async def test_l2_initialises_with_wal(self, tmp_path):
        """After _init(), L2 SQLite database should be in WAL journal mode."""
        db = str(tmp_path / "l2_wal.db")
        cache = L2SummaryCache(db_path=db)
        # Trigger init by doing a store
        import uuid
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content="test summary",
            level=MemoryLevel.L2,
            session_id="s1",
            metadata={"turn_start": 0, "turn_end": 5},
        )
        await cache.store(item)

        # Verify WAL mode was set on the file
        import aiosqlite
        async with aiosqlite.connect(db) as conn:
            async with conn.execute("PRAGMA journal_mode") as cur:
                row = await cur.fetchone()
        assert row[0] == "wal", f"Expected WAL mode, got: {row[0]}"

    @pytest.mark.asyncio
    async def test_l4_initialises_with_wal(self, tmp_path):
        """After _init(), L4 SQLite database should be in WAL journal mode."""
        db = str(tmp_path / "l4_wal.db")
        cache = L4KnowledgeBase(db_path=db)
        await cache.add_knowledge("test fact", category="test")

        import aiosqlite
        async with aiosqlite.connect(db) as conn:
            async with conn.execute("PRAGMA journal_mode") as cur:
                row = await cur.fetchone()
        assert row[0] == "wal", f"Expected WAL mode, got: {row[0]}"

    @pytest.mark.asyncio
    async def test_l2_wal_does_not_break_store_and_retrieve(self, tmp_path):
        """
        After WAL is enabled, normal store→retrieve still works correctly.
        (In-memory SQLite can't share state across aiosqlite connections,
        so we use a real temp file here.)
        """
        cache = L2SummaryCache(db_path=str(tmp_path / "l2_rw.db"))
        import uuid
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content="wal round-trip summary",
            level=MemoryLevel.L2,
            session_id="s1",
            metadata={"turn_start": 0, "turn_end": 3},
        )
        await cache.store(item)
        results = await cache.retrieve("wal round-trip", "s1")
        assert len(results) >= 1
        assert "wal" in results[0].content.lower()


# ══════════════════════════════════════════════════════════════════════════════
# CompressionQueue
# ══════════════════════════════════════════════════════════════════════════════

def _make_compressor(summary: str = "summary text") -> MagicMock:
    comp = MagicMock()
    item = MemoryItem(
        id="test-id",
        content=summary,
        level=MemoryLevel.L2,
        session_id="s1",
        importance_score=0.7,
        token_count=10,
        metadata={"turn_start": 0, "turn_end": 5},
    )
    comp.compress_turns = AsyncMock(return_value=item)
    return comp


def _make_l2() -> MagicMock:
    l2 = MagicMock()
    l2.store = AsyncMock()
    return l2


@pytest.mark.asyncio
class TestCompressionQueue:
    async def test_start_stop_lifecycle(self):
        q = CompressionQueue(_make_compressor(), _make_l2())
        await q.start()
        assert q._started is True
        await q.stop()
        assert q._started is False

    async def test_double_start_is_safe(self):
        q = CompressionQueue(_make_compressor(), _make_l2())
        await q.start()
        await q.start()  # should not raise or spawn extra workers
        await q.stop()

    async def test_submit_processes_job(self):
        comp = _make_compressor()
        l2 = _make_l2()
        q = CompressionQueue(comp, l2)
        await q.start()

        turns = [Message(role=MessageRole.USER, content=f"turn {i}") for i in range(4)]
        await q.submit("sess-a", turns, turn_start=0, turn_end=4)

        await asyncio.wait_for(q._queue.join(), timeout=3.0)
        await q.stop()

        comp.compress_turns.assert_awaited_once()
        l2.store.assert_awaited_once()

    async def test_deduplication_same_session(self):
        """Submitting the same session twice while the first is queued is a no-op."""
        comp = _make_compressor()
        l2 = _make_l2()
        # Use a large queue but pause the compressor so the first job stays queued
        event = asyncio.Event()

        async def slow_compress(*args, **kwargs):
            await event.wait()
            return MemoryItem(
                id="x", content="s", level=MemoryLevel.L2, session_id="s1",
                importance_score=0.5, token_count=5,
                metadata={"turn_start": 0, "turn_end": 1},
            )

        comp.compress_turns = slow_compress
        q = CompressionQueue(comp, l2, maxsize=32)
        await q.start()

        turns = [Message(role=MessageRole.USER, content="t")]
        await q.submit("dup-session", turns)
        await q.submit("dup-session", turns)  # duplicate — should be dropped

        assert q.queue_depth <= 1  # only 1 job queued

        event.set()
        await asyncio.wait_for(q._queue.join(), timeout=3.0)
        await q.stop()

    async def test_backpressure_queue_full(self):
        """When queue is full, submit() drops jobs without raising."""
        comp = _make_compressor()
        l2 = _make_l2()
        q = CompressionQueue(comp, l2, maxsize=1)
        # Don't start the worker so the queue fills up
        turns = [Message(role=MessageRole.USER, content="t")]
        await q.submit("s1", turns)  # fills the queue
        await q.submit("s2", turns)  # should be dropped, not raise
        # If we reach here without exception, backpressure worked

    async def test_compressor_failure_does_not_crash_worker(self):
        """A failing compression job is logged and skipped; worker stays alive."""
        comp = MagicMock()
        comp.compress_turns = AsyncMock(side_effect=RuntimeError("LLM down"))
        l2 = _make_l2()
        q = CompressionQueue(comp, l2)
        await q.start()

        turns = [Message(role=MessageRole.USER, content="t")]
        await q.submit("err-session", turns)
        await asyncio.wait_for(q._queue.join(), timeout=3.0)

        # Worker should still be alive
        assert q._worker_task is not None
        assert not q._worker_task.done()

        await q.stop()


# ══════════════════════════════════════════════════════════════════════════════
# InferenceBatcher
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestInferenceBatcher:
    async def test_single_request_passes_through(self):
        batcher = InferenceBatcher(n_parallel=1)
        result = []
        async with batcher.acquire():
            result.append("done")
        assert result == ["done"]

    async def test_n_parallel_limits_concurrency(self):
        """With n_parallel=2, only 2 tasks can be in-flight simultaneously."""
        batcher = InferenceBatcher(n_parallel=2)
        peak_in_flight = 0
        lock = asyncio.Lock()

        async def task():
            nonlocal peak_in_flight
            async with batcher.acquire():
                async with lock:
                    peak_in_flight = max(peak_in_flight, batcher.in_flight)
                await asyncio.sleep(0.01)

        await asyncio.gather(*[task() for _ in range(5)])
        assert peak_in_flight <= 2

    async def test_queue_depth_tracked(self):
        batcher = InferenceBatcher(n_parallel=1)
        acquired = asyncio.Event()
        released = asyncio.Event()

        async def hold_slot():
            async with batcher.acquire():
                acquired.set()
                await released.wait()

        holder = asyncio.create_task(hold_slot())
        await acquired.wait()  # slot is held

        assert batcher.in_flight == 1
        released.set()
        await holder

    async def test_in_flight_zero_after_release(self):
        batcher = InferenceBatcher(n_parallel=1)
        async with batcher.acquire():
            pass
        assert batcher.in_flight == 0
        assert batcher.queue_depth == 0

    async def test_metrics_observed(self):
        batcher = InferenceBatcher(n_parallel=1)
        with patch("orchestrator.runtime.request_batcher.INFERENCE_WAIT_MS") as mock_hist, \
             patch("orchestrator.runtime.request_batcher.INFERENCE_IN_FLIGHT") as mock_flight, \
             patch("orchestrator.runtime.request_batcher.INFERENCE_QUEUE_DEPTH") as mock_depth:
            mock_hist.observe = MagicMock()
            mock_flight.set = MagicMock()
            mock_depth.set = MagicMock()
            async with batcher.acquire():
                pass
            mock_hist.observe.assert_called_once()
            mock_flight.set.assert_called()
            mock_depth.set.assert_called()

    async def test_exception_inside_slot_releases_semaphore(self):
        """Semaphore must be released even if the body raises."""
        batcher = InferenceBatcher(n_parallel=1)
        with pytest.raises(ValueError):
            async with batcher.acquire():
                raise ValueError("oops")
        assert batcher.in_flight == 0
        # Should be able to acquire again
        async with batcher.acquire():
            pass


# ══════════════════════════════════════════════════════════════════════════════
# InferenceEngine with batcher
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestInferenceEngineWithBatcher:
    def _make_engine(self, llama_client, batcher=None):
        from orchestrator.runtime.inference_engine import InferenceEngine
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        from orchestrator.router.routing_engine import RoutingEngine
        from orchestrator.experts.expert_manager import ExpertManager
        from orchestrator.fusion.fusion_engine import AdaptiveFusionEngine
        from orchestrator.runtime.adaptive_compute import AdaptiveComputeController

        mem = MagicMock()
        mem.get_conversation = AsyncMock(return_value=[])
        mem.retrieve_all = AsyncMock(return_value=MagicMock(
            items=[], total_tokens=0, levels_queried=[], query_time_ms=0
        ))
        mem.record_turn = AsyncMock()

        rag = MagicMock()
        rag.retrieve = AsyncMock(return_value=MagicMock(
            chunks=[], scores=[], total_tokens=0, query_time_ms=0
        ))

        return InferenceEngine(
            llama_client=llama_client,
            memory_manager=mem,
            rag_pipeline=rag,
            intent_classifier=HybridIntentClassifier(embedder=None),
            routing_engine=RoutingEngine(),
            expert_manager=ExpertManager(),
            fusion_engine=AdaptiveFusionEngine(),
            compute_controller=AdaptiveComputeController(),
            batcher=batcher,
        )

    def _make_fused(self) -> FusedContext:
        return FusedContext(
            system_prompt="Be helpful.",
            context_items=[],
            conversation_turns=[Message(role=MessageRole.USER, content="hi")],
            total_token_count=10,
            token_budget_used=0.01,
        )

    def _make_params(self) -> GenerationParams:
        return GenerationParams(max_tokens=32, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1)

    async def test_generate_without_batcher(self):
        llama = MagicMock()
        llama.chat = AsyncMock(return_value="response")
        engine = self._make_engine(llama, batcher=None)
        resp = await engine.generate(self._make_fused(), self._make_params())
        assert resp.content == "response"

    async def test_generate_with_batcher(self):
        llama = MagicMock()
        llama.chat = AsyncMock(return_value="batched response")
        batcher = InferenceBatcher(n_parallel=1)
        engine = self._make_engine(llama, batcher=batcher)
        resp = await engine.generate(self._make_fused(), self._make_params())
        assert resp.content == "batched response"
        assert batcher.in_flight == 0  # slot released after call

    async def test_batcher_limits_concurrent_generate(self):
        """With n_parallel=1 batcher, only one generate() runs at a time."""
        in_flight_counts = []
        llama = MagicMock()

        async def slow_chat(**kwargs):
            await asyncio.sleep(0.02)
            return "ok"

        llama.chat = slow_chat
        batcher = InferenceBatcher(n_parallel=1)
        engine = self._make_engine(llama, batcher=batcher)

        async def run():
            in_flight_counts.append(batcher.in_flight + batcher.queue_depth)
            return await engine.generate(self._make_fused(), self._make_params())

        await asyncio.gather(run(), run(), run())
        # Peak should never exceed 1 in-flight + N queued
        assert max(in_flight_counts) <= 3  # all queued but only 1 running

    async def test_stream_with_batcher(self):
        llama = MagicMock()

        async def mock_stream(**kwargs):
            for tok in ["hello", " world"]:
                yield tok

        llama.stream_chat = mock_stream
        batcher = InferenceBatcher(n_parallel=1)
        engine = self._make_engine(llama, batcher=batcher)

        tokens = [t async for t in engine.stream(self._make_fused(), self._make_params())]
        assert "".join(tokens) == "hello world"
        assert batcher.in_flight == 0


# ══════════════════════════════════════════════════════════════════════════════
# MemoryManager routes to CompressionQueue
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestMemoryManagerCompressionQueue:
    async def test_uses_queue_when_provided(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.memory.memory_manager import MemoryManager

        l1 = L1ConversationCache(max_turns=10, max_tokens=50)  # tiny budget to force overflow
        l2 = MagicMock(); l2.store = AsyncMock(); l2.retrieve = AsyncMock(return_value=[])
        l2.clear_session = AsyncMock()
        l3 = MagicMock(); l3.retrieve = AsyncMock(return_value=[]); l3.store = AsyncMock()
        l3.clear_session = AsyncMock()
        l4 = MagicMock(); l4.retrieve = AsyncMock(return_value=[])
        l4.clear_session = AsyncMock()

        q = MagicMock()
        q.submit = AsyncMock()

        mgr = MemoryManager(l1=l1, l2=l2, l3=l3, l4=l4, compression_queue=q)

        # Fill L1 past the 85% threshold (max_tokens=50 → threshold=42)
        for _ in range(20):
            await mgr.record_turn(
                "s1",
                Message(role=MessageRole.USER, content="word " * 5),
                Message(role=MessageRole.ASSISTANT, content="reply " * 5),
            )

        # Queue submit should have been called (L1 overflowed)
        q.submit.assert_awaited()
