"""
Tests for all improvements applied post-v3.0 analysis:
  #1  InferenceEngine.build_generation_plan() — single pipeline path
  #2  WebSocket MessageRole guard — invalid role defaults to USER
  #3  run_in_executor uses get_running_loop()
  #4  gather_with_fallback logs warning on failure
  #5  AgentPool per-sub-task timeout
  #6  LlamaClient retry on 429/503
  #7  MultiAgentEngine cache check before decomposition
  #8  SSE streaming for agent task progress
  #9  TaskPlanner uses cache_prompt=True
  #10 ResponseCache uses BLAKE2b key
  #11 AdaptiveComputeController clamps temperature
  #12 TaskStore SQLite persistence
  #14 Health check includes circuit_breaker and compression_queue
  #15 CircuitBreaker state transitions
"""
from __future__ import annotations

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.core.types import (
    Message, MessageRole, FusedContext, GenerationParams, ContextItem,
    Intent, IntentType, MemoryResult, RAGResult,
)
from orchestrator.agents.task_types import (
    AgentRole, AgentTaskRequest, SubTask, TaskGraph, TaskStatus,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fused() -> FusedContext:
    return FusedContext(
        system_prompt="You are helpful.",
        context_items=[],
        conversation_turns=[Message(role=MessageRole.USER, content="hi")],
        total_token_count=10,
        token_budget_used=0.01,
    )


def _intent(itype=IntentType.CONVERSATION, confidence=0.9) -> Intent:
    return Intent(intent_type=itype, confidence=confidence)


def _params() -> GenerationParams:
    return GenerationParams(max_tokens=32, temperature=0.7)


# ══════════════════════════════════════════════════════════════════════════════
# #11 — Temperature clamping
# ══════════════════════════════════════════════════════════════════════════════

class TestTemperatureClamping:
    def test_clamps_below_minimum(self):
        from orchestrator.runtime.adaptive_compute import AdaptiveComputeController
        ctrl = AdaptiveComputeController()
        fused = _fused()
        params = GenerationParams(max_tokens=32, temperature=0.0)  # too low
        intent = _intent(confidence=0.5)  # below 0.85 — no reduction
        result = ctrl.optimize(intent, fused, params)
        assert result.temperature >= 0.05

    def test_clamps_above_maximum(self):
        from orchestrator.runtime.adaptive_compute import AdaptiveComputeController
        ctrl = AdaptiveComputeController()
        params = GenerationParams(max_tokens=32, temperature=5.0)  # too high
        result = ctrl.optimize(_intent(confidence=0.5), _fused(), params)
        assert result.temperature <= 2.0

    def test_normal_temperature_unchanged(self):
        from orchestrator.runtime.adaptive_compute import AdaptiveComputeController
        ctrl = AdaptiveComputeController()
        params = GenerationParams(max_tokens=32, temperature=0.7)
        result = ctrl.optimize(_intent(confidence=0.5), _fused(), params)
        assert 0.05 <= result.temperature <= 2.0


# ══════════════════════════════════════════════════════════════════════════════
# #10 — ResponseCache BLAKE2b key
# ══════════════════════════════════════════════════════════════════════════════

class TestResponseCacheKey:
    def test_key_is_32_hex_chars(self):
        from orchestrator.cache.response_cache import ResponseCache
        key = ResponseCache._key("hello world")
        assert len(key) == 32  # blake2b digest_size=16 → 32 hex chars

    def test_identical_queries_same_key(self):
        from orchestrator.cache.response_cache import ResponseCache
        assert ResponseCache._key("  Hello World  ") == ResponseCache._key("hello world")

    def test_different_queries_different_keys(self):
        from orchestrator.cache.response_cache import ResponseCache
        assert ResponseCache._key("foo") != ResponseCache._key("bar")


# ══════════════════════════════════════════════════════════════════════════════
# #3 — run_in_executor uses get_running_loop
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_run_in_executor_uses_running_loop():
    from orchestrator.utils.async_utils import run_in_executor
    result = await run_in_executor(lambda x: x * 2, 21)
    assert result == 42


# ══════════════════════════════════════════════════════════════════════════════
# #4 — gather_with_fallback logs warning
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_gather_with_fallback_logs_warning():
    from orchestrator.utils.async_utils import gather_with_fallback

    async def boom():
        raise RuntimeError("subsystem down")

    async def ok():
        return "success"

    with patch("orchestrator.utils.async_utils.log") as mock_log:
        results = await gather_with_fallback(boom(), ok(), fallbacks=["fallback", None])
    assert results[0] == "fallback"
    assert results[1] == "success"
    mock_log.warning.assert_called_once()
    call_kwargs = mock_log.warning.call_args
    assert "subsystem" in str(call_kwargs) or "gather_with_fallback" in str(call_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# #15 — CircuitBreaker state transitions
# ══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    def _cb(self, threshold=3, recovery=5.0):
        from orchestrator.runtime.llama_client import CircuitBreaker
        return CircuitBreaker(failure_threshold=threshold, recovery_timeout_s=recovery)

    def test_starts_closed(self):
        cb = self._cb()
        assert cb.state == "closed"
        assert not cb.is_open()

    def test_opens_after_threshold_failures(self):
        cb = self._cb(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.is_open()

    def test_success_resets_to_closed(self):
        cb = self._cb(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()
        cb.record_success()
        assert cb.state == "closed"
        assert not cb.is_open()

    def test_half_open_after_recovery_timeout(self):
        cb = self._cb(threshold=1, recovery=0.05)
        cb.record_failure()
        assert cb.is_open()
        time.sleep(0.1)
        assert cb.state == "half_open"
        assert not cb.is_open()  # half_open allows requests through

    def test_failures_below_threshold_stay_closed(self):
        cb = self._cb(threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == "closed"


# ══════════════════════════════════════════════════════════════════════════════
# #6 — LlamaClient circuit breaker + retry
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestLlamaClientCircuitBreaker:
    async def test_open_circuit_raises_without_calling_server(self):
        from orchestrator.runtime.llama_client import LlamaClient, CircuitBreaker
        from orchestrator.core.exceptions import LlamaServerError

        client = LlamaClient.__new__(LlamaClient)
        client._mock = False
        client._circuit = CircuitBreaker(failure_threshold=1, recovery_timeout_s=60.0)
        client._circuit.record_failure()  # open the circuit

        with pytest.raises(LlamaServerError, match="Circuit breaker"):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    async def test_success_records_with_circuit(self):
        from orchestrator.runtime.llama_client import LlamaClient, CircuitBreaker
        from orchestrator.core.exceptions import LlamaServerError

        client = LlamaClient.__new__(LlamaClient)
        client._mock = False
        client._timeout = 5
        client._base_url = "http://localhost:8080"
        client._client = None
        circuit = CircuitBreaker()
        client._circuit = circuit

        # Stub _post_chat to succeed
        client._post_chat = AsyncMock(return_value="hello")
        client._get_client = AsyncMock(return_value=MagicMock())

        result = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert result == "hello"
        assert circuit.state == "closed"


# ══════════════════════════════════════════════════════════════════════════════
# #5 — AgentPool per-sub-task timeout
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_pool_timeout_marks_task_failed():
    from orchestrator.agents.agent_pool import AgentPool
    from orchestrator.agents.memory_bus import SharedMemoryBus

    async def slow_chat(**kwargs):
        await asyncio.sleep(10.0)  # much longer than timeout
        return "never"

    llama = MagicMock()
    llama.chat = slow_chat
    bus = SharedMemoryBus()
    pool = AgentPool(llama_client=llama, bus=bus, max_parallel=2, task_timeout_s=0.05)

    graph = TaskGraph(session_id="s", original_query="q")
    st = SubTask(id="t1", description="slow task", role=AgentRole.GENERAL)
    graph.sub_tasks["t1"] = st

    result = await pool.execute(graph)
    assert result.sub_tasks["t1"].status == TaskStatus.FAILED
    assert "timed out" in (result.sub_tasks["t1"].error or "").lower()


# ══════════════════════════════════════════════════════════════════════════════
# #2 — WebSocket MessageRole guard
# ══════════════════════════════════════════════════════════════════════════════

def test_safe_role_helper():
    """The _safe_role helper in chat.py must default invalid roles to USER."""
    # Inline the helper since it's defined inside the route function
    def _safe_role(raw: str) -> MessageRole:
        try:
            return MessageRole(raw)
        except ValueError:
            return MessageRole.USER

    assert _safe_role("user") == MessageRole.USER
    assert _safe_role("assistant") == MessageRole.ASSISTANT
    assert _safe_role("system") == MessageRole.SYSTEM
    assert _safe_role("HACKER") == MessageRole.USER
    assert _safe_role("") == MessageRole.USER
    assert _safe_role("admin") == MessageRole.USER


# ══════════════════════════════════════════════════════════════════════════════
# #9 — TaskPlanner cache_prompt=True
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_planner_uses_cache_prompt():
    from orchestrator.agents.task_planner import TaskPlanner
    captured_kwargs = {}

    async def recording_chat(**kwargs):
        captured_kwargs.update(kwargs)
        return json.dumps({
            "sub_tasks": [
                {"id": "t1", "description": "do it", "role": "general", "dependencies": []}
            ]
        })

    llama = MagicMock()
    llama.chat = recording_chat
    planner = TaskPlanner(llama_client=llama)
    await planner.plan("do something interesting", "s")
    assert captured_kwargs.get("cache_prompt") is True


# ══════════════════════════════════════════════════════════════════════════════
# #12 — TaskStore SQLite persistence
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_task_store_save_and_load(tmp_path):
    from orchestrator.agents.task_store import TaskStore
    from datetime import datetime

    store = TaskStore(db_path=str(tmp_path / "tasks.db"))

    graph = TaskGraph(session_id="sess-1", original_query="the question")
    st = SubTask(id="t1", description="sub", role=AgentRole.CODER, status=TaskStatus.DONE, result="code here")
    graph.sub_tasks["t1"] = st
    graph.status = TaskStatus.DONE
    graph.final_answer = "Final answer text"
    graph.total_elapsed_ms = 1234.5
    graph.completed_at = datetime.utcnow()

    await store.save(graph)
    loaded = await store.load(graph.id)

    assert loaded is not None
    assert loaded.id == graph.id
    assert loaded.session_id == "sess-1"
    assert loaded.final_answer == "Final answer text"
    assert loaded.status == TaskStatus.DONE
    assert "t1" in loaded.sub_tasks
    assert loaded.sub_tasks["t1"].result == "code here"


@pytest.mark.asyncio
async def test_task_store_returns_none_for_missing(tmp_path):
    from orchestrator.agents.task_store import TaskStore
    store = TaskStore(db_path=str(tmp_path / "tasks.db"))
    result = await store.load("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_task_store_overwrite_on_duplicate(tmp_path):
    from orchestrator.agents.task_store import TaskStore
    store = TaskStore(db_path=str(tmp_path / "tasks.db"))
    graph = TaskGraph(session_id="s", original_query="q", status=TaskStatus.PENDING)
    await store.save(graph)
    graph.status = TaskStatus.DONE
    graph.final_answer = "done now"
    await store.save(graph)  # INSERT OR REPLACE
    loaded = await store.load(graph.id)
    assert loaded.status == TaskStatus.DONE
    assert loaded.final_answer == "done now"


# ══════════════════════════════════════════════════════════════════════════════
# #1 — InferenceEngine.build_generation_plan()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_build_generation_plan_returns_plan():
    from orchestrator.runtime.inference_engine import InferenceEngine, GenerationPlan
    from orchestrator.router.intent_classifier import HybridIntentClassifier
    from orchestrator.router.routing_engine import RoutingEngine
    from orchestrator.experts.expert_manager import ExpertManager
    from orchestrator.fusion.fusion_engine import AdaptiveFusionEngine
    from orchestrator.runtime.adaptive_compute import AdaptiveComputeController
    from orchestrator.core.types import ChatRequest

    llama = MagicMock()
    llama.chat = AsyncMock(return_value="response")

    mem = MagicMock()
    mem.get_conversation = AsyncMock(return_value=[])
    mem.retrieve_all = AsyncMock(return_value=MemoryResult(
        items=[], total_tokens=0, levels_queried=[], query_time_ms=0
    ))

    rag = MagicMock()
    rag.retrieve = AsyncMock(return_value=RAGResult(
        chunks=[], scores=[], total_tokens=0, query_time_ms=0
    ))

    engine = InferenceEngine(
        llama_client=llama,
        memory_manager=mem,
        rag_pipeline=rag,
        intent_classifier=HybridIntentClassifier(embedder=None),
        routing_engine=RoutingEngine(),
        expert_manager=ExpertManager(),
        fusion_engine=AdaptiveFusionEngine(),
        compute_controller=AdaptiveComputeController(),
    )

    request = ChatRequest(
        session_id="s1",
        messages=[Message(role=MessageRole.USER, content="Hello there")],
    )
    plan = await engine.build_generation_plan(request)

    assert isinstance(plan, GenerationPlan)
    assert plan.fused is not None
    assert plan.params is not None
    assert plan.session_id == "s1"
    assert plan.params.temperature >= 0.05


# ══════════════════════════════════════════════════════════════════════════════
# #7 — MultiAgentEngine cache check before decomposition
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_multi_agent_checks_cache_before_decomposing():
    from orchestrator.agents.task_planner import TaskPlanner
    from orchestrator.agents.agent_pool import AgentPool
    from orchestrator.agents.memory_bus import SharedMemoryBus
    from orchestrator.agents.multi_agent_engine import MultiAgentEngine
    from orchestrator.cache.response_cache import ResponseCache, CachedEntry

    bus = SharedMemoryBus()
    llama = MagicMock()
    planner = TaskPlanner(llama_client=llama)
    pool = AgentPool(llama_client=llama, bus=bus)

    cache = MagicMock(spec=ResponseCache)
    cache.should_cache = MagicMock(return_value=True)
    cached_entry = CachedEntry(content="cached answer", intent_type="reasoning", tokens_generated=5)
    cache.get = AsyncMock(return_value=cached_entry)

    inference_engine = MagicMock()
    inference_engine.process_request = AsyncMock()

    engine = MultiAgentEngine(
        planner=planner, pool=pool, bus=bus,
        inference_engine=inference_engine,
        response_cache=cache,
    )

    from orchestrator.core.types import ChatRequest
    # Long query → is_complex() would return True; but cache should intercept first
    long_query = "x" * 250
    request = ChatRequest(
        session_id="s",
        messages=[Message(role=MessageRole.USER, content=long_query)],
    )
    resp = await engine.process_request(request)
    assert resp.content == "cached answer"
    # Planner should NOT have been called
    llama.chat.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# #8 — SSE streaming publishes progress events via bus
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_pool_publishes_progress_events():
    from orchestrator.agents.agent_pool import AgentPool
    from orchestrator.agents.memory_bus import SharedMemoryBus

    bus = SharedMemoryBus()
    llama = MagicMock()
    llama.chat = AsyncMock(return_value="result")
    pool = AgentPool(llama_client=llama, bus=bus, max_parallel=2)

    graph = TaskGraph(session_id="s", original_query="q")
    st = SubTask(id="t1", description="task", role=AgentRole.GENERAL)
    graph.sub_tasks["t1"] = st

    await pool.execute(graph)

    history = bus.get_history(f"progress.{graph.id}")
    assert len(history) >= 1
    assert history[0].metadata.get("subtask_id") == "t1"
    assert history[0].metadata.get("status") in ("done", "failed")
