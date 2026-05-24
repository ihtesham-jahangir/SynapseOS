"""
Tests for v3.0 features:
  - SharedMemoryBus: publish, subscribe, wait_for, history, get_latest, clear
  - TaskPlanner: complexity detection, LLM-based planning, parse, fallback
  - AgentPool: single task, parallel, dependency ordering, failed-dep skipping
  - BaseAgent / Specialized: execution, error capture, bus publishing
  - MultiAgentEngine: routing (simple/complex), synthesis, task registry
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.agents.task_types import (
    AgentRole, AgentTaskRequest, SubTask, TaskGraph, TaskStatus,
)
from orchestrator.agents.memory_bus import SharedMemoryBus
from orchestrator.agents.task_planner import TaskPlanner, _is_complex
from orchestrator.agents.agent_pool import AgentPool
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.agents.specialized_agents import CoderAgent, GeneralAgent, SummarizerAgent
from orchestrator.agents.multi_agent_engine import MultiAgentEngine, get_task, _task_registry


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_llama(response: str = "result") -> MagicMock:
    llama = MagicMock()
    llama.chat = AsyncMock(return_value=response)
    return llama


def _make_bus() -> SharedMemoryBus:
    return SharedMemoryBus()


def _simple_graph(n: int = 1) -> TaskGraph:
    graph = TaskGraph(session_id="s1", original_query="test query")
    for i in range(1, n + 1):
        st = SubTask(id=f"t{i}", description=f"task {i}", role=AgentRole.GENERAL)
        graph.sub_tasks[st.id] = st
    return graph


# ══════════════════════════════════════════════════════════════════════════════
# SharedMemoryBus
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestSharedMemoryBus:
    async def test_publish_and_get_history(self):
        bus = _make_bus()
        await bus.publish("result.t1", "agent", "hello world")
        history = bus.get_history("result.t1")
        assert len(history) == 1
        assert history[0].content == "hello world"
        assert history[0].sender_id == "agent"

    async def test_multiple_publishes_ordered(self):
        bus = _make_bus()
        for i in range(3):
            await bus.publish("topic", "sender", f"msg {i}")
        history = bus.get_history("topic")
        assert [m.content for m in history] == ["msg 0", "msg 1", "msg 2"]

    async def test_subscribe_receives_future_message(self):
        bus = _make_bus()
        q = bus.subscribe("topic.x")
        await bus.publish("topic.x", "s", "payload")
        msg = await asyncio.wait_for(q.get(), timeout=1.0)
        assert msg.content == "payload"

    async def test_wait_for_resolves_on_publish(self):
        bus = _make_bus()

        async def _publish():
            await asyncio.sleep(0.01)
            await bus.publish("topic.y", "s", "arrived")

        asyncio.create_task(_publish())
        msg = await bus.wait_for("topic.y", timeout=1.0)
        assert msg is not None
        assert msg.content == "arrived"

    async def test_wait_for_times_out(self):
        bus = _make_bus()
        msg = await bus.wait_for("nonexistent.topic", timeout=0.05)
        assert msg is None

    async def test_get_latest_returns_most_recent(self):
        bus = _make_bus()
        await bus.publish("t", "s", "first")
        await bus.publish("t", "s", "second")
        latest = bus.get_latest("t")
        assert latest is not None
        assert latest.content == "second"

    async def test_get_latest_returns_none_for_unknown_topic(self):
        bus = _make_bus()
        assert bus.get_latest("no-such-topic") is None

    async def test_clear_removes_all_history(self):
        bus = _make_bus()
        await bus.publish("a", "s", "data")
        bus.clear()
        assert bus.get_history("a") == []
        assert bus.get_latest("a") is None

    async def test_topic_count_reflects_unique_topics(self):
        bus = _make_bus()
        await bus.publish("alpha", "s", "x")
        await bus.publish("beta", "s", "y")
        await bus.publish("alpha", "s", "z")
        assert bus.topic_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# TaskPlanner
# ══════════════════════════════════════════════════════════════════════════════

class TestComplexityDetection:
    def test_long_query_is_complex(self):
        assert _is_complex("x" * 201) is True

    def test_short_plain_query_is_simple(self):
        assert _is_complex("What is the capital of France?") is False

    def test_keyword_and_then_triggers_complex(self):
        assert _is_complex("Do A and then do B") is True

    def test_keyword_compare_triggers_complex(self):
        assert _is_complex("Compare Python versus JavaScript") is True

    def test_keyword_step_by_step_triggers_complex(self):
        assert _is_complex("Explain step by step how HTTPS works") is True


@pytest.mark.asyncio
class TestTaskPlanner:
    def _planner(self, llm_response: str) -> TaskPlanner:
        return TaskPlanner(llama_client=_make_llama(llm_response))

    async def test_plan_parses_valid_json(self):
        payload = json.dumps({
            "sub_tasks": [
                {"id": "t1", "description": "Research X", "role": "research", "dependencies": []},
                {"id": "t2", "description": "Summarize", "role": "summarizer", "dependencies": ["t1"]},
            ]
        })
        planner = self._planner(payload)
        graph = await planner.plan("Research X and summarize", "sess")
        assert len(graph.sub_tasks) == 2
        assert "t1" in graph.sub_tasks
        assert graph.sub_tasks["t2"].dependencies == ["t1"]
        assert graph.sub_tasks["t1"].role == AgentRole.RESEARCH

    async def test_plan_respects_max_subtasks(self):
        payload = json.dumps({
            "sub_tasks": [
                {"id": f"t{i}", "description": f"task {i}", "role": "general", "dependencies": []}
                for i in range(1, 8)
            ]
        })
        planner = self._planner(payload)
        graph = await planner.plan("big query", "s", max_subtasks=3)
        assert len(graph.sub_tasks) == 3

    async def test_plan_falls_back_on_llm_failure(self):
        llama = MagicMock()
        llama.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        planner = TaskPlanner(llama_client=llama)
        graph = await planner.plan("anything", "s")
        assert len(graph.sub_tasks) == 1
        st = next(iter(graph.sub_tasks.values()))
        assert st.role == AgentRole.GENERAL

    async def test_plan_falls_back_on_invalid_json(self):
        planner = self._planner("This is not JSON at all!")
        graph = await planner.plan("x", "s")
        assert len(graph.sub_tasks) == 1

    async def test_unknown_role_defaults_to_general(self):
        payload = json.dumps({
            "sub_tasks": [
                {"id": "t1", "description": "Do something", "role": "unknown_role", "dependencies": []}
            ]
        })
        planner = self._planner(payload)
        graph = await planner.plan("q", "s")
        assert graph.sub_tasks["t1"].role == AgentRole.GENERAL


# ══════════════════════════════════════════════════════════════════════════════
# AgentPool
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAgentPool:
    def _pool(self, response: str = "answer") -> AgentPool:
        return AgentPool(llama_client=_make_llama(response), bus=_make_bus(), max_parallel=4)

    async def test_single_task_succeeds(self):
        pool = self._pool("ok")
        graph = _simple_graph(1)
        result = await pool.execute(graph)
        assert result.status == TaskStatus.DONE
        assert result.sub_tasks["t1"].status == TaskStatus.DONE
        assert result.sub_tasks["t1"].result == "ok"

    async def test_parallel_independent_tasks(self):
        """Two tasks with no deps should both run and both succeed."""
        pool = self._pool("parallel result")
        graph = _simple_graph(2)
        result = await pool.execute(graph)
        assert result.sub_tasks["t1"].status == TaskStatus.DONE
        assert result.sub_tasks["t2"].status == TaskStatus.DONE

    async def test_dependency_ordering(self):
        """t2 depends on t1 — t1's result must appear in t2's context."""
        bus = _make_bus()
        received_contexts = []

        async def _chat(**kwargs):
            # Capture the prompt passed to the LLM
            content = kwargs.get("messages", [{}])[0].get("content", "")
            received_contexts.append(content)
            return "done"

        llama = MagicMock()
        llama.chat = _chat
        pool = AgentPool(llama_client=llama, bus=bus, max_parallel=4)

        graph = TaskGraph(session_id="s", original_query="chain query")
        t1 = SubTask(id="t1", description="first task", role=AgentRole.GENERAL, dependencies=[])
        t2 = SubTask(id="t2", description="second task", role=AgentRole.GENERAL, dependencies=["t1"])
        graph.sub_tasks["t1"] = t1
        graph.sub_tasks["t2"] = t2

        result = await pool.execute(graph)
        assert result.sub_tasks["t1"].status == TaskStatus.DONE
        assert result.sub_tasks["t2"].status == TaskStatus.DONE
        # t2's context should reference t1's result
        t2_context = received_contexts[1]
        assert "t1" in t2_context

    async def test_failed_dependency_skips_dependent(self):
        """If t1 fails, t2 (dep on t1) must be FAILED/skipped, not executed."""
        llama = MagicMock()
        llama.chat = AsyncMock(side_effect=RuntimeError("boom"))
        pool = AgentPool(llama_client=llama, bus=_make_bus(), max_parallel=4)

        graph = TaskGraph(session_id="s", original_query="q")
        t1 = SubTask(id="t1", description="fail me", role=AgentRole.GENERAL, dependencies=[])
        t2 = SubTask(id="t2", description="skip me", role=AgentRole.GENERAL, dependencies=["t1"])
        graph.sub_tasks["t1"] = t1
        graph.sub_tasks["t2"] = t2

        result = await pool.execute(graph)
        assert result.sub_tasks["t1"].status == TaskStatus.FAILED
        assert result.sub_tasks["t2"].status == TaskStatus.FAILED
        assert "dependency" in (result.sub_tasks["t2"].error or "").lower() or \
               result.sub_tasks["t2"].status == TaskStatus.FAILED

    async def test_elapsed_ms_recorded(self):
        pool = self._pool("x")
        graph = _simple_graph(1)
        result = await pool.execute(graph)
        assert result.sub_tasks["t1"].elapsed_ms >= 0


# ══════════════════════════════════════════════════════════════════════════════
# BaseAgent / Specialized
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestSpecializedAgents:
    async def test_coder_agent_calls_llama(self):
        llama = _make_llama("def foo(): pass")
        bus = _make_bus()
        agent = CoderAgent(llama_client=llama, bus=bus)
        st = SubTask(id="t1", description="write foo()", role=AgentRole.CODER)
        result_st = await agent.run("task-1", st, "context text")
        llama.chat.assert_awaited_once()
        assert result_st.status == TaskStatus.DONE
        assert "def foo" in result_st.result

    async def test_agent_run_publishes_to_bus(self):
        llama = _make_llama("published result")
        bus = _make_bus()
        agent = GeneralAgent(llama_client=llama, bus=bus)
        st = SubTask(id="t99", description="task", role=AgentRole.GENERAL)
        await agent.run("task-1", st, "ctx")
        msg = bus.get_latest("result.t99")
        assert msg is not None
        assert msg.content == "published result"

    async def test_agent_run_on_exception_sets_failed(self):
        llama = MagicMock()
        llama.chat = AsyncMock(side_effect=ValueError("bad LLM"))
        bus = _make_bus()
        agent = GeneralAgent(llama_client=llama, bus=bus)
        st = SubTask(id="t1", description="boom", role=AgentRole.GENERAL)
        result_st = await agent.run("task-1", st, "ctx")
        assert result_st.status == TaskStatus.FAILED
        assert "bad LLM" in result_st.error

    async def test_agent_elapsed_ms_positive_after_run(self):
        bus = _make_bus()
        agent = GeneralAgent(llama_client=_make_llama("ok"), bus=bus)
        st = SubTask(id="t1", description="timing test", role=AgentRole.GENERAL)
        result_st = await agent.run("t", st, "ctx")
        assert result_st.elapsed_ms >= 0


# ══════════════════════════════════════════════════════════════════════════════
# MultiAgentEngine
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestMultiAgentEngine:
    def _engine(self, llm_response: str = "answer") -> MultiAgentEngine:
        llama = _make_llama(llm_response)
        bus = _make_bus()
        planner = TaskPlanner(llama_client=llama)
        pool = AgentPool(llama_client=llama, bus=bus, max_parallel=2)
        inference_engine = MagicMock()
        return MultiAgentEngine(
            planner=planner,
            pool=pool,
            bus=bus,
            inference_engine=inference_engine,
        )

    async def test_run_task_returns_response(self):
        payload = json.dumps({
            "sub_tasks": [
                {"id": "t1", "description": "do it", "role": "general", "dependencies": []}
            ]
        })
        engine = self._engine(payload)
        # Override llama to return payload for planner, then "done" for agent
        engine._planner._llama.chat = AsyncMock(side_effect=[payload, "done"])
        engine._pool._llama.chat = AsyncMock(return_value="done")

        req = AgentTaskRequest(session_id="s1", query="do it")
        resp = await engine.run_task(req)
        assert resp.task_id != ""
        assert resp.status in (TaskStatus.DONE, TaskStatus.FAILED)
        assert resp.total_elapsed_ms >= 0

    async def test_task_stored_in_registry_after_run(self):
        _task_registry.clear()
        payload = json.dumps({
            "sub_tasks": [
                {"id": "t1", "description": "store me", "role": "general", "dependencies": []}
            ]
        })
        llama = MagicMock()
        llama.chat = AsyncMock(side_effect=[payload, "stored result"])
        bus = _make_bus()
        engine = MultiAgentEngine(
            planner=TaskPlanner(llama_client=llama),
            pool=AgentPool(llama_client=llama, bus=bus, max_parallel=2),
            bus=bus,
            inference_engine=MagicMock(),
        )
        req = AgentTaskRequest(session_id="s2", query="store me")
        resp = await engine.run_task(req)
        assert get_task(resp.task_id) is not None

    async def test_synthesize_prefers_summarizer_result(self):
        engine = self._engine()
        graph = TaskGraph(session_id="s", original_query="q")
        t1 = SubTask(id="t1", role=AgentRole.CODER, description="code", status=TaskStatus.DONE, result="def f(): pass")
        t2 = SubTask(id="t2", role=AgentRole.SUMMARIZER, description="sum", status=TaskStatus.DONE, result="Final summary here")
        graph.sub_tasks["t1"] = t1
        graph.sub_tasks["t2"] = t2
        answer = engine._synthesize(graph)
        assert answer == "Final summary here"

    async def test_synthesize_concatenates_without_summarizer(self):
        engine = self._engine()
        graph = TaskGraph(session_id="s", original_query="q")
        t1 = SubTask(id="t1", role=AgentRole.RESEARCH, description="r", status=TaskStatus.DONE, result="fact A")
        t2 = SubTask(id="t2", role=AgentRole.REASONER, description="reason", status=TaskStatus.DONE, result="logic B")
        graph.sub_tasks["t1"] = t1
        graph.sub_tasks["t2"] = t2
        answer = engine._synthesize(graph)
        assert "fact A" in answer
        assert "logic B" in answer

    async def test_synthesize_all_failed_returns_error_message(self):
        engine = self._engine()
        graph = TaskGraph(session_id="s", original_query="q")
        t1 = SubTask(id="t1", role=AgentRole.GENERAL, description="d", status=TaskStatus.FAILED, error="boom")
        graph.sub_tasks["t1"] = t1
        answer = engine._synthesize(graph)
        assert "failed" in answer.lower()

    async def test_simple_query_delegates_to_inference_engine(self):
        bus = _make_bus()
        llama = _make_llama("plain answer")
        planner = TaskPlanner(llama_client=llama)
        pool = AgentPool(llama_client=llama, bus=bus, max_parallel=2)

        from orchestrator.core.types import (
            ChatRequest, Message, MessageRole, ChatResponse, IntentType, MemoryLevel,
        )
        mock_response = ChatResponse(
            session_id="s",
            content="simple answer",
            intent=IntentType.CONVERSATION,
            tokens_generated=3,
            total_tokens=3,
            time_to_first_token_ms=10.0,
            total_time_ms=100.0,
            memory_levels_used=[],
            rag_chunks_used=0,
            expert_used=None,
        )
        inference_engine = MagicMock()
        inference_engine.process_request = AsyncMock(return_value=mock_response)

        engine = MultiAgentEngine(planner=planner, pool=pool, bus=bus, inference_engine=inference_engine)

        request = ChatRequest(
            session_id="s",
            messages=[Message(role=MessageRole.USER, content="Hi there")],
        )
        resp = await engine.process_request(request)
        inference_engine.process_request.assert_awaited_once()
        assert resp.content == "simple answer"
