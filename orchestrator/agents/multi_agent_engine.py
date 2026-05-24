"""
MultiAgentEngine — top-level v3.0 orchestrator.

Routes incoming requests through two paths:
  - Simple query  → delegates directly to InferenceEngine.process_request()
  - Complex query → TaskPlanner → AgentPool → synthesize → ChatResponse

Also exposes `run_task()` for the dedicated /v1/agents/tasks API endpoint.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from datetime import datetime
from typing import Optional

from orchestrator.agents.task_types import (
    AgentRole,
    AgentTaskRequest,
    AgentTaskResponse,
    TaskGraph,
    TaskStatus,
)
from orchestrator.agents.task_store import TaskStore
from orchestrator.core.types import ChatRequest, ChatResponse, IntentType
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

# In-process task registry — capped at 1 000 entries (LRU eviction)
_MAX_REGISTRY = 1_000
_task_registry: OrderedDict[str, TaskGraph] = OrderedDict()


def _register(graph: TaskGraph) -> None:
    _task_registry[graph.id] = graph
    if len(_task_registry) > _MAX_REGISTRY:
        _task_registry.popitem(last=False)  # evict oldest


def get_task(task_id: str) -> Optional[TaskGraph]:
    """Look up a TaskGraph by ID from the in-process registry."""
    return _task_registry.get(task_id)


async def get_task_from_store(task_id: str, store: TaskStore) -> Optional[TaskGraph]:
    """Look up a TaskGraph — checks in-process registry first, then SQLite."""
    graph = _task_registry.get(task_id)
    if graph is not None:
        return graph
    return await store.load(task_id)


class MultiAgentEngine:
    """
    Top-level v3.0 orchestrator.

    Simple queries are forwarded to InferenceEngine unchanged.
    Complex queries are decomposed, executed by the AgentPool, and
    synthesized into a single final answer.

    Completed tasks are persisted to SQLite via TaskStore so they survive
    process restarts. The in-process LRU registry is checked first (O(1));
    SQLite is the fallback for cross-restart lookups.
    """

    def __init__(
        self,
        planner,
        pool,
        bus,
        inference_engine,
        task_store: Optional[TaskStore] = None,
        response_cache=None,
    ) -> None:
        self._planner = planner
        self._pool = pool
        self._bus = bus
        self._inference_engine = inference_engine
        self._task_store = task_store
        self._cache = response_cache

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_task(self, request: AgentTaskRequest) -> AgentTaskResponse:
        """Decompose and execute a multi-agent task; returns full result."""
        t0 = time.perf_counter()

        # Publish original query so agents can read it from the bus
        await self._bus.publish(
            topic=f"context.{request.session_id}",
            sender_id="orchestrator",
            content=request.query,
        )

        graph = await self._planner.plan(
            query=request.query,
            session_id=request.session_id,
            max_subtasks=request.max_subtasks,
        )

        graph = await self._pool.execute(graph)
        graph.final_answer = self._synthesize(graph)
        _register(graph)

        # Persist to SQLite (non-blocking — fire and forget)
        if self._task_store is not None:
            import asyncio as _asyncio
            _asyncio.create_task(self._task_store.save(graph))

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "Multi-agent task complete",
            task_id=graph.id,
            subtasks=len(graph.sub_tasks),
            elapsed_ms=f"{elapsed:.0f}",
        )

        return AgentTaskResponse(
            task_id=graph.id,
            session_id=request.session_id,
            status=graph.status,
            sub_tasks=list(graph.sub_tasks.values()),
            final_answer=graph.final_answer,
            total_elapsed_ms=elapsed,
        )

    async def process_request(self, request: ChatRequest) -> ChatResponse:
        """
        Entry point for the normal chat pipeline (v3.0 aware).

        Complex queries are routed through multi-agent decomposition.
        Simple queries fall through to InferenceEngine directly.
        """
        query = request.last_user_message

        # Check response cache before decomposing — a complex query may have
        # been answered and cached on a previous call.
        if self._cache and self._cache.should_cache(query, IntentType.REASONING.value):
            cached = await self._cache.get(query)
            if cached:
                log.info("Multi-agent: returning cached result", query_len=len(query))
                answer = cached.content
                return ChatResponse(
                    session_id=request.session_id,
                    content=answer,
                    intent=IntentType.REASONING,
                    tokens_generated=cached.tokens_generated,
                    total_tokens=cached.tokens_generated,
                    time_to_first_token_ms=0.0,
                    total_time_ms=0.0,
                    memory_levels_used=[],
                    rag_chunks_used=0,
                    expert_used=None,
                    metadata={"from_cache": True, "multi_agent": True},
                )

        if self._planner.is_complex(query):
            log.info("Routing to multi-agent pipeline", query_len=len(query))
            task_req = AgentTaskRequest(
                session_id=request.session_id,
                query=query,
            )
            result = await self.run_task(task_req)
            answer = result.final_answer or ""
            response = ChatResponse(
                session_id=request.session_id,
                content=answer,
                intent=IntentType.REASONING,
                tokens_generated=len(answer.split()),
                total_tokens=len(answer.split()),
                time_to_first_token_ms=0.0,
                total_time_ms=result.total_elapsed_ms,
                memory_levels_used=[],
                rag_chunks_used=0,
                expert_used=None,
                metadata={
                    "multi_agent": True,
                    "task_id": result.task_id,
                    "sub_tasks": len(result.sub_tasks),
                },
            )
            # Store in cache so repeated identical queries are served instantly
            if self._cache and answer and self._cache.should_cache(query, IntentType.REASONING.value):
                import asyncio as _asyncio
                _asyncio.create_task(
                    self._cache.set(query, answer, IntentType.REASONING.value, len(answer.split()))
                )
            return response

        return await self._inference_engine.process_request(request)

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, graph: TaskGraph) -> str:
        """Combine sub-task results into a single final answer."""
        done_tasks = [
            st for st in graph.sub_tasks.values() if st.status == TaskStatus.DONE
        ]
        if not done_tasks:
            return "All sub-tasks failed — no result available."

        # Prefer the last summarizer result as the canonical answer
        for st in reversed(done_tasks):
            if st.role == AgentRole.SUMMARIZER and st.result:
                return st.result

        # Single task — return its result directly
        if len(done_tasks) == 1:
            return done_tasks[0].result or ""

        # Multiple tasks without a summarizer — concatenate with labels
        parts = [
            f"**{st.role.value.title()} ({st.id})**:\n{st.result}"
            for st in done_tasks
        ]
        return "\n\n---\n\n".join(parts)
