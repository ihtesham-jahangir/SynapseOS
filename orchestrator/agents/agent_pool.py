"""
AgentPool — executes a TaskGraph respecting dependency ordering.

Sub-tasks whose dependencies are all DONE run concurrently (up to
max_parallel at a time).  Results are published to the SharedMemoryBus
so downstream agents can read them via context injection.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Set

from orchestrator.agents.task_types import AgentRole, SubTask, TaskGraph, TaskStatus
from orchestrator.agents.specialized_agents import ROLE_TO_AGENT_CLASS, GeneralAgent
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.metrics import (
    AGENT_TASKS_TOTAL,
    AGENT_TASK_DURATION_MS,
    AGENT_POOL_ACTIVE,
    AGENT_RETRY_TOTAL,
)

log = get_logger(__name__)


class AgentPool:
    """
    Executes sub-tasks from a TaskGraph in topological order.

    Independent sub-tasks (no unmet dependencies) run concurrently.
    Sub-tasks whose dependency failed are skipped immediately.
    Each sub-task is bounded by ``task_timeout_s``; a timeout marks it FAILED.
    """

    def __init__(
        self,
        llama_client,
        bus,
        max_parallel: int = 4,
        task_timeout_s: float = 120.0,
        max_retries: int = 1,
        retry_delay_base_s: float = 2.0,
    ) -> None:
        self._llama = llama_client
        self._bus = bus
        self._max_parallel = max(1, max_parallel)
        self._timeout_s = task_timeout_s
        self._max_retries = max_retries
        self._retry_delay_base = retry_delay_base_s
        self._active = 0

    def _make_agent(self, role: AgentRole):
        cls = ROLE_TO_AGENT_CLASS.get(role, GeneralAgent)
        return cls(self._llama, self._bus)

    def _build_context(self, sub_task: SubTask, graph: TaskGraph) -> str:
        """Assemble context from the original query and completed dependency results."""
        parts = [f"Original request: {graph.original_query}"]
        for dep_id in sub_task.dependencies:
            dep = graph.sub_tasks.get(dep_id)
            if dep and dep.result:
                parts.append(f"[{dep_id} — {dep.role.value} result]:\n{dep.result}")
        return "\n\n".join(parts)

    async def _run_one(self, task_id: str, sub_task: SubTask, context: str) -> SubTask:
        agent = self._make_agent(sub_task.role)
        self._active += 1
        AGENT_POOL_ACTIVE.set(self._active)
        try:
            for attempt in range(self._max_retries + 1):
                try:
                    return await asyncio.wait_for(
                        agent.run(task_id, sub_task, context),
                        timeout=self._timeout_s,
                    )
                except (asyncio.TimeoutError, Exception) as exc:
                    is_timeout = isinstance(exc, asyncio.TimeoutError)
                    if attempt < self._max_retries:
                        delay = self._retry_delay_base * (2 ** attempt)
                        log.warning(
                            "Agent sub-task failed — retrying",
                            subtask=sub_task.id,
                            role=sub_task.role.value,
                            attempt=attempt + 1,
                            max_retries=self._max_retries,
                            delay_s=delay,
                            error=str(exc) if not is_timeout else f"timeout after {self._timeout_s:.0f}s",
                        )
                        await asyncio.sleep(delay)
                        AGENT_RETRY_TOTAL.labels(role=sub_task.role.value).inc()
                        # Re-create agent for retry to avoid stale state
                        agent = self._make_agent(sub_task.role)
                        continue
                    # All attempts exhausted
                    if is_timeout:
                        sub_task.status = TaskStatus.FAILED
                        sub_task.error = f"Timed out after {self._timeout_s:.0f}s ({self._max_retries + 1} attempts)"
                        log.warning(
                            "Agent sub-task timed out (all retries exhausted)",
                            subtask=sub_task.id,
                            role=sub_task.role.value,
                            attempts=self._max_retries + 1,
                        )
                    else:
                        sub_task.status = TaskStatus.FAILED
                        sub_task.error = f"{exc} (after {self._max_retries + 1} attempts)"
                        log.error(
                            "Agent sub-task failed (all retries exhausted)",
                            subtask=sub_task.id,
                            role=sub_task.role.value,
                            error=str(exc),
                        )
                    return sub_task
        finally:
            self._active -= 1
            AGENT_POOL_ACTIVE.set(self._active)

    async def execute(self, graph: TaskGraph) -> TaskGraph:
        """Run all sub-tasks respecting the dependency DAG. Returns the updated graph."""
        graph.status = TaskStatus.RUNNING
        t0 = time.perf_counter()

        done: Set[str] = set()
        failed: Set[str] = set()
        remaining: Set[str] = set(graph.sub_tasks.keys())

        while remaining:
            # Cascade-fail tasks whose dependencies failed — repeat until stable
            # so multi-hop chains (A→B→C, A fails) all propagate in one pass.
            while True:
                newly_blocked = {
                    sid for sid in remaining
                    if any(d in failed for d in graph.sub_tasks[sid].dependencies)
                }
                if not newly_blocked:
                    break
                for sid in newly_blocked:
                    remaining.discard(sid)
                    failed.add(sid)
                    graph.sub_tasks[sid].status = TaskStatus.FAILED
                    graph.sub_tasks[sid].error = "Skipped: dependency failed"
                    AGENT_TASKS_TOTAL.labels(
                        role=graph.sub_tasks[sid].role.value, status="skipped"
                    ).inc()

            if not remaining:
                break

            # Tasks ready to run: all dependencies are in `done`
            runnable = [
                sid for sid in remaining
                if all(d in done for d in graph.sub_tasks[sid].dependencies)
            ]

            if not runnable:
                # True cycle or unresolvable graph — fail all remaining
                for sid in list(remaining):
                    graph.sub_tasks[sid].status = TaskStatus.FAILED
                    graph.sub_tasks[sid].error = "Deadlock: circular dependency"
                    failed.add(sid)
                remaining.clear()
                log.error("AgentPool deadlock detected", task_id=graph.id)
                break

            # Execute up to max_parallel tasks concurrently
            batch = runnable[: self._max_parallel]
            coroutines = [
                self._run_one(
                    graph.id,
                    graph.sub_tasks[sid],
                    self._build_context(graph.sub_tasks[sid], graph),
                )
                for sid in batch
            ]
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            for sid, result in zip(batch, results):
                remaining.discard(sid)
                if isinstance(result, Exception):
                    graph.sub_tasks[sid].status = TaskStatus.FAILED
                    graph.sub_tasks[sid].error = str(result)
                    failed.add(sid)
                    AGENT_TASKS_TOTAL.labels(
                        role=graph.sub_tasks[sid].role.value, status="failed"
                    ).inc()
                else:
                    graph.sub_tasks[sid] = result  # updated SubTask from agent.run()
                    st = graph.sub_tasks[sid]
                    if st.status == TaskStatus.DONE:
                        done.add(sid)
                        AGENT_TASKS_TOTAL.labels(role=st.role.value, status="done").inc()
                        AGENT_TASK_DURATION_MS.labels(role=st.role.value).observe(st.elapsed_ms)
                        # Publish progress so SSE stream endpoint can forward it
                        await self._bus.publish(
                            topic=f"progress.{graph.id}",
                            sender_id=st.role.value,
                            content=st.result or "",
                            metadata={"subtask_id": sid, "status": "done", "elapsed_ms": st.elapsed_ms},
                        )
                    else:
                        failed.add(sid)
                        AGENT_TASKS_TOTAL.labels(role=st.role.value, status="failed").inc()
                        await self._bus.publish(
                            topic=f"progress.{graph.id}",
                            sender_id=st.role.value,
                            content=st.error or "",
                            metadata={"subtask_id": sid, "status": "failed"},
                        )

        graph.status = TaskStatus.DONE if not failed else (
            TaskStatus.FAILED if not done else TaskStatus.DONE  # partial success → DONE
        )
        graph.total_elapsed_ms = (time.perf_counter() - t0) * 1000
        graph.completed_at = datetime.utcnow()

        log.info(
            "AgentPool execution complete",
            task_id=graph.id,
            done=len(done),
            failed=len(failed),
            elapsed_ms=f"{graph.total_elapsed_ms:.0f}",
        )
        return graph

    @property
    def active_count(self) -> int:
        return self._active
