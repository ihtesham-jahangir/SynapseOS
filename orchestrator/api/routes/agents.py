"""
v3.0 multi-agent task endpoints.

POST /v1/agents/tasks                   — submit a task (blocking, full result)
GET  /v1/agents/tasks/{task_id}         — poll status / result (in-process + SQLite)
GET  /v1/agents/tasks/{task_id}/stream  — SSE stream of sub-task progress
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from orchestrator.agents.task_types import AgentTaskRequest, AgentTaskResponse, TaskStatus
from orchestrator.agents.multi_agent_engine import get_task, get_task_from_store
from orchestrator.api.dependencies import get_container

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.post("/tasks", response_model=AgentTaskResponse)
async def submit_task(request: AgentTaskRequest) -> AgentTaskResponse:
    """Decompose *query* into sub-tasks and execute them via the agent pool."""
    container = get_container()
    return await container.multi_agent_engine.run_task(request)


@router.get("/tasks/{task_id}", response_model=AgentTaskResponse)
async def get_task_status(task_id: str) -> AgentTaskResponse:
    """Return the current status and results of a previously submitted task.

    Checks in-process LRU registry first, then falls back to SQLite for
    tasks submitted in earlier process runs.
    """
    container = get_container()
    graph = await get_task_from_store(task_id, container.task_store)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return AgentTaskResponse(
        task_id=graph.id,
        session_id=graph.session_id,
        status=graph.status,
        sub_tasks=list(graph.sub_tasks.values()),
        final_answer=graph.final_answer,
        total_elapsed_ms=graph.total_elapsed_ms,
    )


@router.get("/tasks/{task_id}/stream")
async def stream_task_progress(task_id: str) -> StreamingResponse:
    """
    SSE stream of sub-task progress for a completed or in-flight task.

    Immediately replays already-completed sub-task events, then subscribes
    to the bus for live events until the task completes.  Sends a final
    ``complete`` event with the synthesized answer.
    """
    container = get_container()
    graph = await get_task_from_store(task_id, container.task_store)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    bus = container.bus

    async def event_gen():
        # Replay already-finished sub-tasks
        for st in graph.sub_tasks.values():
            if st.status == TaskStatus.DONE:
                payload = json.dumps({
                    "type": "subtask_done",
                    "subtask_id": st.id,
                    "role": st.role.value,
                    "elapsed_ms": st.elapsed_ms,
                    "result_preview": (st.result or "")[:200],
                })
                yield f"data: {payload}\n\n"

        # If already complete, send final event and close
        if graph.status in (TaskStatus.DONE, TaskStatus.FAILED):
            final = json.dumps({
                "type": "complete",
                "status": graph.status.value,
                "final_answer": graph.final_answer,
                "total_elapsed_ms": graph.total_elapsed_ms,
            })
            yield f"data: {final}\n\n"
            return

        # Task still in flight — subscribe to live progress events
        q = bus.subscribe(f"progress.{task_id}")
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    yield "data: {\"type\": \"keepalive\"}\n\n"
                    continue

                meta = msg.metadata
                status = meta.get("status", "unknown")
                payload = json.dumps({
                    "type": f"subtask_{status}",
                    "subtask_id": meta.get("subtask_id"),
                    "role": msg.sender_id,
                    "elapsed_ms": meta.get("elapsed_ms", 0),
                    "result_preview": msg.content[:200],
                })
                yield f"data: {payload}\n\n"

                # Check if all tasks in graph are now settled
                current = get_task(task_id)
                if current and current.status in (TaskStatus.DONE, TaskStatus.FAILED):
                    final = json.dumps({
                        "type": "complete",
                        "status": current.status.value,
                        "final_answer": current.final_answer,
                        "total_elapsed_ms": current.total_elapsed_ms,
                    })
                    yield f"data: {final}\n\n"
                    break
        finally:
            bus.unsubscribe(f"progress.{task_id}", q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
