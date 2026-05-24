"""
TaskStore — SQLite-backed persistence for completed TaskGraphs.

Keeps results durable across process restarts so clients can still
retrieve tasks submitted in a previous session.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import aiosqlite

from orchestrator.agents.task_types import TaskGraph, TaskStatus
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_tasks (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    original_query TEXT NOT NULL,
    status        TEXT NOT NULL,
    final_answer  TEXT,
    total_elapsed_ms REAL DEFAULT 0,
    sub_tasks_json   TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    completed_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_session ON agent_tasks (session_id);
"""


class TaskStore:
    """Async SQLite store for TaskGraph objects."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialized = False

    async def _init(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.execute("PRAGMA journal_mode=WAL")
            await db.commit()
        self._initialized = True

    async def save(self, graph: TaskGraph) -> None:
        await self._init()
        sub_tasks_json = json.dumps(
            {k: v.model_dump(mode="json") for k, v in graph.sub_tasks.items()}
        )
        completed_ts = graph.completed_at.timestamp() if graph.completed_at else None
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO agent_tasks
                  (id, session_id, original_query, status, final_answer,
                   total_elapsed_ms, sub_tasks_json, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    graph.id,
                    graph.session_id,
                    graph.original_query,
                    graph.status.value,
                    graph.final_answer,
                    graph.total_elapsed_ms,
                    sub_tasks_json,
                    graph.created_at.timestamp(),
                    completed_ts,
                ),
            )
            await db.commit()
        log.debug("TaskStore: saved task", task_id=graph.id, status=graph.status.value)

    async def load(self, task_id: str) -> Optional[TaskGraph]:
        await self._init()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_tasks WHERE id = ?", (task_id,)
            ) as cur:
                row = await cur.fetchone()

        if row is None:
            return None

        from orchestrator.agents.task_types import SubTask
        from datetime import datetime

        sub_tasks_data = json.loads(row["sub_tasks_json"] or "{}")
        sub_tasks = {k: SubTask(**v) for k, v in sub_tasks_data.items()}

        return TaskGraph(
            id=row["id"],
            session_id=row["session_id"],
            original_query=row["original_query"],
            status=TaskStatus(row["status"]),
            final_answer=row["final_answer"],
            total_elapsed_ms=row["total_elapsed_ms"] or 0.0,
            sub_tasks=sub_tasks,
            created_at=datetime.fromtimestamp(row["created_at"]),
            completed_at=datetime.fromtimestamp(row["completed_at"]) if row["completed_at"] else None,
        )
