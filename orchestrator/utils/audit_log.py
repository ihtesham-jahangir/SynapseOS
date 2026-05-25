"""
Append-only audit log for admin actions.

Records who did what and when to a SQLite table so destructive admin
operations (session clear, cache flush, document delete) are traceable.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

import aiosqlite

from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,
    actor       TEXT NOT NULL DEFAULT 'api',
    detail      TEXT NOT NULL DEFAULT '',
    timestamp   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log (timestamp DESC);
"""


class AuditLog:
    """Async SQLite-backed append-only audit trail."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialized = False

    async def _init(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialized = True

    async def record(self, action: str, detail: str = "", actor: str = "api") -> None:
        """Append an audit entry (fire-and-forget safe)."""
        try:
            await self._init()
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "INSERT INTO audit_log (action, actor, detail, timestamp) VALUES (?, ?, ?, ?)",
                    (action, actor, detail, time.time()),
                )
                await db.commit()
        except Exception as exc:
            log.error("Audit log write failed", action=action, error=str(exc))

    async def list(self, limit: int = 50, offset: int = 0) -> List[dict]:
        """Return recent audit entries, newest first."""
        try:
            await self._init()
            rows = []
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT id, action, actor, detail, timestamp "
                    "FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ) as cur:
                    async for row in cur:
                        rows.append({
                            "id": row["id"],
                            "action": row["action"],
                            "actor": row["actor"],
                            "detail": row["detail"],
                            "timestamp": datetime.fromtimestamp(row["timestamp"]).isoformat(),
                        })
            return rows
        except Exception as exc:
            log.error("Audit log read failed", error=str(exc))
            return []
