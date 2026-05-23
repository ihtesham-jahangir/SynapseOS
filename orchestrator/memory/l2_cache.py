"""
L2 Cache – Rolling conversation summaries.

When L1 fills, this layer stores compressed summaries produced by TinyLlama.
Backed by SQLite for durability across restarts.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import List, Optional

import aiosqlite

from orchestrator.core.base import BaseMemoryStore
from orchestrator.core.types import MemoryItem, MemoryLevel
from orchestrator.config.settings import get_settings
from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    turn_start  INTEGER NOT NULL,
    turn_end    INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    importance  REAL DEFAULT 0.5,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_l2_session ON conversation_summaries(session_id);
"""


class L2SummaryCache(BaseMemoryStore):
    """
    Persistent rolling summary store.
    Summaries are generated externally (by the compressor module) and
    injected here for retrieval.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or get_settings().storage.sqlite_path
        self._max_tokens = get_settings().memory.l2_max_tokens
        self._initialized = False

    async def _init(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialized = True

    async def store(self, item: MemoryItem) -> None:
        await self._init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO conversation_summaries
                   (id, session_id, summary, turn_start, turn_end,
                    token_count, importance, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.id,
                    item.session_id or "default",
                    item.content,
                    item.metadata.get("turn_start", 0),
                    item.metadata.get("turn_end", 0),
                    item.token_count or count_tokens(item.content),
                    item.importance_score,
                    time.time(),
                ),
            )
            await db.commit()

    async def retrieve(
        self,
        query: str,
        session_id: str,
        limit: int = 3,
    ) -> List[MemoryItem]:
        """Return most recent summaries for this session (recency-ranked)."""
        await self._init()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM conversation_summaries
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        items: List[MemoryItem] = []
        now = time.time()
        for row in rows:
            age_hours = (now - row["created_at"]) / 3600
            recency = max(0.0, 1.0 - age_hours / 24)  # decays over 24h
            items.append(
                MemoryItem(
                    id=row["id"],
                    content=row["summary"],
                    level=MemoryLevel.L2,
                    session_id=session_id,
                    relevance_score=0.8,
                    recency_score=recency,
                    importance_score=row["importance"],
                    token_count=row["token_count"],
                    timestamp=datetime.fromtimestamp(row["created_at"]),
                    metadata={"turn_start": row["turn_start"], "turn_end": row["turn_end"]},
                )
            )
        return items

    async def clear_session(self, session_id: str) -> None:
        await self._init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?", (session_id,)
            )
            await db.commit()

    async def get_all_for_session(self, session_id: str) -> List[MemoryItem]:
        return await self.retrieve("", session_id, limit=100)
