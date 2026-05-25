"""
L2 Cache – Rolling conversation summaries.

When L1 fills, this layer stores compressed summaries produced by TinyLlama.
Backed by SQLite for durability across restarts.
"""
from __future__ import annotations

import asyncio
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

# WAL-mode pragmas run after schema creation (file-level; silently ignored for :memory:)
_WAL_PRAGMAS = [
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
]


class L2SummaryCache(BaseMemoryStore):
    """
    Persistent rolling summary store.
    Summaries are generated externally (by the compressor module) and
    injected here for retrieval.

    Uses a single persistent aiosqlite connection (aiosqlite serialises all
    operations through an internal worker thread, so sharing one connection
    across coroutines is safe and avoids per-query thread creation overhead).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or get_settings().storage.sqlite_path
        self._max_tokens = get_settings().memory.l2_max_tokens
        self._db: Optional[aiosqlite.Connection] = None
        self._init_lock = asyncio.Lock()

    async def _get_db(self) -> aiosqlite.Connection:
        """Return the shared connection, creating and initialising it lazily."""
        if self._db is not None:
            return self._db
        async with self._init_lock:
            if self._db is not None:
                return self._db
            db = await aiosqlite.connect(self._db_path)
            await db.executescript(_SCHEMA)
            for pragma in _WAL_PRAGMAS:
                await db.execute(pragma)
            await db.commit()
            db.row_factory = aiosqlite.Row
            self._db = db
            log.debug("L2 SQLite persistent connection opened", db=self._db_path)
        return self._db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def store(self, item: MemoryItem) -> None:
        db = await self._get_db()
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
        """Return summaries ranked by a blend of recency and keyword relevance."""
        db = await self._get_db()
        # Fetch more candidates than needed so we can re-rank
        fetch_limit = max(limit * 4, 12)
        async with db.execute(
            """SELECT * FROM conversation_summaries
               WHERE session_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (session_id, fetch_limit),
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return []

        query_terms = set(query.lower().split()) - {"a", "an", "the", "is", "in", "of", "to", "and"}
        now = time.time()

        scored: list = []
        for row in rows:
            age_hours = (now - row["created_at"]) / 3600
            recency = max(0.0, 1.0 - age_hours / 24)  # decays over 24h

            # Keyword overlap score
            summary_words = set(row["summary"].lower().split())
            if query_terms:
                overlap = len(query_terms & summary_words) / len(query_terms)
            else:
                overlap = 0.0

            # Blend: 60% recency, 40% keyword relevance
            score = 0.6 * recency + 0.4 * overlap
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        items: List[MemoryItem] = []
        for score, row in scored[:limit]:
            age_hours = (now - row["created_at"]) / 3600
            recency = max(0.0, 1.0 - age_hours / 24)
            items.append(
                MemoryItem(
                    id=row["id"],
                    content=row["summary"],
                    level=MemoryLevel.L2,
                    session_id=session_id,
                    relevance_score=score,
                    recency_score=recency,
                    importance_score=row["importance"],
                    token_count=row["token_count"],
                    timestamp=datetime.fromtimestamp(row["created_at"]),
                    metadata={"turn_start": row["turn_start"], "turn_end": row["turn_end"]},
                )
            )
        return items

    async def clear_session(self, session_id: str) -> None:
        db = await self._get_db()
        await db.execute(
            "DELETE FROM conversation_summaries WHERE session_id = ?", (session_id,)
        )
        await db.commit()

    async def get_all_for_session(self, session_id: str) -> List[MemoryItem]:
        return await self.retrieve("", session_id, limit=100)
