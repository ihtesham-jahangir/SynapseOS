"""
L4 Cache – Persistent knowledge base.

Structured facts, domain knowledge, and curated information stored in SQLite.
Highest priority: always injected unless token budget is exhausted.
Can be pre-populated via admin API or bulk import.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from orchestrator.core.base import BaseMemoryStore
from orchestrator.core.types import MemoryItem, MemoryLevel
from orchestrator.config.settings import get_settings
from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_base (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    category        TEXT DEFAULT 'general',
    keywords        TEXT DEFAULT '',
    priority        REAL DEFAULT 0.8,
    access_count    INTEGER DEFAULT 0,
    last_accessed   REAL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_l4_category ON knowledge_base(category);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    id, content, keywords,
    content='knowledge_base',
    content_rowid='rowid'
);
"""


class L4KnowledgeBase(BaseMemoryStore):
    """
    Keyword + priority-ranked knowledge retrieval.
    Uses FTS5 for fast keyword search within SQLite.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or get_settings().storage.sqlite_path
        self._max_results = get_settings().memory.l4_max_results
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
        category = item.metadata.get("category", "general")
        keywords = item.metadata.get("keywords", "")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO knowledge_base
                   (id, content, category, keywords, priority, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    item.id,
                    item.content,
                    category,
                    keywords,
                    item.importance_score,
                    time.time(),
                ),
            )
            # Keep FTS in sync
            await db.execute(
                "INSERT OR REPLACE INTO knowledge_fts(id, content, keywords) VALUES (?,?,?)",
                (item.id, item.content, keywords),
            )
            await db.commit()

    async def retrieve(
        self,
        query: str,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        """FTS5 keyword search ranked by priority."""
        await self._init()
        limit = limit or self._max_results

        # Build FTS query from query keywords
        fts_query = " OR ".join(
            f'"{word}"' for word in query.split()[:8] if len(word) > 2
        )
        if not fts_query:
            return []

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                async with db.execute(
                    """SELECT kb.*, rank
                       FROM knowledge_fts
                       JOIN knowledge_base kb ON knowledge_fts.id = kb.id
                       WHERE knowledge_fts MATCH ?
                       ORDER BY priority DESC, rank
                       LIMIT ?""",
                    (fts_query, limit),
                ) as cursor:
                    rows = await cursor.fetchall()
            except Exception:
                # FTS query syntax error – fall back to LIKE search
                rows = []
                async with db.execute(
                    """SELECT * FROM knowledge_base
                       WHERE content LIKE ?
                       ORDER BY priority DESC LIMIT ?""",
                    (f"%{query[:50]}%", limit),
                ) as cursor:
                    rows = await cursor.fetchall()

            # Update access stats
            if rows:
                ids = [row["id"] for row in rows]
                placeholders = ",".join("?" * len(ids))
                await db.execute(
                    f"UPDATE knowledge_base SET access_count = access_count + 1, "
                    f"last_accessed = ? WHERE id IN ({placeholders})",
                    [time.time(), *ids],
                )
                await db.commit()

        items: List[MemoryItem] = []
        for row in rows:
            items.append(
                MemoryItem(
                    id=row["id"],
                    content=row["content"],
                    level=MemoryLevel.L4,
                    session_id=None,
                    relevance_score=0.9,
                    recency_score=1.0,  # Knowledge base items don't decay
                    importance_score=row["priority"],
                    token_count=count_tokens(row["content"]),
                    timestamp=datetime.fromtimestamp(row["created_at"]),
                    metadata={"category": row["category"]},
                )
            )
        return items

    async def clear_session(self, session_id: str) -> None:
        pass  # L4 is global, not session-scoped

    async def add_knowledge(
        self,
        content: str,
        category: str = "general",
        keywords: str = "",
        priority: float = 0.8,
    ) -> str:
        """Convenience method for admin knowledge injection."""
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            level=MemoryLevel.L4,
            importance_score=priority,
            metadata={"category": category, "keywords": keywords},
        )
        await self.store(item)
        return item.id

    async def list_categories(self) -> List[str]:
        await self._init()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT DISTINCT category FROM knowledge_base ORDER BY category"
            ) as cursor:
                rows = await cursor.fetchall()
        return [row[0] for row in rows]
