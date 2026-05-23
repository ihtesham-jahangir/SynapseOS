"""
L3 Cache – Semantic vector memory.

Stores conversation facts and important utterances as embeddings.
Retrieval is semantic (cosine similarity), not recency-based.
Uses the shared FAISS index.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import List, Optional

from orchestrator.core.base import BaseMemoryStore
from orchestrator.core.types import MemoryItem, MemoryLevel
from orchestrator.core.exceptions import MemoryError
from orchestrator.config.settings import get_settings
from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.logging_utils import get_logger
from orchestrator.rag.embedder import BGEEmbedder
from orchestrator.rag.indexer import FAISSIndex

log = get_logger(__name__)


class L3VectorMemory(BaseMemoryStore):
    """
    Long-term semantic memory using FAISS embeddings.

    Important facts are extracted from conversations and stored as
    embeddings. Retrieval finds semantically similar memories even
    across sessions (if session_id is not filtered).
    """

    def __init__(
        self,
        embedder: BGEEmbedder,
        index: Optional[FAISSIndex] = None,
    ) -> None:
        self._embedder = embedder
        cfg = get_settings()
        self._index = index or FAISSIndex(
            dimension=384,
            index_path=f"{cfg.rag.faiss_index_path}_memory",
        )
        self._threshold = cfg.memory.l3_similarity_threshold
        self._max_results = cfg.memory.l3_max_results

    async def store(self, item: MemoryItem) -> None:
        try:
            embedding = await self._embedder.embed_single(item.content)
        except Exception as exc:
            raise MemoryError(f"L3 embedding failed: {exc}", "L3") from exc

        meta = {
            "memory_id": item.id,
            "content": item.content,
            "session_id": item.session_id or "",
            "importance": item.importance_score,
            "timestamp": item.timestamp.timestamp(),
            "level": MemoryLevel.L3.value,
            **item.metadata,
        }
        await self._index.add([embedding], [meta])
        await self._index.persist()

    async def retrieve(
        self,
        query: str,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        limit = limit or self._max_results
        try:
            query_vec = await self._embedder.embed_query(query)
        except Exception as exc:
            raise MemoryError(f"L3 query embedding failed: {exc}", "L3") from exc

        results = await self._index.search(
            query_vector=query_vec,
            top_k=limit,
            threshold=self._threshold,
        )

        items: List[MemoryItem] = []
        now = time.time()
        for _id, score, meta in results:
            age_h = (now - meta.get("timestamp", now)) / 3600
            recency = max(0.0, 1.0 - age_h / 168)  # decays over 1 week
            items.append(
                MemoryItem(
                    id=meta.get("memory_id", str(_id)),
                    content=meta.get("content", ""),
                    level=MemoryLevel.L3,
                    session_id=meta.get("session_id"),
                    relevance_score=score,
                    recency_score=recency,
                    importance_score=meta.get("importance", 0.5),
                    token_count=count_tokens(meta.get("content", "")),
                    timestamp=datetime.fromtimestamp(meta.get("timestamp", now)),
                )
            )
        return items

    async def clear_session(self, session_id: str) -> None:
        # FAISS doesn't support selective deletion efficiently;
        # mark items as deleted in metadata on next rebuild
        log.warning("L3 per-session clear is deferred to next index rebuild", session=session_id)
