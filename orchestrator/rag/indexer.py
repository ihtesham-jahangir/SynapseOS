"""
FAISS index management for the RAG vector store.
Supports add, search, persist, and load operations.
Uses IndexFlatIP (inner product) with pre-normalized vectors = cosine similarity.
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from orchestrator.core.exceptions import RAGError
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.async_utils import run_in_executor

log = get_logger(__name__)


class FAISSIndex:
    """
    Persistent FAISS flat index with metadata side-table.

    Architecture:
      - faiss IndexFlatIP: exact cosine search (normalized vecs)
      - metadata dict: maps int64 FAISS id → arbitrary metadata dict
      - All mutating operations are serialized by an asyncio.Lock
    """

    def __init__(self, dimension: int = 384, index_path: Optional[str] = None) -> None:
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self._metadata: Dict[int, Dict] = {}
        self._next_id: int = 0
        self._lock = asyncio.Lock()
        self._index = None  # Lazy-loaded

    def _build_index(self):
        import faiss
        return faiss.IndexFlatIP(self.dimension)

    async def _ensure_index(self) -> None:
        if self._index is not None:
            return
        async with self._lock:
            if self._index is not None:
                return
            if self.index_path and self.index_path.with_suffix(".faiss").exists():
                await self.load()
            else:
                self._index = await run_in_executor(self._build_index)
                log.info("Created new FAISS index", dimension=self.dimension)

    async def add(
        self,
        vectors: List[List[float]],
        metadata_list: List[Dict],
    ) -> List[int]:
        """Add vectors + metadata; returns list of assigned IDs."""
        await self._ensure_index()
        arr = np.array(vectors, dtype=np.float32)

        async with self._lock:
            ids = list(range(self._next_id, self._next_id + len(vectors)))
            self._next_id += len(vectors)

            id_arr = np.array(ids, dtype=np.int64)

            def _add():
                self._index.add(arr)

            await run_in_executor(_add)
            for i, meta in zip(ids, metadata_list):
                self._metadata[i] = meta

        return ids

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 6,
        threshold: float = 0.0,
    ) -> List[Tuple[int, float, Dict]]:
        """
        Return (id, score, metadata) tuples sorted by descending cosine similarity.
        """
        await self._ensure_index()
        if self._index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)

        def _search() -> Tuple[np.ndarray, np.ndarray]:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(q, k)
            return scores[0], indices[0]

        scores, indices = await run_in_executor(_search)

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            if score < threshold:
                continue
            meta = self._metadata.get(int(idx), {})
            results.append((int(idx), float(score), meta))

        return results

    async def persist(self) -> None:
        if self.index_path is None:
            return
        async with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss_file = self.index_path.with_suffix(".faiss")
            meta_file = self.index_path.with_suffix(".meta")

            def _save():
                import faiss
                faiss.write_index(self._index, str(faiss_file))

            await run_in_executor(_save)
            with open(meta_file, "wb") as f:
                pickle.dump({"metadata": self._metadata, "next_id": self._next_id}, f)
            log.info("FAISS index persisted", path=str(faiss_file), total=self._index.ntotal)

    async def load(self) -> None:
        if self.index_path is None:
            return
        faiss_file = self.index_path.with_suffix(".faiss")
        meta_file = self.index_path.with_suffix(".meta")

        def _load():
            import faiss
            return faiss.read_index(str(faiss_file))

        self._index = await run_in_executor(_load)
        if meta_file.exists():
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
                self._metadata = data.get("metadata", {})
                self._next_id = data.get("next_id", self._index.ntotal)
        log.info("FAISS index loaded", total=self._index.ntotal)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Soft-delete all chunks belonging to ``doc_id``.
        Returns the number of chunks marked for deletion.
        Call ``rebuild_without_deleted()`` + ``persist()`` afterward.
        """
        await self._ensure_index()
        async with self._lock:
            count = 0
            for meta in self._metadata.values():
                if meta.get("doc_id") == doc_id and not meta.get("_deleted"):
                    meta["_deleted"] = True
                    count += 1
            log.info("Soft-deleted chunks", doc_id=doc_id, count=count)
            return count

    async def rebuild_without_deleted(self) -> int:
        """
        Rebuild the FAISS index from scratch, excluding soft-deleted entries.

        Uses ``IndexFlatIP.reconstruct()`` to retrieve original vectors —
        safe because IndexFlatIP stores all vectors verbatim.

        Returns the number of vectors in the rebuilt index.
        """
        await self._ensure_index()
        async with self._lock:
            import faiss

            live_ids = [
                vid for vid, meta in sorted(self._metadata.items())
                if not meta.get("_deleted")
            ]

            if not live_ids:
                self._index = await run_in_executor(self._build_index)
                self._metadata = {}
                self._next_id = 0
                log.info("FAISS index rebuilt (empty — all entries deleted)")
                return 0

            # Reconstruct original vectors from the flat index
            def _reconstruct() -> np.ndarray:
                return np.vstack([
                    self._index.reconstruct(vid).reshape(1, -1)
                    for vid in live_ids
                ]).astype(np.float32)

            vectors = await run_in_executor(_reconstruct)
            live_meta = [self._metadata[vid] for vid in live_ids]

            # Build fresh index with only the surviving entries
            new_index = await run_in_executor(self._build_index)

            def _add_back():
                new_index.add(vectors)

            await run_in_executor(_add_back)

            self._index = new_index
            self._metadata = {i: meta for i, meta in enumerate(live_meta)}
            self._next_id = len(live_ids)

            log.info("FAISS index rebuilt", surviving=len(live_ids))
            return len(live_ids)

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal
