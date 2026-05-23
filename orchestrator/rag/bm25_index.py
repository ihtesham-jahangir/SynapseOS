"""
BM25 keyword index — complements FAISS vector search.

Stores a plain-text corpus alongside the FAISS index and provides
fast keyword-based retrieval using the BM25Okapi ranking function.
Persists to a JSON file so the corpus survives restarts.

Used by HybridRetriever which merges BM25 + semantic scores via RRF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.core.types import DocumentChunk
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class BM25Index:
    """
    Keyword index backed by BM25Okapi.

    Corpus is kept in memory and optionally persisted as a JSON file.
    The BM25 model is rebuilt on every ``add()`` call — this is
    O(n) but fast enough for corpora up to ~100 k chunks.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._chunks: List[DocumentChunk] = []
        self._corpus: List[List[str]] = []   # tokenized text per chunk
        self._bm25 = None
        self._persist_path = persist_path

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, chunks: List[DocumentChunk]) -> None:
        """Append chunks and rebuild the BM25 model."""
        if not chunks:
            return
        from rank_bm25 import BM25Plus  # BM25Plus has always-positive IDF (log((N+1)/n))

        self._chunks.extend(chunks)
        self._corpus.extend(c.content.lower().split() for c in chunks)
        self._bm25 = BM25Plus(self._corpus)
        log.debug("BM25 index updated", total_docs=len(self._chunks))

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Return (chunk, score) pairs ranked by BM25 relevance.
        Zero-score results are excluded.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = query.lower().split()
        scores: List[float] = self._bm25.get_scores(tokens).tolist()

        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(c, s) for c, s in ranked[:top_k] if s > 0.0]

    def persist(self) -> None:
        """Write corpus to disk as JSON."""
        if not self._persist_path:
            return
        data: List[Dict[str, Any]] = [
            {
                "id": c.id,
                "content": c.content,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log.debug("BM25 index persisted", path=str(path), docs=len(data))

    def load(self) -> None:
        """Load corpus from disk and rebuild BM25 model."""
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            data: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
            chunks = [
                DocumentChunk(
                    id=d["id"],
                    content=d["content"],
                    source=d["source"],
                    chunk_index=d["chunk_index"],
                    token_count=d["token_count"],
                    metadata=d.get("metadata", {}),
                )
                for d in data
            ]
            if chunks:
                self.add(chunks)
                log.info("BM25 index loaded", path=str(path), docs=len(chunks))
        except Exception as exc:
            log.warning("BM25 index load failed", error=str(exc))

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to ``doc_id`` and rebuild the BM25 model.
        Returns the number of chunks removed.
        Automatically re-persists if a persist_path is configured.
        """
        from rank_bm25 import BM25Plus

        before = len(self._chunks)
        self._chunks = [
            c for c in self._chunks
            if c.metadata.get("doc_id") != doc_id
        ]
        deleted = before - len(self._chunks)

        if deleted == 0:
            return 0

        self._corpus = [c.content.lower().split() for c in self._chunks]
        self._bm25 = BM25Plus(self._corpus) if self._chunks else None

        if self._persist_path:
            self.persist()

        log.info("BM25 chunks deleted", doc_id=doc_id, deleted=deleted, remaining=len(self._chunks))
        return deleted

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_docs(self) -> int:
        return len(self._chunks)
