"""
Semantic and hybrid retrieval: query → embed → FAISS search → return chunks.

Two retrievers are provided:
  SemanticRetriever  — pure FAISS vector search (default)
  HybridRetriever    — BM25 + FAISS merged with Reciprocal Rank Fusion
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from orchestrator.core.types import DocumentChunk, RAGResult
from orchestrator.core.exceptions import RAGError
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.token_counter import count_tokens

from .embedder import BGEEmbedder
from .indexer import FAISSIndex

if TYPE_CHECKING:
    from .bm25_index import BM25Index

log = get_logger(__name__)


class SemanticRetriever:
    """
    Retrieves document chunks semantically relevant to a query.

    Retrieval flow:
      1. Embed query with BGE instruction prefix
      2. ANN search in FAISS index (top_k * 2 candidates)
      3. Score filter by similarity threshold
      4. Return DocumentChunk list with scores
    """

    def __init__(self, embedder: BGEEmbedder, index: FAISSIndex) -> None:
        self._embedder = embedder
        self._index = index
        self._settings = get_settings().rag

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: float = 0.3,
    ) -> RAGResult:
        t0 = time.perf_counter()
        top_k = top_k or self._settings.rag_top_k

        try:
            query_vec = await self._embedder.embed_query(query)
        except Exception as exc:
            raise RAGError(f"Query embedding failed: {exc}") from exc

        # Over-retrieve then filter
        candidates = await self._index.search(
            query_vector=query_vec,
            top_k=top_k * 2,
            threshold=threshold,
        )

        chunks: List[DocumentChunk] = []
        scores: List[float] = []
        total_tokens = 0

        for _idx, score, meta in candidates[:top_k]:
            content = meta.get("content", "")
            tok = count_tokens(content)
            chunks.append(
                DocumentChunk(
                    id=meta.get("chunk_id", str(_idx)),
                    content=content,
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", 0),
                    token_count=tok,
                    metadata=meta,
                )
            )
            scores.append(score)
            total_tokens += tok

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "Retrieval complete",
            query_preview=query[:60],
            candidates=len(candidates),
            returned=len(chunks),
            elapsed_ms=f"{elapsed_ms:.1f}",
        )

        return RAGResult(
            chunks=chunks,
            scores=scores,
            total_tokens=total_tokens,
            query_time_ms=elapsed_ms,
        )


class HybridRetriever:
    """
    Combines BM25 keyword search and FAISS semantic search via
    Reciprocal Rank Fusion (RRF).

    RRF score = alpha * 1/(k + sem_rank) + (1-alpha) * 1/(k + bm25_rank)

    ``alpha=1.0``  → pure semantic
    ``alpha=0.0``  → pure keyword
    ``alpha=0.6``  → 60% semantic, 40% keyword (default)
    """

    _RRF_K = 60  # standard RRF smoothing constant

    def __init__(
        self,
        semantic: SemanticRetriever,
        bm25: "BM25Index",
        alpha: float = 0.6,
    ) -> None:
        self._semantic = semantic
        self._bm25 = bm25
        self._alpha = alpha

    async def retrieve(
        self,
        query: str,
        top_k: int = 6,
        threshold: float = 0.3,
    ) -> RAGResult:
        t0 = time.perf_counter()
        candidate_k = top_k * 3  # over-fetch for better fusion coverage

        # Run semantic and keyword search concurrently
        sem_task = asyncio.create_task(
            self._semantic.retrieve(query, top_k=candidate_k, threshold=threshold)
        )
        bm25_pairs: List[Tuple[DocumentChunk, float]] = await asyncio.to_thread(
            self._bm25.search, query, candidate_k
        )
        sem_result = await sem_task

        merged = self._rrf_merge(
            sem_list=sem_result.chunks,
            bm25_list=[c for c, _ in bm25_pairs],
        )
        final = merged[:top_k]
        total_tokens = sum(c.token_count for c in final)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        log.debug(
            "Hybrid retrieval complete",
            query_preview=query[:60],
            semantic=len(sem_result.chunks),
            bm25=len(bm25_pairs),
            merged=len(final),
            elapsed_ms=f"{elapsed_ms:.1f}",
        )

        return RAGResult(
            chunks=final,
            scores=[1.0 / (i + 1) for i in range(len(final))],  # RRF rank as proxy score
            total_tokens=total_tokens,
            query_time_ms=elapsed_ms,
        )

    def _rrf_merge(
        self,
        sem_list: List[DocumentChunk],
        bm25_list: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        for rank, chunk in enumerate(sem_list):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + self._alpha / (self._RRF_K + rank + 1)
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(bm25_list):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + (1.0 - self._alpha) / (self._RRF_K + rank + 1)
            chunk_map[chunk.id] = chunk

        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids]
