"""
Full RAG pipeline: ingest documents + retrieve context.

Accepts injected chunker / retriever / reranker so the caller
can swap in SemanticChunker, HybridRetriever, or CrossEncoderReranker
without any code changes inside this file.

Defaults (when nothing is injected):
  chunker   → RecursiveTextChunker
  retriever → SemanticRetriever
  reranker  → EmbeddingReranker
"""
from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from typing import Any, Dict, List, Optional

from orchestrator.core.base import BaseRAGPipeline
from orchestrator.core.types import DocumentChunk, RAGResult
from orchestrator.core.exceptions import RAGError
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.metrics import RAG_RETRIEVED

from .chunker import RecursiveTextChunker
from .embedder import BGEEmbedder
from .indexer import FAISSIndex
from .retriever import SemanticRetriever
from .reranker import EmbeddingReranker

log = get_logger(__name__)


class RAGPipeline(BaseRAGPipeline):
    """
    Orchestrates the full RAG loop:

    Ingest path:
        text → chunk → embed → FAISS add → (BM25 add) → persist

    Retrieval path:
        query → retriever (semantic or hybrid) → reranker → RAGResult

    All major components are dependency-injected for easy swapping.
    """

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        index: Optional[FAISSIndex] = None,
        chunker: Optional[Any] = None,          # RecursiveTextChunker | SemanticChunker
        retriever: Optional[Any] = None,         # SemanticRetriever | HybridRetriever
        reranker: Optional[Any] = None,          # EmbeddingReranker | CrossEncoderReranker
        bm25_index: Optional[Any] = None,        # BM25Index — updated on ingest when set
    ) -> None:
        cfg = get_settings()
        self._cfg = cfg.rag
        self._embedder = embedder or BGEEmbedder()
        self._index = index or FAISSIndex(
            dimension=384,
            index_path=self._cfg.faiss_index_path,
        )
        self._chunker = chunker or RecursiveTextChunker(
            chunk_size=self._cfg.chunk_size,
            chunk_overlap=self._cfg.chunk_overlap,
        )
        self._retriever = retriever or SemanticRetriever(self._embedder, self._index)
        self._reranker = reranker or EmbeddingReranker(self._embedder)
        self._bm25 = bm25_index   # None → hybrid search disabled

    async def ingest(
        self,
        content: str,
        source: str = "user_upload",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Chunk, embed, and index a document.
        Returns number of chunks created.
        """
        t0 = time.perf_counter()
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", str(uuid.uuid4()))

        # SemanticChunker.chunk() is a coroutine; RecursiveTextChunker.chunk() is sync
        if inspect.iscoroutinefunction(self._chunker.chunk):
            chunks = await self._chunker.chunk(content)
        else:
            chunks = self._chunker.chunk(content)

        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        try:
            embeddings = await self._embedder.embed(texts)
        except Exception as exc:
            raise RAGError(f"Embedding during ingest failed: {exc}") from exc

        meta_list = [
            {
                "chunk_id": f"{doc_id}_{c.chunk_index}",
                "doc_id": doc_id,
                "content": c.content,
                "source": source,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                **metadata,
            }
            for c in chunks
        ]

        await self._index.add(embeddings, meta_list)
        await self._index.persist()

        # Update BM25 index in parallel when hybrid search is enabled
        if self._bm25 is not None:
            doc_chunks = [
                DocumentChunk(
                    id=m["chunk_id"],
                    content=c.content,
                    source=source,
                    chunk_index=c.chunk_index,
                    token_count=c.token_count,
                    metadata=m,
                )
                for c, m in zip(chunks, meta_list)
            ]
            await asyncio.to_thread(self._bm25.add, doc_chunks)
            await asyncio.to_thread(self._bm25.persist)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Document ingested",
            doc_id=doc_id,
            source=source,
            chunks=len(chunks),
            elapsed_ms=f"{elapsed_ms:.0f}",
        )
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RAGResult:
        """Retrieve and rerank the most relevant chunks for a query."""
        top_k = top_k or self._cfg.rag_top_k
        rerank_n = self._cfg.rag_rerank_top_n

        raw_result = await self._retriever.retrieve(query, top_k=top_k)

        if not raw_result.chunks:
            return raw_result

        reranked = await self._reranker.rerank(
            query=query,
            chunks=raw_result.chunks,
            top_n=rerank_n,
        )

        final_chunks = [chunk for chunk, _ in reranked]
        final_scores = [score for _, score in reranked]
        total_tokens = sum(c.token_count for c in final_chunks)

        RAG_RETRIEVED.inc(len(final_chunks))

        return RAGResult(
            chunks=final_chunks,
            scores=final_scores,
            total_tokens=total_tokens,
            query_time_ms=raw_result.query_time_ms,
        )

    async def delete_document(self, doc_id: str) -> dict:
        """
        Remove a document and all its chunks from every index.

        Workflow:
          1. Soft-mark chunks as deleted in FAISS metadata
          2. Rebuild FAISS index without those entries (uses reconstruct())
          3. Persist the rebuilt index to disk
          4. Remove from BM25 index (if hybrid search is enabled)

        Returns a dict with ``chunks_deleted`` and ``remaining_vectors``.
        Raises ``ValueError`` if the document does not exist.
        """
        deleted = await self._index.delete_by_doc_id(doc_id)
        if deleted == 0:
            raise ValueError(f"Document '{doc_id}' not found in index")

        remaining = await self._index.rebuild_without_deleted()
        await self._index.persist()

        bm25_deleted = 0
        if self._bm25 is not None:
            bm25_deleted = await asyncio.to_thread(self._bm25.delete_by_doc_id, doc_id)

        log.info(
            "Document deleted",
            doc_id=doc_id,
            faiss_deleted=deleted,
            bm25_deleted=bm25_deleted,
            remaining=remaining,
        )
        return {
            "doc_id": doc_id,
            "chunks_deleted": deleted,
            "remaining_vectors": remaining,
        }

    @property
    def index_size(self) -> int:
        return self._index.total_vectors
