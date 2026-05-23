"""
RAG rerankers — two interchangeable implementations with the same interface:

  EmbeddingReranker    — bi-encoder cosine similarity + MMR diversity (default)
  CrossEncoderReranker — ms-marco-MiniLM cross-encoder, more accurate (~5 ms/pair CPU)

Both implement async ``rerank(query, chunks, top_n)`` returning
``List[Tuple[DocumentChunk, float]]``.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import numpy as np

from orchestrator.core.types import DocumentChunk
from orchestrator.utils.logging_utils import get_logger
from .embedder import BGEEmbedder

log = get_logger(__name__)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-10
    return float(np.dot(va, vb) / denom)


class EmbeddingReranker:
    """
    Reranks retrieved chunks by re-computing cosine similarity between
    the query embedding and chunk embeddings.

    This catches ranking errors from ANN approximate search and also
    applies a maximal marginal relevance (MMR) penalty to reduce redundancy.
    """

    def __init__(self, embedder: BGEEmbedder, diversity_lambda: float = 0.5) -> None:
        self._embedder = embedder
        self._lambda = diversity_lambda  # MMR trade-off: 1.0 = pure relevance

    async def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_n: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Apply Maximal Marginal Relevance reranking.

        MMR score = λ * sim(query, doc) - (1-λ) * max_sim(doc, selected)
        """
        if not chunks:
            return []

        top_n = min(top_n, len(chunks))

        # Embed query and all chunks
        texts = [f"{self._embedder._settings.query_instruction}{query}"] + [
            c.content for c in chunks
        ]
        embeddings = await self._embedder.embed(texts)
        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        # Relevance scores
        relevance = [_cosine_sim(query_emb, emb) for emb in chunk_embs]

        selected_indices: List[int] = []
        selected_embs: List[List[float]] = []
        candidates = list(range(len(chunks)))

        while len(selected_indices) < top_n and candidates:
            mmr_scores: List[Tuple[int, float]] = []
            for i in candidates:
                rel = relevance[i]
                if selected_embs:
                    max_redundancy = max(
                        _cosine_sim(chunk_embs[i], sel_emb) for sel_emb in selected_embs
                    )
                else:
                    max_redundancy = 0.0
                score = self._lambda * rel - (1 - self._lambda) * max_redundancy
                mmr_scores.append((i, score))

            best_i, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_i)
            selected_embs.append(chunk_embs[best_i])
            candidates.remove(best_i)

        return [
            (chunks[i], relevance[i]) for i in selected_indices
        ]


class CrossEncoderReranker:
    """
    True cross-encoder reranker using sentence-transformers CrossEncoder.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - 22 M parameters, ~5 ms per pair on CPU
      - Downloaded once (~85 MB) on first use

    Unlike the bi-encoder (EmbeddingReranker), the cross-encoder reads
    the query and document *jointly*, producing a much stronger relevance
    signal — especially for longer documents with indirect relevance.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            log.info("Loading cross-encoder model", model=self._model_name)
            from sentence_transformers import CrossEncoder

            def _load():
                return CrossEncoder(self._model_name)

            self._model = await asyncio.to_thread(_load)
            log.info("Cross-encoder loaded", model=self._model_name)

    async def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_n: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Score each (query, chunk) pair; return top_n by descending score."""
        if not chunks:
            return []
        top_n = min(top_n, len(chunks))

        await self._ensure_loaded()

        pairs = [[query, c.content] for c in chunks]
        raw_scores: np.ndarray = await asyncio.to_thread(self._model.predict, pairs)
        scores: List[float] = raw_scores.tolist()

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        log.debug(
            "Cross-encoder rerank complete",
            query_preview=query[:60],
            input_chunks=len(chunks),
            returned=top_n,
        )
        return list(ranked[:top_n])
