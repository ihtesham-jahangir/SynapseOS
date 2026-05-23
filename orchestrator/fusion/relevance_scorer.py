"""
Relevance scoring for context fusion candidates.

Composite score formula:

  score = α * semantic_sim      (0.45 weight)
        + β * recency_score     (0.25 weight)
        + γ * source_priority   (0.20 weight)
        + δ * importance_score  (0.10 weight)

Source priority (γ factor):
  L1 (hot conversation)  = 1.0  – immediate context, highest weight
  L4 (knowledge base)    = 0.9  – curated facts, always relevant
  RAG                    = 0.8  – external document context
  L2 (summaries)         = 0.7  – compressed history
  L3 (vector memory)     = 0.6  – semantically similar older context
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from orchestrator.core.types import ContextItem, DocumentChunk, MemoryItem, MemoryLevel
from orchestrator.config.settings import get_settings

# Source priority weights
_SOURCE_PRIORITY = {
    MemoryLevel.L1: 1.0,
    MemoryLevel.L4: 0.9,
    "rag": 0.8,
    MemoryLevel.L2: 0.7,
    MemoryLevel.L3: 0.6,
}


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-10
    return float(np.dot(va, vb) / denom)


def _recency_decay(recency_score: float, half_life: float = 0.5) -> float:
    """Apply exponential decay transform to make differences more pronounced."""
    return 1.0 - math.exp(-recency_score / half_life)


class CompositeScorer:
    """
    Scores each candidate context item against the query embedding.

    For items without stored embeddings (e.g. L1 turns), the
    relevance_score already provided by the memory tier is used directly.
    """

    def __init__(self) -> None:
        cfg = get_settings().fusion
        self._α = cfg.fusion_semantic_weight
        self._β = cfg.fusion_recency_weight
        self._γ = cfg.fusion_priority_weight
        self._δ = cfg.fusion_importance_weight

    def score_memory_item(
        self,
        item: MemoryItem,
        query_embedding: Optional[List[float]] = None,
    ) -> float:
        semantic = (
            _cosine_sim(query_embedding, item.metadata.get("embedding", []))
            if query_embedding and item.metadata.get("embedding")
            else item.relevance_score
        )
        recency = _recency_decay(item.recency_score)
        priority = _SOURCE_PRIORITY.get(item.level, 0.5)
        importance = item.importance_score

        return (
            self._α * semantic
            + self._β * recency
            + self._γ * priority
            + self._δ * importance
        )

    def score_rag_chunk(
        self,
        chunk: DocumentChunk,
        retrieval_score: float,
        query_embedding: Optional[List[float]] = None,
    ) -> float:
        semantic = retrieval_score  # already cosine similarity from FAISS
        recency = 0.8  # Documents don't have recency by default
        priority = _SOURCE_PRIORITY["rag"]
        importance = 0.7  # Documents are moderately important by default

        return (
            self._α * semantic
            + self._β * recency
            + self._γ * priority
            + self._δ * importance
        )
