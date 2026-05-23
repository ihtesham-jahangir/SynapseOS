"""
Semantic deduplication of candidate context items.

Uses cosine similarity between BGE embeddings to identify near-duplicate
content pieces. When two items are too similar, only the higher-scored
one is kept — the higher-scored item wins.

For items without pre-computed embeddings, falls back to character-level
Jaccard similarity.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from orchestrator.core.types import ContextItem
from orchestrator.config.settings import get_settings


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def _jaccard_sim(s1: str, s2: str, ngram: int = 3) -> float:
    """Character n-gram Jaccard similarity (O(n) space, fast)."""
    def ngrams(s: str) -> set:
        s = s.lower()
        return {s[i : i + ngram] for i in range(max(0, len(s) - ngram + 1))}

    g1, g2 = ngrams(s1), ngrams(s2)
    union = g1 | g2
    if not union:
        return 1.0
    return len(g1 & g2) / len(union)


ScoredItem = Tuple[ContextItem, float, Optional[List[float]]]  # (item, score, embedding)


class SemanticDeduplicator:
    """
    Greedy deduplication:
      1. Sort candidates by descending score
      2. For each candidate: if it's too similar to any already-selected item,
         discard it; otherwise keep it.

    Complexity: O(n²) in number of candidates (n ≤ 30 in practice).
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self._threshold = threshold or get_settings().fusion.fusion_dedup_threshold

    def deduplicate(
        self,
        scored_items: List[ScoredItem],
    ) -> List[ScoredItem]:
        """Return deduplicated list, preserving score order."""
        sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        kept: List[ScoredItem] = []

        for candidate in sorted_items:
            item, score, emb = candidate
            is_duplicate = False

            for kept_item, _, kept_emb in kept:
                if emb is not None and kept_emb is not None:
                    sim = _cosine_sim(
                        np.array(emb, dtype=np.float32),
                        np.array(kept_emb, dtype=np.float32),
                    )
                else:
                    sim = _jaccard_sim(item.content, kept_item.content)

                if sim >= self._threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(candidate)

        return kept
