"""
Response-level LRU cache with TTL expiration + semantic similarity layer.

Two-tier lookup:
  1. Exact hash match  — O(1), sub-millisecond
  2. Semantic match    — cosine similarity via BGE embeddings (optional)
     activated when an embedder callable is injected at construction time.

Safety rules — a query is NOT cached when:
  • Intent is session-dependent (conversation, retrieval)
  • Query contains personal pronouns (my, I, me, our, you)
  • Query length is under 8 characters
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import string
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Set

import numpy as np

from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Cacheable intents ─────────────────────────────────────────────────────────
# All intents that produce deterministic, context-free answers.
# "conversation" and "retrieval" are excluded — answers depend on session state.
_CACHEABLE_INTENTS: Set[str] = {
    # original
    "math", "coding", "translation", "summarization", "reasoning",
    # v3.5 new domains
    "writing", "security", "planning", "education", "creative", "data_analysis",
}

# Personal-pronoun pattern — if matched, skip caching (answer is user-specific)
_PERSONAL_RE = re.compile(
    r"\b(my|me|i am|i'm|i've|i'll|our|your|his|her|their|we are|we're)\b",
    re.IGNORECASE,
)

_MIN_QUERY_LEN = 8
_SEM_THRESHOLD = 0.90   # cosine similarity floor for a semantic cache hit


@dataclass
class CachedEntry:
    content: str
    intent_type: str
    tokens_generated: int
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


# Type alias for the optional async embedder callable
EmbedFn = Callable[[str], Awaitable[List[float]]]


class ResponseCache:
    """
    Async-safe LRU response cache with optional semantic similarity layer.

    Pass ``embedder=some_async_fn`` (e.g. ``BGEEmbedder().embed_query``) to
    enable semantic matching on cache misses.  Entries expire after
    ``ttl_seconds`` and are evicted LRU-first when the cache exceeds
    ``maxsize``.
    """

    def __init__(
        self,
        maxsize: int = 512,
        ttl_seconds: int = 3600,
        embedder: Optional[EmbedFn] = None,
        sem_threshold: float = _SEM_THRESHOLD,
    ) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, CachedEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._total_hits = 0
        self._total_misses = 0
        self._sem_hits = 0
        # Semantic layer (only active when embedder is provided)
        self._embedder: Optional[EmbedFn] = embedder
        self._sem_threshold = sem_threshold
        self._embeddings: Dict[str, np.ndarray] = {}   # cache-key → unit vector

    # ── public API ─────────────────────────────────────────────────────────────

    def should_cache(self, query: str, intent_type: str) -> bool:
        """Return True only when this query/intent pair is safe to cache."""
        if intent_type not in _CACHEABLE_INTENTS:
            return False
        if len(query.strip()) < _MIN_QUERY_LEN:
            return False
        if _PERSONAL_RE.search(query):
            return False
        return True

    async def get(self, query: str) -> Optional[CachedEntry]:
        # ── Tier 1: exact hash lookup (O(1)) ──────────────────────────────────
        key = self._key(query)
        async with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                if time.time() - entry.created_at > self._ttl:
                    del self._cache[key]
                    self._embeddings.pop(key, None)
                else:
                    entry.hit_count += 1
                    self._total_hits += 1
                    self._cache.move_to_end(key)
                    log.info(
                        "Cache hit (exact)",
                        query_preview=query[:50],
                        hits=entry.hit_count,
                        intent=entry.intent_type,
                    )
                    return entry
            self._total_misses += 1

        # ── Tier 2: semantic similarity lookup ────────────────────────────────
        if self._embedder is not None:
            sem_entry = await self._semantic_get(query)
            if sem_entry is not None:
                return sem_entry

        return None

    async def set(
        self,
        query: str,
        content: str,
        intent_type: str,
        tokens_generated: int,
    ) -> None:
        key = self._key(query)
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            self._cache[key] = CachedEntry(
                content=content,
                intent_type=intent_type,
                tokens_generated=tokens_generated,
            )
            self._evict()

        # Store embedding for semantic tier — outside lock, non-blocking
        if self._embedder is not None:
            try:
                raw = await self._embedder(query)
                vec = np.array(raw, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    async with self._lock:
                        self._embeddings[key] = vec / norm
            except Exception as exc:
                log.debug("Semantic cache embed failed", error=str(exc))

    async def invalidate(self, query: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        key = self._key(query)
        async with self._lock:
            self._embeddings.pop(key, None)
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._embeddings.clear()
            self._total_hits = 0
            self._total_misses = 0
            self._sem_hits = 0

    def stats(self) -> Dict:
        total = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total if total else 0.0
        return {
            "entries": len(self._cache),
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "semantic_hits": self._sem_hits,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self._ttl,
            "maxsize": self._maxsize,
            "semantic_enabled": self._embedder is not None,
            "embeddings_stored": len(self._embeddings),
        }

    # ── internals ──────────────────────────────────────────────────────────────

    async def _semantic_get(self, query: str) -> Optional[CachedEntry]:
        """Find the closest cached response by cosine similarity."""
        try:
            raw = await self._embedder(query)  # type: ignore[misc]
            q_vec = np.array(raw, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm == 0:
                return None
            q_vec /= norm

            best_key: Optional[str] = None
            best_sim = self._sem_threshold  # only accept hits above threshold

            async with self._lock:
                for k, emb in self._embeddings.items():
                    sim = float(np.dot(q_vec, emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_key = k

                if best_key and best_key in self._cache:
                    entry = self._cache[best_key]
                    if time.time() - entry.created_at <= self._ttl:
                        entry.hit_count += 1
                        self._total_hits += 1
                        self._sem_hits += 1
                        self._cache.move_to_end(best_key)
                        log.info(
                            "Cache hit (semantic)",
                            query_preview=query[:50],
                            similarity=round(best_sim, 3),
                            intent=entry.intent_type,
                        )
                        return entry
        except Exception as exc:
            log.debug("Semantic cache lookup failed", error=str(exc))
        return None

    # Punctuation table for key normalization
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    @staticmethod
    def _key(query: str) -> str:
        normalized = query.strip().lower().translate(ResponseCache._PUNCT_TABLE)
        normalized = " ".join(normalized.split())
        return hashlib.blake2b(normalized.encode(), digest_size=16).hexdigest()

    def _evict(self) -> None:
        while len(self._cache) > self._maxsize:
            oldest_key, _ = self._cache.popitem(last=False)
            self._embeddings.pop(oldest_key, None)
            log.debug("Response cache evicted", key=oldest_key[:8])
