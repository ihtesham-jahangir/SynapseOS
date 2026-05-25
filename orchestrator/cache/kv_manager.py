"""
KV Cache Manager.

Provides a prompt-level caching layer that avoids re-encoding identical
system prompts + context across turns in the same session.

llama.cpp maintains KV cache internally. This layer tracks which
prompt prefix has been cached server-side, enabling prefix reuse.

Cache hierarchy:
  1. Exact-match prefix cache (for identical system prompts + context)
  2. Longest-common-prefix detection for partial reuse
  3. LRU eviction when cache fills
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class CacheEntry:
    prompt_hash: str
    token_count: int
    hit_count: int = 0
    last_hit: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    importance_score: float = 0.5


class KVCacheManager:
    """
    Tracks prompt-level cache entries to advise the llama.cpp server
    on prefix reuse.

    When the same system prompt + context prefix appears across multiple
    turns in a session, the server can reuse its KV state without
    recomputing attention from scratch.
    """

    def __init__(self, max_entries: int = 128, max_tokens_tracked: int = 50_000) -> None:
        self._max_entries = max_entries
        self._max_tokens = max_tokens_tracked
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_tokens = 0

    def get_prefix_hash(self, messages: List[Dict]) -> str:
        """Hash all messages except the last user turn (the prefix to cache)."""
        prefix_messages = messages[:-1] if len(messages) > 1 else messages
        content = str(prefix_messages)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_cached(self, prefix_hash: str) -> bool:
        return prefix_hash in self._cache

    def record_hit(self, prefix_hash: str) -> Optional[CacheEntry]:
        if prefix_hash not in self._cache:
            return None
        entry = self._cache[prefix_hash]
        entry.hit_count += 1
        entry.last_hit = time.time()
        self._cache.move_to_end(prefix_hash)
        log.debug("KV cache hit", hash=prefix_hash[:8], hits=entry.hit_count)
        return entry

    def register(self, prefix_hash: str, messages: List[Dict]) -> CacheEntry:
        """Register a new prefix as cached."""
        if prefix_hash in self._cache:
            return self._cache[prefix_hash]

        # Calculate token count
        prefix = messages[:-1] if len(messages) > 1 else messages
        tok = sum(count_tokens(m.get("content", "")) for m in prefix)

        entry = CacheEntry(
            prompt_hash=prefix_hash,
            token_count=tok,
        )

        self._cache[prefix_hash] = entry
        self._total_tokens += tok
        self._evict_if_needed()
        return entry

    def _evict_if_needed(self) -> None:
        """LRU eviction: remove oldest entries when over budget."""
        while (
            len(self._cache) > self._max_entries
            or self._total_tokens > self._max_tokens
        ):
            if not self._cache:
                break
            oldest_key, oldest_entry = self._cache.popitem(last=False)
            self._total_tokens -= oldest_entry.token_count
            log.debug("KV cache evicted", hash=oldest_key[:8], tokens=oldest_entry.token_count)

    def get_stats(self) -> Dict:
        total_hits = sum(e.hit_count for e in self._cache.values())
        return {
            "entries": len(self._cache),
            "total_tokens_tracked": self._total_tokens,
            "total_hits": total_hits,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._total_tokens = 0


# Import Dict for type hints
from typing import Dict
