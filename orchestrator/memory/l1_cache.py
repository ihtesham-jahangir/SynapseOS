"""
L1 Cache – Hot conversation buffer.

In-memory deque per session. O(1) push/pop. Token-aware FIFO eviction.
This is the fastest path: zero I/O, used for every turn.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Deque, List, Optional

from orchestrator.core.base import BaseMemoryStore
from orchestrator.core.types import MemoryItem, MemoryLevel, Message, MessageRole
from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class L1ConversationCache(BaseMemoryStore):
    """
    Per-session sliding window of raw conversation turns.

    Eviction policy: token-capped FIFO – oldest turns drop when
    the token budget is exceeded. Sessions that have not been accessed
    within ttl_seconds are evicted by the background cleanup task.
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens: int = 2048,
        ttl_seconds: int = 3600,
    ) -> None:
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._ttl = ttl_seconds
        self._sessions: Dict[str, Deque[Message]] = defaultdict(deque)
        self._token_counts: Dict[str, int] = defaultdict(int)
        self._last_access: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def push_turn(self, session_id: str, message: Message) -> None:
        """Append a message and evict oldest if budget exceeded."""
        async with self._lock:
            buf = self._sessions[session_id]
            tok = count_tokens(message.content)

            buf.append(message)
            self._token_counts[session_id] += tok
            self._last_access[session_id] = time.monotonic()

            # Token cap
            while (
                self._token_counts[session_id] > self._max_tokens
                and len(buf) > 1
            ):
                oldest = buf.popleft()
                self._token_counts[session_id] -= count_tokens(oldest.content)

            # Turn cap
            while len(buf) > self._max_turns:
                oldest = buf.popleft()
                self._token_counts[session_id] -= count_tokens(oldest.content)

    async def get_turns(self, session_id: str) -> List[Message]:
        async with self._lock:
            self._last_access[session_id] = time.monotonic()
            return list(self._sessions.get(session_id, deque()))

    async def evict_expired(self) -> List[str]:
        """Remove sessions that have been idle longer than ttl_seconds. Returns evicted IDs."""
        now = time.monotonic()
        evicted: List[str] = []
        async with self._lock:
            expired = [
                sid for sid, last in self._last_access.items()
                if now - last > self._ttl
            ]
            for sid in expired:
                self._sessions.pop(sid, None)
                self._token_counts.pop(sid, None)
                self._last_access.pop(sid, None)
                evicted.append(sid)
        if evicted:
            log.info("L1 TTL eviction", count=len(evicted))
        return evicted

    async def pop_turns(self, session_id: str, n: int) -> None:
        """Remove the oldest n turns from L1 after they have been compressed into L2."""
        async with self._lock:
            buf = self._sessions.get(session_id)
            if buf is None:
                return
            for _ in range(min(n, len(buf))):
                oldest = buf.popleft()
                self._token_counts[session_id] -= count_tokens(oldest.content)
            if not buf:
                del self._sessions[session_id]
                self._token_counts.pop(session_id, None)

    async def store(self, item: MemoryItem) -> None:
        msg = Message(
            role=MessageRole.USER if "user" in item.metadata.get("role", "user") else MessageRole.ASSISTANT,
            content=item.content,
            timestamp=item.timestamp,
        )
        await self.push_turn(item.session_id or "default", msg)

    async def retrieve(
        self,
        query: str,
        session_id: str,
        limit: int = 10,
    ) -> List[MemoryItem]:
        turns = await self.get_turns(session_id)
        items: List[MemoryItem] = []
        for i, turn in enumerate(turns[-limit:]):
            recency = (i + 1) / max(len(turns), 1)  # 0→1, newest=1
            items.append(
                MemoryItem(
                    content=f"{turn.role.value}: {turn.content}",
                    level=MemoryLevel.L1,
                    session_id=session_id,
                    relevance_score=1.0,
                    recency_score=recency,
                    importance_score=0.9,
                    token_count=count_tokens(turn.content),
                    timestamp=turn.timestamp,
                    metadata={"role": turn.role.value},
                )
            )
        return items

    async def clear_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
            self._token_counts.pop(session_id, None)
            self._last_access.pop(session_id, None)

    def session_token_count(self, session_id: str) -> int:
        return self._token_counts.get(session_id, 0)

    def active_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def session_idle_seconds(self, session_id: str) -> Optional[float]:
        last = self._last_access.get(session_id)
        return (time.monotonic() - last) if last is not None else None
