"""
L1 Cache – Hot conversation buffer.

In-memory deque per session. O(1) push/pop. Token-aware FIFO eviction.
This is the fastest path: zero I/O, used for every turn.
"""
from __future__ import annotations

import asyncio
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
    the token budget is exceeded.
    """

    def __init__(self, max_turns: int = 20, max_tokens: int = 2048) -> None:
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._sessions: Dict[str, Deque[Message]] = defaultdict(deque)
        self._token_counts: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def push_turn(self, session_id: str, message: Message) -> None:
        """Append a message and evict oldest if budget exceeded."""
        async with self._lock:
            buf = self._sessions[session_id]
            tok = count_tokens(message.content)

            buf.append(message)
            self._token_counts[session_id] += tok

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
            return list(self._sessions.get(session_id, deque()))

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

    def session_token_count(self, session_id: str) -> int:
        return self._token_counts.get(session_id, 0)

    def active_sessions(self) -> List[str]:
        return list(self._sessions.keys())
