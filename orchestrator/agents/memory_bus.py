"""
SharedMemoryBus — in-process async pub/sub for multi-agent coordination.

Agents publish results to named topics; other agents can subscribe to
receive future messages or poll the history of past messages.
Topics auto-create on first publish.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from orchestrator.agents.task_types import BusMessage
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.metrics import BUS_MESSAGES_TOTAL

log = get_logger(__name__)

_MAX_HISTORY = 64  # messages retained per topic


class SharedMemoryBus:
    """
    Async pub/sub message bus.

    publish()     — broadcast a message to a topic
    subscribe()   — get a Queue that receives future messages on a topic
    wait_for()    — block until the next message arrives (or timeout)
    get_history() — return all retained messages for a topic
    get_latest()  — return the most recent message on a topic
    """

    def __init__(self) -> None:
        self._history: Dict[str, Deque[BusMessage]] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTORY)
        )
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(
        self,
        topic: str,
        sender_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        msg = BusMessage(
            topic=topic,
            sender_id=sender_id,
            content=content,
            metadata=metadata or {},
        )
        async with self._lock:
            self._history[topic].append(msg)
            subs = list(self._subscribers[topic])

        for q in subs:
            await q.put(msg)

        BUS_MESSAGES_TOTAL.labels(topic=topic.split(".")[0]).inc()
        log.debug("Bus message published", topic=topic, sender=sender_id)

    def subscribe(self, topic: str) -> asyncio.Queue:
        """Return a queue that receives all future messages on *topic*."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[topic].append(q)
        return q

    def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(topic, [])
        try:
            subs.remove(queue)
        except ValueError:
            pass

    def get_history(self, topic: str) -> List[BusMessage]:
        """Return all retained messages for a topic (oldest first)."""
        return list(self._history.get(topic, []))

    def get_latest(self, topic: str) -> Optional[BusMessage]:
        """Return the most recent message on a topic, or None."""
        hist = self._history.get(topic)
        if hist:
            return hist[-1]
        return None

    async def wait_for(self, topic: str, timeout: float = 30.0) -> Optional[BusMessage]:
        """Block until the next message arrives on *topic* (or timeout → None)."""
        q = self.subscribe(topic)
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(topic, q)

    def clear(self) -> None:
        """Remove all history and subscribers (used in tests)."""
        self._history.clear()
        self._subscribers.clear()

    @property
    def topic_count(self) -> int:
        return len(self._history)
