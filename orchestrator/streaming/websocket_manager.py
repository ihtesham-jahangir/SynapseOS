"""
WebSocket connection manager.

Tracks active connections per session, handles graceful disconnection,
and provides a broadcast mechanism for multi-client sessions.
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from orchestrator.core.types import StreamEvent, StreamEventType
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.metrics import ACTIVE_SESSIONS

log = get_logger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections.

    One session can have multiple connected WebSocket clients
    (e.g. browser tab + mobile app watching the same session).
    """

    def __init__(self) -> None:
        # session_id → set of active WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = set()
                ACTIVE_SESSIONS.inc()
            self._connections[session_id].add(websocket)
        log.debug("WebSocket connected", session=session_id)

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        async with self._lock:
            if session_id in self._connections:
                self._connections[session_id].discard(websocket)
                if not self._connections[session_id]:
                    del self._connections[session_id]
                    ACTIVE_SESSIONS.dec()
        log.debug("WebSocket disconnected", session=session_id)

    async def send_event(self, session_id: str, event: StreamEvent) -> None:
        """Send a StreamEvent to all clients in a session."""
        sockets = list(self._connections.get(session_id, set()))
        if not sockets:
            return

        payload = event.model_dump_json()
        dead: List[WebSocket] = []

        for ws in sockets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.get(session_id, set()).discard(ws)

    async def send_token(self, session_id: str, token: str) -> None:
        event = StreamEvent(
            event_type=StreamEventType.TOKEN,
            session_id=session_id,
            content=token,
            done=False,
        )
        await self.send_event(session_id, event)

    async def send_done(self, session_id: str, metadata: dict) -> None:
        event = StreamEvent(
            event_type=StreamEventType.DONE,
            session_id=session_id,
            done=True,
            metadata=metadata,
        )
        await self.send_event(session_id, event)

    async def send_error(self, session_id: str, error: str) -> None:
        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            session_id=session_id,
            content=error,
            done=True,
        )
        await self.send_event(session_id, event)

    def active_session_count(self) -> int:
        return len(self._connections)

    def is_connected(self, session_id: str) -> bool:
        return bool(self._connections.get(session_id))
