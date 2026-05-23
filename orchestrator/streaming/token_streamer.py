"""
Token streamer – wraps the async generator from InferenceEngine and
adds metadata injection, cancellation support, and error recovery.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Callable, Optional

from orchestrator.core.types import StreamEvent, StreamEventType
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class TokenStreamer:
    """
    Manages a single response stream:
      - Forwards tokens from inference engine to caller
      - Tracks TTFT (time to first token)
      - Counts tokens generated
      - Handles cancellation and errors gracefully
    """

    def __init__(
        self,
        session_id: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._session_id = session_id
        self._on_token = on_token
        self._tokens_generated = 0
        self._ttft_ms: Optional[float] = None
        self._total_ms: float = 0.0
        self._start = time.perf_counter()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    async def stream(
        self,
        token_generator: AsyncGenerator[str, None],
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Consume raw token strings from the inference engine and
        yield typed StreamEvent objects.
        """
        try:
            async for token in token_generator:
                if self._cancelled:
                    log.debug("Stream cancelled", session=self._session_id)
                    break

                if self._ttft_ms is None:
                    self._ttft_ms = (time.perf_counter() - self._start) * 1000

                self._tokens_generated += 1
                if self._on_token:
                    self._on_token(token)

                yield StreamEvent(
                    event_type=StreamEventType.TOKEN,
                    session_id=self._session_id,
                    content=token,
                    done=False,
                )

            self._total_ms = (time.perf_counter() - self._start) * 1000

            yield StreamEvent(
                event_type=StreamEventType.DONE,
                session_id=self._session_id,
                content="",
                done=True,
                metadata={
                    "tokens_generated": self._tokens_generated,
                    "ttft_ms": self._ttft_ms or 0,
                    "total_ms": self._total_ms,
                    "cancelled": self._cancelled,
                },
            )

        except asyncio.CancelledError:
            self._cancelled = True
            yield StreamEvent(
                event_type=StreamEventType.DONE,
                session_id=self._session_id,
                done=True,
                metadata={"cancelled": True},
            )

        except Exception as exc:
            log.error("Stream error", session=self._session_id, error=str(exc))
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                session_id=self._session_id,
                content=str(exc),
                done=True,
            )

    @property
    def ttft_ms(self) -> float:
        return self._ttft_ms or 0.0

    @property
    def total_ms(self) -> float:
        return self._total_ms

    @property
    def tokens_generated(self) -> int:
        return self._tokens_generated
