"""
Tests for TokenStreamer – token event generation, TTFT tracking, cancellation.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest

from orchestrator.core.types import StreamEventType
from orchestrator.streaming.token_streamer import TokenStreamer


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _token_gen(*tokens: str) -> AsyncGenerator[str, None]:
    for t in tokens:
        yield t


async def _raising_gen(exc: Exception) -> AsyncGenerator[str, None]:
    yield "first"
    raise exc


# ── TokenStreamer ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTokenStreamer:
    async def test_token_events_emitted(self):
        streamer = TokenStreamer(session_id="s1")
        events = [e async for e in streamer.stream(_token_gen("Hello", " World"))]

        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        assert len(token_events) == 2
        assert token_events[0].content == "Hello"
        assert token_events[1].content == " World"

    async def test_done_event_last(self):
        streamer = TokenStreamer(session_id="s1")
        events = [e async for e in streamer.stream(_token_gen("a", "b", "c"))]

        assert events[-1].event_type == StreamEventType.DONE
        assert events[-1].done is True

    async def test_done_metadata_contains_counts(self):
        streamer = TokenStreamer(session_id="s1")
        events = [e async for e in streamer.stream(_token_gen("tok1", "tok2"))]

        done = events[-1]
        assert done.metadata["tokens_generated"] == 2
        assert done.metadata["ttft_ms"] >= 0
        assert done.metadata["total_ms"] >= 0

    async def test_empty_generator_still_yields_done(self):
        streamer = TokenStreamer(session_id="s1")

        async def empty():
            return
            yield  # make it an async generator

        events = [e async for e in streamer.stream(empty())]
        assert len(events) == 1
        assert events[0].event_type == StreamEventType.DONE

    async def test_cancellation_stops_stream(self):
        streamer = TokenStreamer(session_id="s1")
        streamer.cancel()

        events = [e async for e in streamer.stream(_token_gen("a", "b", "c", "d"))]
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        # After cancel() the first token check stops iteration
        assert len(token_events) == 0

    async def test_error_in_generator_yields_error_event(self):
        streamer = TokenStreamer(session_id="s1")
        events = [
            e async for e in streamer.stream(_raising_gen(ValueError("boom")))
        ]
        # First event is the yielded token, last is the error
        error_events = [e for e in events if e.event_type == StreamEventType.ERROR]
        assert len(error_events) == 1
        assert "boom" in error_events[0].content

    async def test_ttft_measured_on_first_token(self):
        streamer = TokenStreamer(session_id="s1")
        events = [e async for e in streamer.stream(_token_gen("first", "second"))]
        done = events[-1]
        # ttft_ms should be small but non-negative
        assert done.metadata["ttft_ms"] >= 0

    async def test_token_counter_increments(self):
        streamer = TokenStreamer(session_id="s1")
        tokens = ["a", "b", "c", "d", "e"]
        events = [e async for e in streamer.stream(_token_gen(*tokens))]
        assert streamer.tokens_generated == len(tokens)

    async def test_on_token_callback_called(self):
        seen = []
        streamer = TokenStreamer(session_id="s1", on_token=seen.append)
        await asyncio.gather(
            *([] for _ in []),
        )
        events = [e async for e in streamer.stream(_token_gen("x", "y", "z"))]
        assert seen == ["x", "y", "z"]

    async def test_session_id_set_in_events(self):
        streamer = TokenStreamer(session_id="my-session")
        events = [e async for e in streamer.stream(_token_gen("hi"))]
        for e in events:
            assert e.session_id == "my-session"
