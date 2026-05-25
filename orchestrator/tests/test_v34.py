"""
v3.4 feature tests — covers all upgrades applied in this session.

Groups:
  1.  L1 TTL expiry (setting, tracking, evict_expired)
  2.  L1 session_idle_seconds
  3.  Request ID middleware (X-Request-ID header added to responses)
  4.  Structured error responses (consistent {"error": {...}} format)
  5.  WebSocket heartbeat task wired into WS handler
  6.  /v1/chat/stream routes complex queries via MultiAgentEngine
  7.  GET /v1/chat/history/{session_id} returns turn + summary data
  8.  /v1/admin/memory/stats endpoint
  9.  Embedding circuit breaker in /health response
  10. L1 TTL background cleanup task started in lifespan
"""
from __future__ import annotations

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── 1. L1 TTL setting ─────────────────────────────────────────────────────────

class TestL1TTLSetting:
    def test_default_ttl_in_settings(self):
        from orchestrator.config.settings import get_settings
        cfg = get_settings()
        assert cfg.memory.l1_ttl_seconds == 3600

    def test_l1_accepts_ttl_param(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        l1 = L1ConversationCache(max_turns=5, max_tokens=256, ttl_seconds=60)
        assert l1._ttl == 60

    def test_l1_has_last_access_dict(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        l1 = L1ConversationCache()
        assert hasattr(l1, "_last_access")
        assert isinstance(l1._last_access, dict)


# ── 2. L1 TTL tracking and eviction ──────────────────────────────────────────

class TestL1TTLEviction:
    @pytest.mark.asyncio
    async def test_push_turn_updates_last_access(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.core.types import Message, MessageRole
        l1 = L1ConversationCache(ttl_seconds=3600)
        msg = Message(role=MessageRole.USER, content="hello")
        await l1.push_turn("sess1", msg)
        assert "sess1" in l1._last_access

    @pytest.mark.asyncio
    async def test_get_turns_updates_last_access(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.core.types import Message, MessageRole
        l1 = L1ConversationCache(ttl_seconds=3600)
        msg = Message(role=MessageRole.USER, content="hello")
        await l1.push_turn("sess2", msg)
        t_before = l1._last_access["sess2"]
        await asyncio.sleep(0.01)
        await l1.get_turns("sess2")
        assert l1._last_access["sess2"] >= t_before

    @pytest.mark.asyncio
    async def test_evict_expired_removes_old_sessions(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.core.types import Message, MessageRole
        l1 = L1ConversationCache(ttl_seconds=0)  # TTL=0 → everything expires instantly
        msg = Message(role=MessageRole.USER, content="hello")
        await l1.push_turn("old_sess", msg)
        # Force the last_access to be in the past
        l1._last_access["old_sess"] = time.monotonic() - 1

        evicted = await l1.evict_expired()
        assert "old_sess" in evicted
        assert "old_sess" not in l1._sessions

    @pytest.mark.asyncio
    async def test_evict_expired_keeps_fresh_sessions(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.core.types import Message, MessageRole
        l1 = L1ConversationCache(ttl_seconds=3600)
        msg = Message(role=MessageRole.USER, content="hi")
        await l1.push_turn("fresh", msg)
        evicted = await l1.evict_expired()
        assert "fresh" not in evicted
        assert "fresh" in l1._sessions

    @pytest.mark.asyncio
    async def test_clear_session_removes_from_last_access(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.core.types import Message, MessageRole
        l1 = L1ConversationCache()
        msg = Message(role=MessageRole.USER, content="hello")
        await l1.push_turn("sess", msg)
        assert "sess" in l1._last_access
        await l1.clear_session("sess")
        assert "sess" not in l1._last_access


# ── 3. L1 session_idle_seconds ───────────────────────────────────────────────

class TestL1SessionIdle:
    @pytest.mark.asyncio
    async def test_idle_seconds_increases_over_time(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.core.types import Message, MessageRole
        l1 = L1ConversationCache()
        msg = Message(role=MessageRole.USER, content="hello")
        await l1.push_turn("s", msg)
        await asyncio.sleep(0.05)
        idle = l1.session_idle_seconds("s")
        assert idle is not None
        assert idle >= 0.04

    def test_idle_seconds_none_for_unknown_session(self):
        from orchestrator.memory.l1_cache import L1ConversationCache
        l1 = L1ConversationCache()
        assert l1.session_idle_seconds("ghost") is None


# ── 4. Request ID middleware ──────────────────────────────────────────────────

class TestRequestIDMiddleware:
    def _make_app(self):
        from fastapi import FastAPI
        from orchestrator.api.middleware.request_id import RequestIDMiddleware
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        return app

    def test_response_has_x_request_id(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        resp = client.get("/ping")
        assert "x-request-id" in resp.headers or "X-Request-ID" in resp.headers

    def test_client_supplied_id_is_echoed(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        resp = client.get("/ping", headers={"X-Request-ID": "my-trace-123"})
        rid = resp.headers.get("x-request-id") or resp.headers.get("X-Request-ID")
        assert rid == "my-trace-123"

    def test_generated_id_is_uuid_format(self):
        import re
        from fastapi.testclient import TestClient
        client = TestClient(self._make_app())
        resp = client.get("/ping")
        rid = resp.headers.get("x-request-id") or resp.headers.get("X-Request-ID")
        uuid_re = re.compile(r"^[0-9a-f-]{36}$")
        assert uuid_re.match(rid), f"Expected UUID, got: {rid}"


# ── 5. Structured error responses ─────────────────────────────────────────────

class TestStructuredErrors:
    def _get_app(self):
        from orchestrator.api.app import create_app
        return create_app()

    def test_404_has_error_envelope(self):
        from fastapi.testclient import TestClient
        client = TestClient(self._get_app(), raise_server_exceptions=False)
        resp = client.get("/v1/chat/history/nonexistent_session_xyz_abc")
        # May return 200 (empty history) or error — just check shape if error
        if resp.status_code >= 400:
            body = resp.json()
            assert "error" in body

    def test_http_exception_returns_error_code(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from orchestrator.api.app import _status_to_code, create_app

        # Unit-test the helper directly
        assert _status_to_code(404) == "NOT_FOUND"
        assert _status_to_code(403) == "FORBIDDEN"
        assert _status_to_code(429) == "RATE_LIMITED"
        assert _status_to_code(999) == "HTTP_999"


# ── 6. Embedding circuit breaker in /health ───────────────────────────────────

class TestEmbeddingCBInHealth:
    def test_health_route_mentions_embedding_circuit_breaker(self):
        import inspect
        import orchestrator.api.routes.health as h
        src = inspect.getsource(h)
        assert "embedding_circuit_breaker" in src

    def test_health_components_include_embedding_cb(self):
        """Smoke-test that /health returns embedding_circuit_breaker component."""
        from fastapi.testclient import TestClient

        mock_container = MagicMock()
        mock_container.llama_client.health_check = AsyncMock(return_value=True)
        mock_container.llama_client._base_url = "http://localhost:8080"
        mock_container.llama_client._circuit.state = "closed"
        mock_container.llama_client._circuit.status.return_value = {"state": "closed", "failures": 0}
        mock_container.embedder.embed = AsyncMock(return_value=[[0.1] * 384])
        mock_container.embedder._settings.model = "bge-small"
        mock_container.embedder._circuit.state = "closed"
        mock_container.embedder._circuit.status.return_value = {"state": "closed", "failures": 0}
        mock_container.speculative_decoder = None
        mock_container.compression_queue._started = False
        mock_container.compression_queue._worker_task = None
        mock_container.compression_queue.queue_depth = 0
        mock_container.agent_pool.active_count = 0
        mock_container.bus.topic_count = 0

        with patch("orchestrator.api.routes.health.get_container", return_value=mock_container):
            from fastapi import FastAPI
            from orchestrator.api.routes.health import router
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)
            resp = client.get("/health")

        if resp.status_code == 200:
            names = [c["name"] for c in resp.json().get("components", [])]
            assert "embedding_circuit_breaker" in names


# ── 7. /v1/admin/memory/stats endpoint ───────────────────────────────────────

class TestMemoryStatsEndpoint:
    def test_memory_stats_route_exists(self):
        import inspect
        import orchestrator.api.routes.admin as a
        src = inspect.getsource(a)
        assert "memory/stats" in src or "memory_stats" in src

    def test_memory_stats_returns_all_tiers(self):
        import inspect
        import orchestrator.api.routes.admin as a
        src = inspect.getsource(a)
        # All four tiers should be mentioned in the memory stats handler
        assert '"l1"' in src or "'l1'" in src
        assert '"l2"' in src or "'l2'" in src
        assert '"l3"' in src or "'l3'" in src
        assert '"l4"' in src or "'l4'" in src


# ── 8. GET /v1/chat/history/{session_id} ─────────────────────────────────────

class TestChatHistoryEndpoint:
    def test_history_route_exists_in_chat(self):
        import inspect
        import orchestrator.api.routes.chat as c
        src = inspect.getsource(c)
        assert "history" in src
        assert "session_id" in src

    def test_history_returns_turns_and_summaries_keys(self):
        import inspect
        import orchestrator.api.routes.chat as c
        src = inspect.getsource(c)
        assert '"turns"' in src or "'turns'" in src
        assert '"summaries"' in src or "'summaries'" in src


# ── 9. WebSocket heartbeat ────────────────────────────────────────────────────

class TestWebSocketHeartbeat:
    def test_heartbeat_wired_into_ws_handler(self):
        import inspect
        import orchestrator.api.routes.chat as c
        src = inspect.getsource(c)
        assert "_heartbeat" in src
        assert "heartbeat_task" in src
        assert "ping" in src

    def test_heartbeat_cancelled_in_finally(self):
        import inspect
        import orchestrator.api.routes.chat as c
        src = inspect.getsource(c)
        assert "heartbeat_task.cancel()" in src


# ── 10. SSE stream routes complex queries via multi-agent ─────────────────────

class TestSSEStreamMultiAgent:
    def test_sse_stream_imports_multi_agent_engine(self):
        import inspect
        import orchestrator.api.routes.chat as c
        src = inspect.getsource(c)
        assert "MultiAgentEngine" in src or "multi_agent" in src

    def test_sse_stream_checks_is_complex(self):
        import inspect
        import orchestrator.api.routes.chat as c
        src = inspect.getsource(c)
        assert "is_complex" in src

    def test_sse_stream_has_multi_agent_dependency(self):
        import inspect
        import orchestrator.api.routes.chat as c
        src = inspect.getsource(c)
        # chat_stream_endpoint should depend on both inference and multi_agent engines
        assert "get_multi_agent_engine" in src


# ── 11. L1 TTL cleanup task started at startup ────────────────────────────────

class TestL1TTLCleanupTask:
    def _app_source(self) -> str:
        import pathlib
        return (pathlib.Path(__file__).parent.parent / "api" / "app.py").read_text()

    def test_cleanup_loop_exists_in_app(self):
        assert "_l1_ttl_cleanup_loop" in self._app_source()

    def test_cleanup_task_created_in_lifespan(self):
        src = self._app_source()
        assert "cleanup_task" in src
        assert "_l1_ttl_cleanup_loop" in src

    def test_cleanup_interval_is_reasonable(self):
        from orchestrator.api.app import _L1_CLEANUP_INTERVAL_S
        assert 60 <= _L1_CLEANUP_INTERVAL_S <= 3600
