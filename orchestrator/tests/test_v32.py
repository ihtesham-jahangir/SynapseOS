"""
v3.2 feature tests — covers all upgrades applied in this session.

Groups:
  1. Agent retry with exponential backoff
  2. Per-API-key rate limiting
  3. WebSocket max-connections cap
  4. SQLite health check in /health/ready
  5. Dead code removal (fact_extractor)
  6. Bulk document ingestion  (POST /v1/documents/batch)
  7. Document update          (PUT  /v1/documents/{doc_id})
  8. External prompt file     (task_planner loads from .txt)
  9. Admin audit log          (GET  /v1/admin/audit)
  10. Graceful shutdown drain  (API_SHUTDOWN_TIMEOUT_S setting)
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── 1. Agent retry with exponential backoff ───────────────────────────────────

class TestAgentRetry:
    @pytest.fixture
    def pool(self):
        from orchestrator.agents.agent_pool import AgentPool
        llama = MagicMock()
        bus = MagicMock()
        bus.publish = AsyncMock()
        return AgentPool(
            llama_client=llama,
            bus=bus,
            max_parallel=1,
            task_timeout_s=0.2,
            max_retries=2,
            retry_delay_base_s=0.01,
        )

    def test_pool_stores_max_retries(self, pool):
        assert pool._max_retries == 2

    def test_pool_stores_retry_delay(self, pool):
        assert pool._retry_delay_base == 0.01

    @pytest.mark.asyncio
    async def test_subtask_retried_on_timeout(self, pool):
        from orchestrator.agents.task_types import SubTask, AgentRole, TaskStatus

        call_count = 0

        async def slow_execute(sub_task, context):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(10)  # always timeout

        with patch.object(
            pool, "_make_agent",
            side_effect=lambda role: _mock_agent(slow_execute),
        ):
            sub_task = SubTask(id="t1", description="test", role=AgentRole.GENERAL)
            result = await pool._run_one("graph-1", sub_task, "ctx")

        assert result.status == TaskStatus.FAILED
        assert "3 attempts" in (result.error or "")  # 1 + 2 retries = 3 attempts
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_subtask_succeeds_on_second_attempt(self, pool):
        from orchestrator.agents.task_types import SubTask, AgentRole, TaskStatus

        attempt = 0

        async def flaky_execute(sub_task, context):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                await asyncio.sleep(10)  # first call times out
            sub_task.status = TaskStatus.DONE
            sub_task.result = "success"
            return sub_task

        with patch.object(
            pool, "_make_agent",
            side_effect=lambda role: _mock_agent(flaky_execute),
        ):
            sub_task = SubTask(id="t1", description="test", role=AgentRole.GENERAL)
            result = await pool._run_one("graph-1", sub_task, "ctx")

        assert result.status == TaskStatus.DONE
        assert result.result == "success"


def _mock_agent(execute_fn):
    """Build a minimal agent mock that delegates _execute to execute_fn."""
    from orchestrator.agents.task_types import SubTask, TaskStatus
    import time

    agent = MagicMock()

    async def run(task_id, sub_task, context):
        t0 = time.perf_counter()
        result = await execute_fn(sub_task, context)
        sub_task.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result if result is not None else sub_task

    agent.run = run
    return agent


# ── 2. Per-API-key rate limiting ──────────────────────────────────────────────

class TestPerKeyRateLimiting:
    def test_bucket_key_uses_ip_when_auth_disabled(self):
        from orchestrator.api.middleware.rate_limit import RateLimitMiddleware
        from unittest.mock import MagicMock
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        mw._auth_enabled = False

        request = MagicMock()
        request.headers = {}
        request.client.host = "1.2.3.4"
        assert mw._bucket_key(request) == "ip:1.2.3.4"

    def test_bucket_key_uses_api_key_when_auth_enabled(self):
        from orchestrator.api.middleware.rate_limit import RateLimitMiddleware
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        mw._auth_enabled = True

        request = MagicMock()
        request.headers = {"X-API-Key": "secret-key-12345"}
        key = mw._bucket_key(request)
        assert key.startswith("key:")
        assert "secret-key-1" in key  # first 16 chars

    def test_bucket_key_falls_back_to_ip_when_no_key_header(self):
        from orchestrator.api.middleware.rate_limit import RateLimitMiddleware
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        mw._auth_enabled = True

        request = MagicMock()
        request.headers = {}
        request.client.host = "9.9.9.9"
        assert mw._bucket_key(request) == "ip:9.9.9.9"


# ── 3. WebSocket max-connections cap ──────────────────────────────────────────

class TestWebSocketMaxConnections:
    @pytest.fixture
    def manager(self):
        from orchestrator.streaming.websocket_manager import WebSocketManager
        return WebSocketManager(max_connections=2)

    @pytest.mark.asyncio
    async def test_connect_returns_true_when_capacity_available(self, manager):
        ws = MagicMock()
        ws.accept = AsyncMock()
        result = await manager.connect(ws, "sess-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_connect_rejects_when_at_capacity(self, manager):
        ws1, ws2 = MagicMock(), MagicMock()
        ws1.accept = AsyncMock()
        ws2.accept = AsyncMock()
        await manager.connect(ws1, "sess-1")
        await manager.connect(ws2, "sess-2")

        ws3 = MagicMock()
        ws3.close = AsyncMock()
        result = await manager.connect(ws3, "sess-3")
        assert result is False
        ws3.close.assert_called_once_with(code=1013, reason="Server at capacity")

    @pytest.mark.asyncio
    async def test_disconnect_frees_slot(self, manager):
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws2 = MagicMock()
        ws2.accept = AsyncMock()

        await manager.connect(ws1, "sess-1")
        await manager.connect(ws2, "sess-2")
        await manager.disconnect(ws1, "sess-1")

        ws3 = MagicMock()
        ws3.accept = AsyncMock()
        result = await manager.connect(ws3, "sess-3")
        assert result is True


# ── 4. SQLite health check ────────────────────────────────────────────────────

class TestSQLiteHealthCheck:
    @pytest.mark.asyncio
    async def test_check_sqlite_returns_true_for_valid_path(self, tmp_path):
        from orchestrator.api.routes.health import _check_sqlite
        db_path = str(tmp_path / "test.db")
        result = await _check_sqlite(db_path)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_sqlite_returns_false_for_bad_path(self):
        from orchestrator.api.routes.health import _check_sqlite
        result = await _check_sqlite("/nonexistent/path/test.db")
        assert result is False


# ── 5. Dead code removal in fact_extractor ────────────────────────────────────

class TestFactExtractorDeadCode:
    def test_no_module_level_seen_cache(self):
        import orchestrator.utils.fact_extractor as fe
        assert not hasattr(fe, "_seen_cache"), \
            "_seen_cache was dead code and should have been removed"

    def test_extract_facts_still_deduplicates_within_call(self):
        from orchestrator.utils.fact_extractor import extract_facts
        # Two patterns might match the same fact — should be deduplicated
        facts = extract_facts("My name is Alice.")
        texts = [f for f, _ in facts]
        assert len(texts) == len(set(texts))


# ── 6. Bulk document ingestion ────────────────────────────────────────────────

class TestBulkDocumentIngestion:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c

    def test_batch_ingest_two_documents(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline
        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=3)
        mock_rag.index_size = 6

        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.post("/v1/documents/batch", json={
                "documents": [
                    {"content": "Doc one content here", "source": "s1", "metadata": {}},
                    {"content": "Doc two content here", "source": "s2", "metadata": {}},
                ]
            })
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["results"]) == 2
            assert data["total_chunks"] == 6
            assert data["total_time_ms"] >= 0
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)

    def test_batch_ingest_returns_uuids(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline
        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=1)
        mock_rag.index_size = 2

        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.post("/v1/documents/batch", json={
                "documents": [
                    {"content": "Some text", "source": "s", "metadata": {}},
                ]
            })
            assert resp.status_code == 200
            doc_id = resp.json()["results"][0]["document_id"]
            assert len(doc_id) == 36  # UUID
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)

    def test_batch_ingest_empty_list_returns_400(self, client):
        resp = client.post("/v1/documents/batch", json={"documents": []})
        assert resp.status_code == 400


# ── 7. Document update (PUT) ──────────────────────────────────────────────────

class TestDocumentUpdate:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c

    def test_put_document_returns_same_doc_id(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline
        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=2)
        mock_rag.delete_document = AsyncMock(return_value={"removed": 3})

        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.put(
                "/v1/documents/my-doc-id",
                json={"content": "Updated content here", "source": "s", "metadata": {}},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["document_id"] == "my-doc-id"
            assert data["chunks_created"] == 2
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)

    def test_put_document_works_even_if_doc_never_existed(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline
        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=1)
        mock_rag.delete_document = AsyncMock(side_effect=ValueError("not found"))

        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.put(
                "/v1/documents/new-doc",
                json={"content": "Brand new content", "source": "s", "metadata": {}},
            )
            assert resp.status_code == 200
            assert resp.json()["document_id"] == "new-doc"
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)


# ── 8. External prompt file ───────────────────────────────────────────────────

class TestExternalPromptFile:
    def test_prompt_file_exists(self):
        from pathlib import Path
        prompt_path = Path(__file__).parent.parent / "prompts" / "task_planner.txt"
        assert prompt_path.exists(), "orchestrator/prompts/task_planner.txt must exist"

    def test_planner_system_prompt_loaded_from_file(self):
        from orchestrator.agents.task_planner import _PLAN_SYSTEM
        assert len(_PLAN_SYSTEM) > 50
        assert "sub_tasks" in _PLAN_SYSTEM
        assert "summarizer" in _PLAN_SYSTEM

    def test_planner_prompt_not_hardcoded_in_module(self):
        import ast
        from pathlib import Path
        src = (Path(__file__).parent.parent / "agents" / "task_planner.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_PLAN_SYSTEM":
                        # Should be loaded from file, not a string literal
                        assert not isinstance(node.value, ast.Constant), \
                            "_PLAN_SYSTEM must be loaded from the prompt file, not a string literal"


# ── 9. Admin audit log ────────────────────────────────────────────────────────

class TestAdminAuditLog:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c

    _HEADERS = {"X-API-Key": "changeme-in-production"}

    def test_audit_endpoint_returns_list(self, client):
        resp = client.get("/v1/admin/audit", headers=self._HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_audit_log_record_and_list(self, tmp_path):
        from orchestrator.utils.audit_log import AuditLog

        async def _run():
            log = AuditLog(db_path=str(tmp_path / "audit.db"))
            await log.record("test_action", detail="some-session", actor="test")
            await log.record("another_action", detail="", actor="api")
            entries = await log.list(limit=10)
            return entries

        entries = asyncio.get_event_loop().run_until_complete(_run())
        assert len(entries) == 2
        actions = [e["action"] for e in entries]
        assert "test_action" in actions
        assert "another_action" in actions

    def test_audit_log_newest_first(self, tmp_path):
        from orchestrator.utils.audit_log import AuditLog

        async def _run():
            log = AuditLog(db_path=str(tmp_path / "audit2.db"))
            await log.record("first")
            await log.record("second")
            await log.record("third")
            return await log.list()

        entries = asyncio.get_event_loop().run_until_complete(_run())
        assert entries[0]["action"] == "third"
        assert entries[-1]["action"] == "first"

    def test_audit_log_requires_api_key(self, client):
        resp = client.get("/v1/admin/audit")
        assert resp.status_code == 403


# ── 10. Graceful shutdown setting ─────────────────────────────────────────────

class TestGracefulShutdownSetting:
    def test_shutdown_timeout_setting_exists(self):
        from orchestrator.config.settings import APISettings
        s = APISettings()
        assert hasattr(s, "shutdown_timeout_s")
        assert s.shutdown_timeout_s == 30

    def test_shutdown_timeout_env_override(self, monkeypatch):
        monkeypatch.setenv("API_SHUTDOWN_TIMEOUT_S", "60")
        from orchestrator.config.settings import APISettings
        s = APISettings()
        assert s.shutdown_timeout_s == 60
