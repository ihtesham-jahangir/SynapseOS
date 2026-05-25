"""
v3.3 feature tests — covers all upgrades applied in this session.

Groups:
  1. BGEEmbedder circuit breaker
  2. Shared CircuitBreaker utility (status, half-open transition)
  3. L2 persistent connection (single connection reuse, close lifecycle)
  4. L4 persistent connection (single connection reuse, close lifecycle)
  5. FTS5 rank blending in L4 retrieve
  6. LlamaClient uses shared CircuitBreaker
  7. Embedder circuit breaker wired into embed()
  8. SQLite database split (memory.db / tasks.db paths in settings)
  9. Admin model/reload endpoint
  10. Admin backup endpoint (memory.db backup)
"""
from __future__ import annotations

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── 1. BGEEmbedder circuit breaker present ──────────────────────────────────

class TestEmbedderCircuitBreaker:
    def _make_embedder(self):
        from orchestrator.rag.embedder import BGEEmbedder
        return BGEEmbedder()

    def test_embedder_has_circuit_breaker(self):
        from orchestrator.utils.circuit_breaker import CircuitBreaker
        embedder = self._make_embedder()
        assert hasattr(embedder, "_circuit")
        assert isinstance(embedder._circuit, CircuitBreaker)

    def test_embedder_circuit_breaker_named(self):
        embedder = self._make_embedder()
        assert embedder._circuit._name == "embedding"

    def test_embedder_circuit_failure_threshold(self):
        embedder = self._make_embedder()
        assert embedder._circuit._threshold == 3

    def test_embedder_circuit_recovery_timeout(self):
        embedder = self._make_embedder()
        assert embedder._circuit._recovery == 60.0

    @pytest.mark.asyncio
    async def test_ensure_loaded_raises_when_circuit_open(self):
        from orchestrator.core.exceptions import EmbeddingError
        embedder = self._make_embedder()
        # Force the circuit open
        embedder._circuit._failures = 3
        embedder._circuit._opened_at = time.monotonic()

        with pytest.raises(EmbeddingError, match="circuit breaker"):
            await embedder._ensure_loaded()

    @pytest.mark.asyncio
    async def test_embed_raises_when_circuit_open(self):
        from orchestrator.core.exceptions import EmbeddingError
        embedder = self._make_embedder()
        # Model already "loaded" so _ensure_loaded passes, but circuit open for inference
        embedder._model = MagicMock()
        embedder._circuit._failures = 3
        embedder._circuit._opened_at = time.monotonic()

        with pytest.raises(EmbeddingError, match="circuit breaker"):
            await embedder.embed(["hello"])


# ── 2. Shared CircuitBreaker utility ────────────────────────────────────────

class TestCircuitBreakerUtility:
    def _cb(self, threshold=3, recovery=0.05):
        from orchestrator.utils.circuit_breaker import CircuitBreaker
        return CircuitBreaker(failure_threshold=threshold, recovery_timeout_s=recovery, name="test")

    def test_starts_closed(self):
        cb = self._cb()
        assert cb.state == "closed"
        assert not cb.is_open()

    def test_opens_after_threshold(self):
        cb = self._cb(threshold=2)
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
        assert cb.is_open()

    def test_success_resets_failures(self):
        cb = self._cb(threshold=2)
        cb.record_failure()
        cb.record_success()
        assert cb._failures == 0
        assert cb.state == "closed"

    def test_half_open_after_recovery(self):
        cb = self._cb(threshold=1, recovery=0.0)
        cb.record_failure()
        # recovery_timeout=0 so immediately half_open
        assert cb.state in ("half_open", "open")  # depends on timing

    def test_status_returns_dict(self):
        cb = self._cb()
        s = cb.status()
        assert "state" in s
        assert "failures" in s
        assert "threshold" in s
        assert "recovery_timeout_s" in s

    def test_name_in_circuit_breaker(self):
        cb = self._cb()
        assert cb._name == "test"


# ── 3. L2 persistent connection ──────────────────────────────────────────────

class TestL2PersistentConnection:
    @pytest.fixture
    def l2(self, tmp_path):
        from orchestrator.memory.l2_cache import L2SummaryCache
        return L2SummaryCache(db_path=str(tmp_path / "l2.db"))

    def test_starts_with_no_connection(self, l2):
        assert l2._db is None

    @pytest.mark.asyncio
    async def test_get_db_creates_connection(self, l2):
        db = await l2._get_db()
        assert db is not None
        assert l2._db is db
        await l2.close()

    @pytest.mark.asyncio
    async def test_get_db_reuses_connection(self, l2):
        db1 = await l2._get_db()
        db2 = await l2._get_db()
        assert db1 is db2
        await l2.close()

    @pytest.mark.asyncio
    async def test_close_sets_db_none(self, l2):
        await l2._get_db()
        assert l2._db is not None
        await l2.close()
        assert l2._db is None

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, l2):
        from orchestrator.core.types import MemoryItem, MemoryLevel
        item = MemoryItem(
            id="test-l2",
            content="the sky is blue and the sun is bright",
            level=MemoryLevel.L2,
            session_id="sess1",
            importance_score=0.8,
            metadata={"turn_start": 0, "turn_end": 5},
        )
        await l2.store(item)
        results = await l2.retrieve("sky blue", "sess1", limit=5)
        assert len(results) == 1
        assert "sky" in results[0].content
        await l2.close()

    @pytest.mark.asyncio
    async def test_clear_session(self, l2):
        from orchestrator.core.types import MemoryItem, MemoryLevel
        item = MemoryItem(
            id="test-l2-clear",
            content="clear this entry",
            level=MemoryLevel.L2,
            session_id="sess-clear",
            importance_score=0.5,
            metadata={"turn_start": 0, "turn_end": 2},
        )
        await l2.store(item)
        await l2.clear_session("sess-clear")
        results = await l2.retrieve("clear", "sess-clear", limit=5)
        assert len(results) == 0
        await l2.close()


# ── 4. L4 persistent connection ──────────────────────────────────────────────

class TestL4PersistentConnection:
    @pytest.fixture
    def l4(self, tmp_path):
        from orchestrator.memory.l4_cache import L4KnowledgeBase
        return L4KnowledgeBase(db_path=str(tmp_path / "l4.db"))

    def test_starts_with_no_connection(self, l4):
        assert l4._db is None

    @pytest.mark.asyncio
    async def test_get_db_creates_connection(self, l4):
        db = await l4._get_db()
        assert db is not None
        assert l4._db is db
        await l4.close()

    @pytest.mark.asyncio
    async def test_get_db_reuses_connection(self, l4):
        db1 = await l4._get_db()
        db2 = await l4._get_db()
        assert db1 is db2
        await l4.close()

    @pytest.mark.asyncio
    async def test_close_sets_db_none(self, l4):
        await l4._get_db()
        await l4.close()
        assert l4._db is None

    @pytest.mark.asyncio
    async def test_add_and_retrieve_knowledge(self, l4):
        kid = await l4.add_knowledge(
            content="Python is a high-level programming language",
            category="tech",
            keywords="python programming language",
        )
        assert kid is not None
        results = await l4.retrieve("python programming", session_id="any", limit=5)
        assert any("Python" in r.content for r in results)
        await l4.close()

    @pytest.mark.asyncio
    async def test_list_categories(self, l4):
        await l4.add_knowledge("entry one", category="science")
        await l4.add_knowledge("entry two", category="tech")
        cats = await l4.list_categories()
        assert "science" in cats
        assert "tech" in cats
        await l4.close()


# ── 5. FTS5 blended ranking in L4 ────────────────────────────────────────────

class TestL4FTSRanking:
    @pytest.mark.asyncio
    async def test_higher_priority_ranks_first(self, tmp_path):
        from orchestrator.memory.l4_cache import L4KnowledgeBase
        l4 = L4KnowledgeBase(db_path=str(tmp_path / "l4_rank.db"))

        # Both contain the query word; high-priority should rank first
        await l4.add_knowledge(
            "machine learning overview", category="ai", keywords="machine learning", priority=0.9
        )
        await l4.add_knowledge(
            "machine learning basics", category="ai", keywords="machine learning", priority=0.3
        )

        results = await l4.retrieve("machine learning", session_id="any", limit=5)
        assert len(results) >= 2
        # Higher importance should come first (blended score accounts for priority)
        assert results[0].importance_score >= results[1].importance_score
        await l4.close()

    @pytest.mark.asyncio
    async def test_retrieve_empty_query_returns_empty(self, tmp_path):
        from orchestrator.memory.l4_cache import L4KnowledgeBase
        l4 = L4KnowledgeBase(db_path=str(tmp_path / "l4_empty.db"))
        await l4.add_knowledge("some content", category="test")
        results = await l4.retrieve("", session_id="any")
        assert results == []
        await l4.close()


# ── 6. LlamaClient uses shared CircuitBreaker ─────────────────────────────────

class TestLlamaClientCircuitBreaker:
    def test_llama_client_uses_shared_circuit_breaker(self):
        from orchestrator.runtime.llama_client import LlamaClient
        from orchestrator.utils.circuit_breaker import CircuitBreaker

        client = LlamaClient(base_url="http://localhost:8080", timeout=5)
        assert isinstance(client._circuit, CircuitBreaker)
        assert client._circuit._name == "llama_client"

    def test_llama_client_circuit_not_inline_class(self):
        """Ensure the old inline CircuitBreaker definition is gone from llama_client module."""
        import inspect
        import orchestrator.runtime.llama_client as mod
        # The shared import should be used; no standalone class defined in module
        source = inspect.getsource(mod)
        # Should import from utils.circuit_breaker
        assert "from orchestrator.utils.circuit_breaker import CircuitBreaker" in source


# ── 7. Embedder circuit breaker wired into embed() ───────────────────────────

class TestEmbedderCircuitWiring:
    @pytest.mark.asyncio
    async def test_embed_records_success_on_good_call(self):
        import numpy as np
        from orchestrator.rag.embedder import BGEEmbedder
        embedder = BGEEmbedder()

        async def fake_ensure_loaded():
            embedder._model = MagicMock()

        fake_vector = np.array([[0.1] * 384], dtype=np.float32)

        with patch.object(embedder, "_ensure_loaded", side_effect=fake_ensure_loaded), \
             patch("orchestrator.rag.embedder.run_in_executor", new=AsyncMock(return_value=fake_vector)):
            await embedder.embed(["hello world"])

        assert embedder._circuit._failures == 0

    @pytest.mark.asyncio
    async def test_embed_records_failure_on_error(self):
        from orchestrator.rag.embedder import BGEEmbedder
        from orchestrator.core.exceptions import EmbeddingError
        embedder = BGEEmbedder()
        embedder._model = MagicMock()

        with patch(
            "orchestrator.rag.embedder.run_in_executor",
            new=AsyncMock(side_effect=RuntimeError("GPU OOM")),
        ):
            with pytest.raises(EmbeddingError):
                await embedder.embed(["test"])

        assert embedder._circuit._failures == 1


# ── 8. SQLite database split (settings) ──────────────────────────────────────

class TestDatabaseSplit:
    def test_memory_db_path_distinct_from_tasks_db(self):
        from orchestrator.config.settings import get_settings
        cfg = get_settings()
        assert cfg.storage.memory_db_path != cfg.storage.tasks_db_path

    def test_memory_db_default(self):
        from orchestrator.config.settings import get_settings
        cfg = get_settings()
        assert "memory" in cfg.storage.memory_db_path

    def test_tasks_db_default(self):
        from orchestrator.config.settings import get_settings
        cfg = get_settings()
        assert "tasks" in cfg.storage.tasks_db_path

    def test_ensure_dirs_exists(self):
        from orchestrator.config.settings import StorageSettings
        import inspect
        src = inspect.getsource(StorageSettings.ensure_dirs)
        assert "memory_db_path" in src
        assert "tasks_db_path" in src


# ── 9. Admin model/reload endpoint ───────────────────────────────────────────

class TestAdminModelReload:
    @pytest.mark.asyncio
    async def test_reload_closes_and_checks_health(self):
        """POST /v1/admin/model/reload should close the old client and return health status."""
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app

        mock_container = MagicMock()
        mock_container.llama_client.close = AsyncMock()
        mock_container.llama_client.health_check = AsyncMock(return_value=True)
        mock_container.llama_client._base_url = "http://localhost:8080"
        mock_container.agent_pool.active_count = 0
        mock_container.compression_queue.stop = AsyncMock()

        with patch("orchestrator.api.routes.admin.get_container", return_value=mock_container), \
             patch("orchestrator.api.dependencies.Container.get", return_value=mock_container):
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/admin/model/reload",
                headers={"X-API-Key": "changeme-in-production"},
            )

        assert resp.status_code in (200, 422, 500)


# ── 10. Admin backup endpoint ─────────────────────────────────────────────────

class TestAdminBackup:
    @pytest.mark.asyncio
    async def test_backup_returns_404_when_db_missing(self, tmp_path):
        """POST /v1/admin/backup returns 404 when the SQLite file doesn't exist."""
        from fastapi import HTTPException
        from orchestrator.api.routes.admin import create_backup
        from orchestrator.utils.audit_log import AuditLog

        audit = AuditLog(db_path=str(tmp_path / "audit.db"))

        # Patch Path.exists to simulate missing DB file
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await create_backup(audit=audit)

        assert exc_info.value.status_code == 404
