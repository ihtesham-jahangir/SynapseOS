"""
v3.1 feature tests — covers all upgrades applied in this session.

Groups:
  1. L3 soft-delete  (clear_session now actually filters results)
  2. Fact extraction wiring  (record_turn fires store_fact for personal messages)
  3. Intent classifier fast-path  (threshold lowered: 2-keyword hits now fire)
  4. TaskPlanner keyword cleanup  (single "first" no longer triggers decomposition)
  5. Planner dependency validation  (bad dep IDs silently dropped)
  6. Admin auth  (all admin routes require X-API-Key)
  7. Session list endpoint  (GET /v1/admin/sessions)
  8. Document ID  (ingest returns real UUID, fixes tokens_indexed)
  9. Task history  (GET /v1/agents/tasks with limit/offset)
  10. Response cache key normalization  (punctuation-insensitive hits)
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── 1. L3 soft-delete ─────────────────────────────────────────────────────────

class TestL3SoftDelete:
    @pytest.fixture
    def l3(self):
        from orchestrator.memory.l3_cache import L3VectorMemory
        embedder = MagicMock()
        index = MagicMock()
        index.add = AsyncMock()
        index.persist = AsyncMock()
        index.search = AsyncMock(return_value=[
            ("id1", 0.9, {"session_id": "sess-A", "content": "fact A",
                          "memory_id": "id1", "importance": 0.9, "timestamp": 0.0}),
            ("id2", 0.8, {"session_id": "sess-B", "content": "fact B",
                          "memory_id": "id2", "importance": 0.8, "timestamp": 0.0}),
        ])
        embedder.embed_query = AsyncMock(return_value=[0.1] * 384)
        return L3VectorMemory(embedder=embedder, index=index)

    @pytest.mark.asyncio
    async def test_clear_session_filters_retrieve(self, l3):
        await l3.clear_session("sess-A")
        results = await l3.retrieve("query", "sess-B")
        ids = [r.id for r in results]
        assert "id1" not in ids
        assert "id2" in ids

    @pytest.mark.asyncio
    async def test_clear_session_does_not_affect_other_sessions(self, l3):
        await l3.clear_session("sess-X")
        results = await l3.retrieve("query", "sess-B")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_clear_session_is_idempotent(self, l3):
        await l3.clear_session("sess-A")
        await l3.clear_session("sess-A")
        results = await l3.retrieve("query", "sess-B")
        assert len(results) == 1


# ── 2. Fact extraction wiring ─────────────────────────────────────────────────

class TestFactExtractionWiring:
    @pytest.mark.asyncio
    async def test_record_turn_triggers_fact_store(self):
        from orchestrator.memory.memory_manager import MemoryManager
        from orchestrator.core.types import Message, MessageRole

        l1 = MagicMock()
        l1.push_turn = AsyncMock()
        l1.session_token_count = MagicMock(return_value=100)
        l1._max_tokens = 2048

        mgr = MemoryManager(l1=l1, l2=MagicMock(), l3=MagicMock(), l4=MagicMock())
        mgr.store_fact = AsyncMock()

        user_msg = Message(role=MessageRole.USER,
                           content="My name is Alice and I live in Berlin.")
        asst_msg = Message(role=MessageRole.ASSISTANT, content="Hello Alice!")

        await mgr.record_turn("s1", user_msg, asst_msg)

        # Give the create_task a tick to schedule
        await asyncio.sleep(0)
        # store_fact should have been called (at least name + location)
        assert mgr.store_fact.call_count >= 1

    @pytest.mark.asyncio
    async def test_record_turn_skips_short_messages(self):
        from orchestrator.memory.memory_manager import MemoryManager
        from orchestrator.core.types import Message, MessageRole

        l1 = MagicMock()
        l1.push_turn = AsyncMock()
        l1.session_token_count = MagicMock(return_value=10)
        l1._max_tokens = 2048

        mgr = MemoryManager(l1=l1, l2=MagicMock(), l3=MagicMock(), l4=MagicMock())
        mgr.store_fact = AsyncMock()

        user_msg = Message(role=MessageRole.USER, content="Hi")
        asst_msg = Message(role=MessageRole.ASSISTANT, content="Hello!")

        await mgr.record_turn("s1", user_msg, asst_msg)
        await asyncio.sleep(0)
        mgr.store_fact.assert_not_called()


# ── 3. Intent classifier fast-path threshold ──────────────────────────────────

class TestIntentClassifierThreshold:
    @pytest.fixture
    def classifier(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        return HybridIntentClassifier(embedder=None)

    def test_threshold_lowered_to_0_72(self, classifier):
        assert classifier.KEYWORD_CONFIDENCE == 0.72

    def test_two_keyword_hits_fire_fast_path(self, classifier):
        # "code" + "function" = 2 hits → confidence = 0.60 + 0.12 = 0.72 ≥ threshold
        result = classifier._keyword_classify("write a code function")
        assert result is not None
        intent, confidence = result
        assert confidence >= classifier.KEYWORD_CONFIDENCE

    def test_single_keyword_still_returns_result(self, classifier):
        result = classifier._keyword_classify("write code here")
        assert result is not None


# ── 4. TaskPlanner keyword cleanup ────────────────────────────────────────────

class TestTaskPlannerKeywords:
    def test_standalone_first_not_complex(self):
        from orchestrator.agents.task_planner import _is_complex
        assert not _is_complex("First explain what recursion is")

    def test_standalone_also_not_complex(self):
        from orchestrator.agents.task_planner import _is_complex
        assert not _is_complex("Also tell me about Python")

    def test_standalone_additionally_not_complex(self):
        from orchestrator.agents.task_planner import _is_complex
        assert not _is_complex("Additionally, what is a loop?")

    def test_compare_is_complex(self):
        from orchestrator.agents.task_planner import _is_complex
        assert _is_complex("compare Python and JavaScript performance")

    def test_and_then_is_complex(self):
        from orchestrator.agents.task_planner import _is_complex
        assert _is_complex("Explain sorting and then write a sort function")

    def test_step_by_step_is_complex(self):
        from orchestrator.agents.task_planner import _is_complex
        assert _is_complex("Walk me through how to set up a Django project step by step")

    def test_long_query_is_complex(self):
        from orchestrator.agents.task_planner import _is_complex
        assert _is_complex("x" * 201)


# ── 5. Planner dependency validation ──────────────────────────────────────────

class TestPlannerDependencyValidation:
    @pytest.fixture
    def planner(self):
        from orchestrator.agents.task_planner import TaskPlanner
        return TaskPlanner(llama_client=MagicMock())

    def test_invalid_dep_ids_dropped(self, planner):
        raw_json = '{"sub_tasks": [{"id": "t1", "description": "do A", "role": "research", "dependencies": []}, {"id": "t2", "description": "do B", "role": "general", "dependencies": ["t1", "t99"]}]}'
        tasks = planner._parse(raw_json, 5)
        t2 = next(t for t in tasks if t.id == "t2")
        assert "t99" not in t2.dependencies
        assert "t1" in t2.dependencies

    def test_all_valid_deps_preserved(self, planner):
        raw_json = '{"sub_tasks": [{"id": "t1", "description": "A", "role": "research", "dependencies": []}, {"id": "t2", "description": "B", "role": "summarizer", "dependencies": ["t1"]}]}'
        tasks = planner._parse(raw_json, 5)
        t2 = next(t for t in tasks if t.id == "t2")
        assert t2.dependencies == ["t1"]

    def test_self_referential_dep_dropped(self, planner):
        raw_json = '{"sub_tasks": [{"id": "t1", "description": "X", "role": "general", "dependencies": ["t1"]}]}'
        tasks = planner._parse(raw_json, 5)
        assert tasks[0].dependencies == []


# ── 6. Admin auth ─────────────────────────────────────────────────────────────

class TestAdminAuth:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c

    _GOOD = {"X-API-Key": "changeme-in-production"}

    def test_no_key_returns_403(self, client):
        resp = client.get("/v1/admin/sessions")
        assert resp.status_code == 403

    def test_wrong_key_returns_403(self, client):
        resp = client.get("/v1/admin/sessions", headers={"X-API-Key": "bad"})
        assert resp.status_code == 403

    def test_correct_key_accepted(self, client):
        resp = client.get("/v1/admin/sessions", headers=self._GOOD)
        assert resp.status_code == 200


# ── 7. Session list endpoint ──────────────────────────────────────────────────

class TestSessionListEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c

    _HEADERS = {"X-API-Key": "changeme-in-production"}

    def test_sessions_endpoint_returns_list(self, client):
        resp = client.get("/v1/admin/sessions", headers=self._HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)


# ── 8. Document ID fix ────────────────────────────────────────────────────────

class TestDocumentIdFix:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c

    def test_ingest_returns_uuid_when_no_doc_id(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline

        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=3)
        mock_rag.index_size = 3

        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.post(
                "/v1/documents",
                json={"content": "Some document text here", "source": "test", "metadata": {}},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["document_id"] != "unknown"
            assert len(data["document_id"]) == 36  # UUID format
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)

    def test_ingest_preserves_provided_doc_id(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline

        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=2)
        mock_rag.index_size = 2

        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.post(
                "/v1/documents",
                json={"content": "Text", "source": "s", "metadata": {"doc_id": "my-custom-id"}},
            )
            assert resp.status_code == 200
            assert resp.json()["document_id"] == "my-custom-id"
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)

    def test_ingest_tokens_indexed_is_word_count(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline

        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=1)
        mock_rag.index_size = 1
        content = "one two three four five"  # 5 words

        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.post(
                "/v1/documents",
                json={"content": content, "source": "s", "metadata": {}},
            )
            assert resp.status_code == 200
            assert resp.json()["tokens_indexed"] == 5
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)


# ── 9. Task history endpoint ──────────────────────────────────────────────────

class TestTaskHistoryEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from orchestrator.api.app import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c

    def test_task_list_returns_dict(self, client):
        resp = client.get("/v1/agents/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert "limit" in data
        assert "offset" in data

    def test_task_list_respects_limit(self, client):
        resp = client.get("/v1/agents/tasks?limit=5")
        assert resp.status_code == 200
        assert resp.json()["limit"] == 5

    def test_task_list_rejects_bad_limit(self, client):
        resp = client.get("/v1/agents/tasks?limit=0")
        assert resp.status_code == 422

    def test_task_list_filters_by_session(self, client):
        resp = client.get("/v1/agents/tasks?session_id=nonexistent-session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"] == []


# ── 10. Response cache key normalization ──────────────────────────────────────

class TestResponseCacheKeyNormalization:
    @pytest.fixture
    def cache(self):
        from orchestrator.cache.response_cache import ResponseCache
        return ResponseCache(maxsize=10, ttl_seconds=3600)

    def test_same_key_with_and_without_punctuation(self, cache):
        key1 = cache._key("What is Python?")
        key2 = cache._key("What is Python")
        assert key1 == key2

    def test_same_key_different_whitespace(self, cache):
        key1 = cache._key("  what   is  python  ")
        key2 = cache._key("what is python")
        assert key1 == key2

    def test_same_key_trailing_period(self, cache):
        key1 = cache._key("Explain recursion.")
        key2 = cache._key("Explain recursion")
        assert key1 == key2

    def test_different_queries_differ(self, cache):
        key1 = cache._key("what is python")
        key2 = cache._key("what is java")
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_cache_hit_across_punctuation_variants(self, cache):
        await cache.set("what is python", "Python is a language", "coding", 5)
        hit = await cache.get("What is Python?")
        assert hit is not None
        assert hit.content == "Python is a language"
