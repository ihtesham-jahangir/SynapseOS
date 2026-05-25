"""
API integration tests using FastAPI's TestClient.

The llama.cpp backend and embedding model are both mocked so no
external servers or downloaded weights are required.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from orchestrator.api.app import create_app
from orchestrator.core.types import (
    ChatResponse,
    IntentType,
    MemoryResult,
    RAGResult,
)


# ── App / client fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_rag_pipeline():
    """RAGPipeline stub that skips the real embedding model."""
    rag = MagicMock()
    rag.ingest = AsyncMock(return_value=2)
    rag.retrieve = AsyncMock(
        return_value=RAGResult(chunks=[], scores=[], total_tokens=0, query_time_ms=0)
    )
    rag.index_size = 42
    return rag


# ── Health endpoints ──────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_liveness(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "SynapseOS" in data["name"]

    def test_root_has_version(self, client):
        data = client.get("/").json()
        assert "version" in data

    def test_root_has_docs_link(self, client):
        data = client.get("/").json()
        assert "docs" in data


# ── Admin endpoints ───────────────────────────────────────────────────────────

class TestAdminEndpoints:
    # Default API key from settings (APISettings.key default)
    _ADMIN_HEADERS = {"X-API-Key": "changeme-in-production"}

    def test_classify_coding_intent(self, client):
        resp = client.post(
            "/v1/admin/classify",
            json={"text": "Write a Python sorting function"},
            headers=self._ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "coding"
        assert data["confidence"] > 0.5

    def test_classify_math_intent(self, client):
        resp = client.post(
            "/v1/admin/classify",
            json={"text": "Solve this equation: x^2 + 5x + 6 = 0"},
            headers=self._ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] == "math"

    def test_classify_translation_intent(self, client):
        resp = client.post(
            "/v1/admin/classify",
            json={"text": "Translate 'good morning' to Spanish"},
            headers=self._ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] == "translation"

    def test_classify_summarization_intent(self, client):
        resp = client.post(
            "/v1/admin/classify",
            json={"text": "Summarize this article for me"},
            headers=self._ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] == "summarization"

    def test_classify_missing_text_returns_422(self, client):
        resp = client.post("/v1/admin/classify", json={}, headers=self._ADMIN_HEADERS)
        assert resp.status_code == 422

    def test_admin_requires_api_key(self, client):
        resp = client.get("/v1/admin/experts")
        assert resp.status_code == 403

    def test_admin_rejects_wrong_key(self, client):
        resp = client.get("/v1/admin/experts", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_list_experts(self, client):
        resp = client.get("/v1/admin/experts", headers=self._ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "experts" in data
        assert len(data["experts"]) > 0
        assert "code" in data["experts"]
        assert "math" in data["experts"]

    def test_benchmark(self, client):
        resp = client.post(
            "/v1/admin/benchmark",
            json={"prompts": ["Write a Python function", "Solve 2+2"], "iterations": 2},
            headers=self._ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2
        for r in results:
            assert "avg_latency_ms" in r
            assert r["avg_latency_ms"] > 0

    def test_benchmark_zero_iterations_safe(self, client):
        resp = client.post(
            "/v1/admin/benchmark",
            json={"prompts": ["test"], "iterations": 1},
            headers=self._ADMIN_HEADERS,
        )
        assert resp.status_code == 200


# ── Document endpoints ────────────────────────────────────────────────────────

class TestDocumentEndpoints:
    def test_ingest_document(self, client):
        """Ingest a document via the API; RAG pipeline dependency is overridden."""
        from orchestrator.api.dependencies import get_rag_pipeline

        mock_rag = _mock_rag_pipeline()
        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.post(
                "/v1/documents",
                json={
                    "content": (
                        "The Python programming language was created by Guido van Rossum."
                        " It emphasizes code readability."
                    ),
                    "source": "test_doc",
                    "metadata": {"doc_id": "test123"},
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["chunks_created"] >= 1
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)

    def test_ingest_empty_content_returns_zero_chunks(self, client):
        from orchestrator.api.dependencies import get_rag_pipeline

        mock_rag = _mock_rag_pipeline()
        mock_rag.ingest = AsyncMock(return_value=0)
        client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        try:
            resp = client.post(
                "/v1/documents",
                json={"content": "", "source": "empty", "metadata": {}},
            )
            assert resp.status_code == 200
            assert resp.json()["chunks_created"] == 0
        finally:
            client.app.dependency_overrides.pop(get_rag_pipeline, None)

    def test_get_document_stats(self, client):
        resp = client.get("/v1/documents/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_vectors" in data
        assert isinstance(data["total_vectors"], int)

    def test_add_knowledge(self, client):
        resp = client.post(
            "/v1/documents/knowledge",
            json={
                "content": "SynapseOS is an AI operating system.",
                "category": "product",
                "keywords": "synapseos AI orchestrator",
                "priority": 0.9,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["category"] == "product"

    def test_add_knowledge_default_category(self, client):
        resp = client.post(
            "/v1/documents/knowledge",
            json={"content": "Some general fact."},
        )
        assert resp.status_code == 200
        assert resp.json()["category"] == "general"

    def test_list_knowledge_categories(self, client):
        client.post(
            "/v1/documents/knowledge",
            json={"content": "test content", "category": "science"},
        )
        resp = client.get("/v1/documents/knowledge/categories")
        assert resp.status_code == 200
        cats = resp.json()
        assert isinstance(cats, list)
        assert "science" in cats

    def test_knowledge_tokens_counted(self, client):
        resp = client.post(
            "/v1/documents/knowledge",
            json={"content": "word1 word2 word3 word4 word5"},
        )
        assert resp.status_code == 200
        assert resp.json()["tokens"] == 5


# ── Chat endpoint (mocked engine) ────────────────────────────────────────────

class TestChatEndpoint:
    def _chat_payload(self, text: str, session_id: str = "api-test") -> dict:
        return {
            "session_id": session_id,
            "messages": [{"role": "user", "content": text}],
        }

    def test_chat_returns_200(self, client):
        from orchestrator.api.dependencies import get_multi_agent_engine
        from orchestrator.core.types import ChatResponse

        mock_engine = MagicMock()
        mock_engine.process_request = AsyncMock(
            return_value=ChatResponse(
                session_id="api-test",
                content="Mock response",
                intent=IntentType.CONVERSATION,
                tokens_generated=10,
                total_tokens=50,
                time_to_first_token_ms=100.0,
                total_time_ms=300.0,
                memory_levels_used=[],
                rag_chunks_used=0,
                expert_used=None,
            )
        )

        client.app.dependency_overrides[get_multi_agent_engine] = lambda: mock_engine
        try:
            resp = client.post("/v1/chat", json=self._chat_payload("Hello"))
            assert resp.status_code == 200
            data = resp.json()
            assert data["content"] == "Mock response"
            assert data["session_id"] == "api-test"
        finally:
            client.app.dependency_overrides.pop(get_multi_agent_engine, None)

    def test_chat_missing_messages_returns_422(self, client):
        resp = client.post("/v1/chat", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_chat_timeout_returns_504(self, client):
        from orchestrator.api.dependencies import get_multi_agent_engine
        from orchestrator.core.exceptions import LlamaTimeoutError

        mock_engine = MagicMock()
        mock_engine.process_request = AsyncMock(side_effect=LlamaTimeoutError())

        client.app.dependency_overrides[get_multi_agent_engine] = lambda: mock_engine
        try:
            resp = client.post("/v1/chat", json=self._chat_payload("hi"))
            assert resp.status_code == 504
        finally:
            client.app.dependency_overrides.pop(get_multi_agent_engine, None)

    def test_chat_llama_error_returns_502(self, client):
        from orchestrator.api.dependencies import get_multi_agent_engine
        from orchestrator.core.exceptions import LlamaServerError

        mock_engine = MagicMock()
        mock_engine.process_request = AsyncMock(
            side_effect=LlamaServerError("backend down")
        )

        client.app.dependency_overrides[get_multi_agent_engine] = lambda: mock_engine
        try:
            resp = client.post("/v1/chat", json=self._chat_payload("hi"))
            assert resp.status_code == 502
        finally:
            client.app.dependency_overrides.pop(get_multi_agent_engine, None)

    def test_chat_orchestrator_error_returns_500(self, client):
        from orchestrator.api.dependencies import get_multi_agent_engine
        from orchestrator.core.exceptions import OrchestratorError

        mock_engine = MagicMock()
        mock_engine.process_request = AsyncMock(
            side_effect=OrchestratorError("internal problem")
        )

        client.app.dependency_overrides[get_multi_agent_engine] = lambda: mock_engine
        try:
            resp = client.post("/v1/chat", json=self._chat_payload("hi"))
            assert resp.status_code == 500
        finally:
            client.app.dependency_overrides.pop(get_multi_agent_engine, None)


# ── Exception hierarchy ───────────────────────────────────────────────────────

class TestExceptionHierarchy:
    def test_llama_timeout_is_orchestrator_error(self):
        from orchestrator.core.exceptions import (
            LlamaTimeoutError,
            LlamaServerError,
            OrchestratorError,
        )
        exc = LlamaTimeoutError()
        assert isinstance(exc, LlamaServerError)
        assert isinstance(exc, OrchestratorError)
        assert exc.code == "LLAMA_TIMEOUT"

    def test_llama_server_error_has_message(self):
        from orchestrator.core.exceptions import LlamaServerError
        exc = LlamaServerError("connection refused")
        assert exc.message == "connection refused"
        assert exc.code == "LLAMA_SERVER_ERROR"

    def test_embedding_error_code(self):
        from orchestrator.core.exceptions import EmbeddingError
        exc = EmbeddingError("model failed")
        assert exc.code == "EMBEDDING_ERROR"

    def test_memory_error_includes_level(self):
        from orchestrator.core.exceptions import MemoryError
        exc = MemoryError("L3 broken", "L3")
        assert "L3" in exc.code

    def test_rag_error_code(self):
        from orchestrator.core.exceptions import RAGError
        exc = RAGError("index empty")
        assert exc.code == "RAG_ERROR"

    def test_token_budget_exceeded_stores_values(self):
        from orchestrator.core.exceptions import TokenBudgetExceededError
        exc = TokenBudgetExceededError(budget=1000, actual=1500)
        assert exc.budget == 1000
        assert exc.actual == 1500
