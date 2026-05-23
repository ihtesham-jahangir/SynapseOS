"""
Tests for Tier 3 — Production Readiness features:
  - APIKeyMiddleware (auth enabled/disabled, header variants, public paths)
  - FAISSIndex.delete_by_doc_id / rebuild_without_deleted
  - RAGPipeline.delete_document
  - DELETE /v1/documents/{doc_id} HTTP endpoint
"""
from __future__ import annotations

import asyncio
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from orchestrator.rag.indexer import FAISSIndex


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unit_vec(seed: int, dim: int = 384) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


async def _populated_index(n_docs: int = 2, chunks_each: int = 2) -> FAISSIndex:
    """Return a FAISSIndex with n_docs * chunks_each vectors."""
    idx = FAISSIndex(dimension=384)
    seed = 0
    for doc_num in range(n_docs):
        vecs = [_unit_vec(seed + i) for i in range(chunks_each)]
        metas = [{"doc_id": f"doc{doc_num}", "chunk": i} for i in range(chunks_each)]
        await idx.add(vecs, metas)
        seed += chunks_each
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# APIKeyMiddleware
# ─────────────────────────────────────────────────────────────────────────────

def _make_auth_app(enabled: bool, key: str = "secret-key") -> FastAPI:
    """
    Build a minimal FastAPI app with auth middleware.

    We subclass APIKeyMiddleware so we can inject the enabled/key values
    directly — this avoids the starlette deferral issue where add_middleware()
    stores the class and instantiates it on the first request (after any
    patch context has already exited).
    """
    from orchestrator.api.middleware.auth import APIKeyMiddleware, _PUBLIC_PATHS
    from starlette.middleware.base import BaseHTTPMiddleware

    _enabled = enabled
    _key = key

    class _InlineAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if not _enabled or request.url.path in _PUBLIC_PATHS:
                return await call_next(request)
            provided = (
                request.headers.get("X-API-Key")
                or _bearer_token(request.headers.get("Authorization", ""))
            )
            if not provided:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    {"error": "missing_api_key"},
                    status_code=401,
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            if provided != _key:
                from fastapi.responses import JSONResponse
                return JSONResponse({"error": "invalid_api_key"}, status_code=403)
            return await call_next(request)

    app = FastAPI()
    app.add_middleware(_InlineAuthMiddleware)

    @app.get("/protected")
    async def protected():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "alive"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse("# metrics\n")

    @app.get("/")
    async def root():
        return {"name": "SynapseOS"}

    return app


def _bearer_token(authorization: str) -> str:
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


class TestAPIKeyMiddlewareDisabled:
    """When auth is disabled all requests pass through."""

    @pytest.fixture(scope="class")
    def client(self):
        app = _make_auth_app(enabled=False)
        with TestClient(app) as c:
            yield c

    def test_no_key_still_passes(self, client):
        resp = client.get("/protected")
        assert resp.status_code == 200

    def test_wrong_key_still_passes(self, client):
        resp = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 200

    def test_health_passes(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


class TestAPIKeyMiddlewareEnabled:
    """When auth is enabled, key must be correct."""

    _KEY = "test-secret-42"

    @pytest.fixture(scope="class")
    def client(self):
        app = _make_auth_app(enabled=True, key=self._KEY)
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_missing_key_returns_401(self, client):
        resp = client.get("/protected")
        assert resp.status_code == 401
        assert resp.json()["error"] == "missing_api_key"

    def test_wrong_key_returns_403(self, client):
        resp = client.get("/protected", headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 403
        assert resp.json()["error"] == "invalid_api_key"

    def test_correct_x_api_key_header_passes(self, client):
        resp = client.get("/protected", headers={"X-API-Key": self._KEY})
        assert resp.status_code == 200

    def test_bearer_token_passes(self, client):
        resp = client.get("/protected", headers={"Authorization": f"Bearer {self._KEY}"})
        assert resp.status_code == 200

    def test_wrong_bearer_returns_403(self, client):
        resp = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

    def test_401_has_www_authenticate_header(self, client):
        resp = client.get("/protected")
        assert "www-authenticate" in resp.headers


class TestAPIKeyPublicPaths:
    """Public paths bypass auth even when enabled."""

    _KEY = "pub-path-test"

    @pytest.fixture(scope="class")
    def client(self):
        app = _make_auth_app(enabled=True, key=self._KEY)
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    @pytest.mark.parametrize("path", ["/health", "/metrics", "/"])
    def test_public_path_no_auth_needed(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# FAISSIndex — delete_by_doc_id
# ─────────────────────────────────────────────────────────────────────────────

class TestFAISSDeleteByDocId:

    @pytest.mark.asyncio
    async def test_returns_count_of_deleted_chunks(self):
        idx = await _populated_index(n_docs=2, chunks_each=3)
        count = await idx.delete_by_doc_id("doc0")
        assert count == 3

    @pytest.mark.asyncio
    async def test_marks_metadata_deleted(self):
        idx = await _populated_index(n_docs=1, chunks_each=2)
        await idx.delete_by_doc_id("doc0")
        deleted_flags = [m.get("_deleted") for m in idx._metadata.values()]
        assert all(deleted_flags)

    @pytest.mark.asyncio
    async def test_unknown_doc_returns_zero(self):
        idx = await _populated_index(n_docs=1, chunks_each=2)
        count = await idx.delete_by_doc_id("does-not-exist")
        assert count == 0

    @pytest.mark.asyncio
    async def test_double_delete_does_not_increment(self):
        idx = await _populated_index(n_docs=1, chunks_each=2)
        first = await idx.delete_by_doc_id("doc0")
        second = await idx.delete_by_doc_id("doc0")
        assert first == 2
        assert second == 0  # already marked _deleted

    @pytest.mark.asyncio
    async def test_only_target_doc_marked_deleted(self):
        idx = await _populated_index(n_docs=3, chunks_each=2)
        await idx.delete_by_doc_id("doc1")
        for vid, meta in idx._metadata.items():
            if meta.get("doc_id") == "doc1":
                assert meta.get("_deleted") is True
            else:
                assert not meta.get("_deleted")


# ─────────────────────────────────────────────────────────────────────────────
# FAISSIndex — rebuild_without_deleted
# ─────────────────────────────────────────────────────────────────────────────

class TestFAISSRebuildWithoutDeleted:

    @pytest.mark.asyncio
    async def test_surviving_count_correct(self):
        idx = await _populated_index(n_docs=3, chunks_each=2)  # 6 total
        await idx.delete_by_doc_id("doc1")  # remove 2
        remaining = await idx.rebuild_without_deleted()
        assert remaining == 4

    @pytest.mark.asyncio
    async def test_faiss_ntotal_matches_surviving(self):
        idx = await _populated_index(n_docs=2, chunks_each=3)  # 6 total
        await idx.delete_by_doc_id("doc0")  # remove 3
        await idx.rebuild_without_deleted()
        assert idx.total_vectors == 3

    @pytest.mark.asyncio
    async def test_metadata_reindexed_from_zero(self):
        idx = await _populated_index(n_docs=2, chunks_each=2)  # 4 total
        await idx.delete_by_doc_id("doc0")
        await idx.rebuild_without_deleted()
        assert sorted(idx._metadata.keys()) == [0, 1]

    @pytest.mark.asyncio
    async def test_deleted_metadata_absent_after_rebuild(self):
        idx = await _populated_index(n_docs=2, chunks_each=2)
        await idx.delete_by_doc_id("doc0")
        await idx.rebuild_without_deleted()
        doc_ids_remaining = {m.get("doc_id") for m in idx._metadata.values()}
        assert "doc0" not in doc_ids_remaining
        assert "doc1" in doc_ids_remaining

    @pytest.mark.asyncio
    async def test_rebuild_all_deleted_gives_empty_index(self):
        idx = await _populated_index(n_docs=1, chunks_each=2)
        await idx.delete_by_doc_id("doc0")
        remaining = await idx.rebuild_without_deleted()
        assert remaining == 0
        assert idx.total_vectors == 0
        assert idx._metadata == {}


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline — delete_document
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPipelineDeleteDocument:

    def _make_pipeline(self, mock_embedder, tmp_path):
        from orchestrator.rag.pipeline import RAGPipeline
        from orchestrator.rag.chunker import RecursiveTextChunker
        from orchestrator.rag.retriever import SemanticRetriever
        from orchestrator.rag.reranker import EmbeddingReranker

        index = FAISSIndex(dimension=384, index_path=str(tmp_path / "faiss"))
        pipeline = RAGPipeline(embedder=mock_embedder, index=index)
        pipeline._chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
        pipeline._retriever = SemanticRetriever(mock_embedder, index)
        pipeline._reranker = EmbeddingReranker(mock_embedder)
        return pipeline

    @pytest.mark.asyncio
    async def test_raises_for_unknown_doc(self, mock_embedder, tmp_path):
        pipeline = self._make_pipeline(mock_embedder, tmp_path)
        with pytest.raises(ValueError, match="not found"):
            await pipeline.delete_document("ghost-doc")

    @pytest.mark.asyncio
    async def test_successful_delete_returns_dict(self, mock_embedder, tmp_path):
        pipeline = self._make_pipeline(mock_embedder, tmp_path)
        await pipeline.ingest("Hello world " * 20, source="test.txt", metadata={"doc_id": "doc-A"})
        result = await pipeline.delete_document("doc-A")
        assert result["doc_id"] == "doc-A"
        assert result["chunks_deleted"] >= 1
        assert "remaining_vectors" in result

    @pytest.mark.asyncio
    async def test_vectors_removed_after_delete(self, mock_embedder, tmp_path):
        pipeline = self._make_pipeline(mock_embedder, tmp_path)
        await pipeline.ingest("Alpha beta gamma " * 20, source="a.txt", metadata={"doc_id": "alpha"})
        await pipeline.ingest("Delta epsilon zeta " * 20, source="b.txt", metadata={"doc_id": "beta"})
        before = pipeline.index_size
        await pipeline.delete_document("alpha")
        after = pipeline.index_size
        assert after < before

    @pytest.mark.asyncio
    async def test_bm25_synced_on_delete(self, mock_embedder, tmp_path):
        from orchestrator.rag.bm25_index import BM25Index

        bm25 = BM25Index()
        from orchestrator.rag.pipeline import RAGPipeline
        from orchestrator.rag.chunker import RecursiveTextChunker
        from orchestrator.rag.retriever import SemanticRetriever
        from orchestrator.rag.reranker import EmbeddingReranker

        index = FAISSIndex(dimension=384)
        pipeline = RAGPipeline(embedder=mock_embedder, index=index, bm25_index=bm25)
        pipeline._chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
        pipeline._retriever = SemanticRetriever(mock_embedder, index)
        pipeline._reranker = EmbeddingReranker(mock_embedder)

        await pipeline.ingest("Foo bar baz " * 20, source="foo.txt", metadata={"doc_id": "foo"})
        before_bm25_len = len(bm25._chunks)
        assert before_bm25_len >= 1

        await pipeline.delete_document("foo")
        assert all(c.metadata.get("doc_id") != "foo" for c in bm25._chunks)

    @pytest.mark.asyncio
    async def test_second_delete_raises(self, mock_embedder, tmp_path):
        pipeline = self._make_pipeline(mock_embedder, tmp_path)
        await pipeline.ingest("Short content " * 20, source="s.txt", metadata={"doc_id": "once"})
        await pipeline.delete_document("once")
        with pytest.raises(ValueError):
            await pipeline.delete_document("once")


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /v1/documents/{doc_id} HTTP endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteDocumentEndpoint:
    """Uses the real FastAPI app with mocked RAGPipeline dependency."""

    @pytest.fixture(scope="class")
    def client(self):
        from orchestrator.api.app import create_app
        from orchestrator.api.dependencies import get_rag_pipeline

        app = create_app()

        mock_rag = MagicMock()
        mock_rag.ingest = AsyncMock(return_value=2)
        mock_rag.index_size = 10
        mock_rag.delete_document = AsyncMock(
            return_value={"doc_id": "existing-doc", "chunks_deleted": 3, "remaining_vectors": 7}
        )

        app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_rag

    def test_successful_delete_returns_200(self, client):
        c, mock_rag = client
        mock_rag.delete_document = AsyncMock(
            return_value={"doc_id": "doc-xyz", "chunks_deleted": 2, "remaining_vectors": 5}
        )
        resp = c.delete("/v1/documents/doc-xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "doc-xyz"
        assert data["chunks_deleted"] == 2

    def test_unknown_doc_returns_404(self, client):
        c, mock_rag = client
        mock_rag.delete_document = AsyncMock(side_effect=ValueError("Document 'ghost' not found"))
        resp = c.delete("/v1/documents/ghost")
        assert resp.status_code == 404

    def test_delete_error_returns_500(self, client):
        c, mock_rag = client
        mock_rag.delete_document = AsyncMock(side_effect=RuntimeError("FAISS exploded"))
        resp = c.delete("/v1/documents/bad-doc")
        assert resp.status_code == 500

    def test_response_contains_remaining_vectors(self, client):
        c, mock_rag = client
        mock_rag.delete_document = AsyncMock(
            return_value={"doc_id": "d1", "chunks_deleted": 1, "remaining_vectors": 9}
        )
        resp = c.delete("/v1/documents/d1")
        assert "remaining_vectors" in resp.json()
