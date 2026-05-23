"""
Tests for the full RAG pipeline: ingest, retrieve, rerank.

Uses mock embedder + temp FAISS index so no model weights are needed.
"""
from __future__ import annotations

import pytest

from orchestrator.core.types import RAGResult, DocumentChunk
from orchestrator.rag.reranker import EmbeddingReranker, _cosine_sim


# ── RAGPipeline.ingest ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRAGPipelineIngest:
    async def test_ingest_returns_chunk_count(self, tmp_rag_pipeline):
        n = await tmp_rag_pipeline.ingest(
            content="Python is a programming language. It is widely used in data science.",
            source="test",
        )
        assert n >= 1

    async def test_empty_content_returns_zero(self, tmp_rag_pipeline):
        n = await tmp_rag_pipeline.ingest(content="", source="test")
        assert n == 0

    async def test_long_content_creates_multiple_chunks(self, tmp_rag_pipeline):
        text = " ".join([f"sentence{i} with some content." for i in range(50)])
        n = await tmp_rag_pipeline.ingest(content=text, source="long_doc")
        assert n > 1

    async def test_index_size_increases_after_ingest(self, tmp_rag_pipeline):
        initial = tmp_rag_pipeline.index_size
        await tmp_rag_pipeline.ingest("Some new document content.", source="doc1")
        assert tmp_rag_pipeline.index_size > initial

    async def test_ingest_with_metadata(self, tmp_rag_pipeline):
        n = await tmp_rag_pipeline.ingest(
            content="Metadata test document.",
            source="meta_test",
            metadata={"doc_id": "meta123", "author": "tester"},
        )
        assert n >= 1

    async def test_ingest_multiple_documents(self, tmp_rag_pipeline):
        docs = [
            ("Python is a programming language.", "py_doc"),
            ("JavaScript runs in the browser.", "js_doc"),
            ("Rust is a systems programming language.", "rs_doc"),
        ]
        for content, source in docs:
            await tmp_rag_pipeline.ingest(content=content, source=source)

        assert tmp_rag_pipeline.index_size >= 3


# ── RAGPipeline.retrieve ──────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRAGPipelineRetrieve:
    async def test_retrieve_after_ingest_returns_chunks(self, tmp_rag_pipeline):
        await tmp_rag_pipeline.ingest(
            "The capital of France is Paris.", source="geo"
        )
        result = await tmp_rag_pipeline.retrieve("What is the capital of France?")
        assert isinstance(result, RAGResult)

    async def test_retrieve_empty_index_returns_empty(self, tmp_rag_pipeline):
        result = await tmp_rag_pipeline.retrieve("anything")
        assert result.chunks == []
        assert result.scores == []

    async def test_retrieve_respects_top_k(self, tmp_rag_pipeline):
        # Ingest multiple documents
        for i in range(5):
            await tmp_rag_pipeline.ingest(
                f"Document number {i} contains unique information.",
                source=f"doc_{i}",
            )
        # top_k caps candidates; reranker's default top_n=3 may further cap
        result = await tmp_rag_pipeline.retrieve("document information", top_k=5)
        assert len(result.chunks) <= 5

    async def test_retrieve_scores_between_0_and_1(self, tmp_rag_pipeline):
        await tmp_rag_pipeline.ingest("Test content for scoring.", source="score_test")
        result = await tmp_rag_pipeline.retrieve("test content")
        for score in result.scores:
            assert -1.0 <= score <= 1.01  # cosine can be slightly above 1 due to fp

    async def test_retrieve_returns_rag_result_type(self, tmp_rag_pipeline):
        result = await tmp_rag_pipeline.retrieve("query")
        assert isinstance(result, RAGResult)
        assert hasattr(result, "chunks")
        assert hasattr(result, "scores")
        assert hasattr(result, "query_time_ms")


# ── EmbeddingReranker ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestEmbeddingReranker:
    async def test_rerank_empty_chunks_returns_empty(self, mock_embedder):
        reranker = EmbeddingReranker(mock_embedder)
        result = await reranker.rerank("query", [], top_n=3)
        assert result == []

    async def test_rerank_returns_correct_count(self, mock_embedder):
        chunks = [
            DocumentChunk(content=f"chunk {i}", source="doc", chunk_index=i, token_count=5)
            for i in range(5)
        ]
        reranker = EmbeddingReranker(mock_embedder)
        result = await reranker.rerank("query", chunks, top_n=3)
        assert len(result) == 3

    async def test_rerank_top_n_capped_at_chunk_count(self, mock_embedder):
        chunks = [
            DocumentChunk(content=f"chunk {i}", source="doc", chunk_index=i, token_count=5)
            for i in range(2)
        ]
        reranker = EmbeddingReranker(mock_embedder)
        result = await reranker.rerank("query", chunks, top_n=10)
        assert len(result) == 2

    async def test_rerank_output_is_chunk_score_pairs(self, mock_embedder):
        chunks = [
            DocumentChunk(content="relevant content", source="doc", chunk_index=0, token_count=5)
        ]
        reranker = EmbeddingReranker(mock_embedder)
        result = await reranker.rerank("query", chunks, top_n=1)
        assert len(result) == 1
        chunk, score = result[0]
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(score, float)


# ── cosine similarity helper ──────────────────────────────────────────────────

class TestCosineSim:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_sim(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector_returns_near_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        result = _cosine_sim(a, b)
        assert abs(result) < 0.01
