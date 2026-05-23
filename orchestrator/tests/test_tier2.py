"""
Tests for Tier 2 architecture upgrades:
  - BM25Index
  - HybridRetriever
  - CrossEncoderReranker
  - SemanticChunker
  - Speculative decoding wiring in InferenceEngine
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from orchestrator.core.types import (
    DocumentChunk,
    ExpertType,
    FusedContext,
    GenerationParams,
    Intent,
    IntentType,
    MemoryResult,
    Message,
    MessageRole,
    RAGResult,
    ContextItem,
)
from orchestrator.rag.bm25_index import BM25Index
from orchestrator.rag.chunker import SemanticChunker, RecursiveTextChunker
from orchestrator.rag.retriever import HybridRetriever, SemanticRetriever
from orchestrator.rag.reranker import CrossEncoderReranker, EmbeddingReranker
from orchestrator.rag.indexer import FAISSIndex
from orchestrator.rag.pipeline import RAGPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(content: str, idx: int = 0, source: str = "test") -> DocumentChunk:
    return DocumentChunk(
        id=f"chunk_{idx}",
        content=content,
        source=source,
        chunk_index=idx,
        token_count=len(content.split()),
    )


def _mock_embedder():
    """Deterministic embedder that returns unit vectors seeded by text hash."""
    embedder = MagicMock()
    embedder.dimension = 384
    _s = MagicMock()
    _s.query_instruction = "Represent this sentence for searching relevant passages: "
    _s.batch_size = 32
    _s.normalize = True
    _s.device = "cpu"
    embedder._settings = _s

    def _vec(text: str) -> List[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(384).astype(np.float32)
        return (v / (np.linalg.norm(v) + 1e-10)).tolist()

    async def embed(texts):
        return [_vec(t) for t in texts]

    async def embed_query(query):
        return _vec(f"query:{query}")

    embedder.embed = embed
    embedder.embed_query = embed_query
    return embedder


# ══════════════════════════════════════════════════════════════════════════════
# BM25Index
# ══════════════════════════════════════════════════════════════════════════════

class TestBM25Index:
    def test_empty_search_returns_empty(self):
        idx = BM25Index()
        result = idx.search("anything")
        assert result == []

    def test_add_and_search(self):
        idx = BM25Index()
        chunks = [
            _make_chunk("Python is a programming language", 0),
            _make_chunk("Machine learning uses neural networks", 1),
            _make_chunk("FastAPI is a web framework", 2),
        ]
        idx.add(chunks)
        results = idx.search("Python programming", top_k=3)
        assert len(results) >= 1
        contents = [c.content for c, _ in results]
        assert any("Python" in c for c in contents)

    def test_search_returns_scores(self):
        idx = BM25Index()
        idx.add([_make_chunk("hello world python", 0)])
        results = idx.search("python")
        assert results
        _, score = results[0]
        assert score > 0.0

    def test_zero_score_results_excluded(self):
        idx = BM25Index()
        idx.add([_make_chunk("cats and dogs", 0)])
        results = idx.search("zzzznotaword")
        assert results == []

    def test_top_k_respected(self):
        idx = BM25Index()
        for i in range(10):
            idx.add([_make_chunk(f"document about python topic {i}", i)])
        results = idx.search("python", top_k=3)
        assert len(results) <= 3

    def test_total_docs(self):
        idx = BM25Index()
        assert idx.total_docs == 0
        idx.add([_make_chunk("a", 0), _make_chunk("b", 1)])
        assert idx.total_docs == 2

    def test_persist_and_load(self, tmp_path):
        path = str(tmp_path / "bm25.json")
        idx = BM25Index(persist_path=path)
        idx.add([
            _make_chunk("machine learning", 0),
            _make_chunk("deep neural networks", 1),
        ])
        idx.persist()
        assert Path(path).exists()

        # Load into a new instance
        idx2 = BM25Index(persist_path=path)
        idx2.load()
        assert idx2.total_docs == 2
        results = idx2.search("machine learning")
        assert len(results) >= 1

    def test_load_missing_file_is_safe(self, tmp_path):
        idx = BM25Index(persist_path=str(tmp_path / "nonexistent.json"))
        idx.load()  # should not raise
        assert idx.total_docs == 0

    def test_persist_without_path_is_safe(self):
        idx = BM25Index()  # no persist_path
        idx.add([_make_chunk("hello", 0)])
        idx.persist()  # should not raise

    def test_load_corrupt_file_is_handled(self, tmp_path):
        path = tmp_path / "bm25.json"
        path.write_text("not valid json", encoding="utf-8")
        idx = BM25Index(persist_path=str(path))
        idx.load()  # should not raise, just log warning
        assert idx.total_docs == 0

    def test_multiple_add_calls_accumulate(self):
        idx = BM25Index()
        idx.add([_make_chunk("first document", 0)])
        idx.add([_make_chunk("second document", 1)])
        assert idx.total_docs == 2

    def test_search_ranking_order(self):
        idx = BM25Index()
        idx.add([
            _make_chunk("python python python", 0),
            _make_chunk("python once", 1),
            _make_chunk("java java java", 2),
        ])
        results = idx.search("python", top_k=3)
        assert len(results) >= 2
        # Highest score first
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# HybridRetriever
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridRetriever:
    @pytest.fixture
    def tmp_faiss(self, tmp_path):
        return FAISSIndex(dimension=384, index_path=str(tmp_path / "faiss"))

    @pytest.fixture
    def embedder(self):
        return _mock_embedder()

    @pytest.mark.asyncio
    async def test_returns_rag_result(self, tmp_faiss, embedder):
        bm25 = BM25Index()
        semantic = SemanticRetriever(embedder, tmp_faiss)
        retriever = HybridRetriever(semantic, bm25)
        result = await retriever.retrieve("python")
        assert isinstance(result, RAGResult)

    @pytest.mark.asyncio
    async def test_empty_index_returns_empty(self, tmp_faiss, embedder):
        bm25 = BM25Index()
        semantic = SemanticRetriever(embedder, tmp_faiss)
        retriever = HybridRetriever(semantic, bm25)
        result = await retriever.retrieve("anything")
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_hybrid_merges_both_sources(self, embedder, tmp_path):
        # Use a fresh FAISS index to guarantee deterministic state
        faiss_idx = FAISSIndex(dimension=384, index_path=str(tmp_path / "hybrid_faiss"))

        chunk = _make_chunk("unique_keyword_xyz is a special term", 0)
        bm25 = BM25Index()
        bm25.add([chunk])

        # Index in FAISS as well
        vec = await embedder.embed([chunk.content])
        await faiss_idx.add(vec, [{"chunk_id": chunk.id, "content": chunk.content,
                                   "source": "test", "chunk_index": 0}])

        # Use threshold=0.0 so FAISS returns the chunk even with low similarity
        semantic = SemanticRetriever(embedder, faiss_idx)
        retriever = HybridRetriever(semantic, bm25, alpha=0.5)
        # Retrieve with no similarity threshold — rely on RRF from BM25 side
        result = await retriever.retrieve("unique_keyword_xyz", top_k=5, threshold=0.0)
        # At minimum BM25 side should contribute this chunk
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_top_k_respected(self, tmp_faiss, embedder):
        bm25 = BM25Index()
        # Add 5 chunks
        for i in range(5):
            bm25.add([_make_chunk(f"document about testing {i}", i)])
        semantic = SemanticRetriever(embedder, tmp_faiss)
        retriever = HybridRetriever(semantic, bm25)
        result = await retriever.retrieve("testing", top_k=2)
        assert len(result.chunks) <= 2

    def test_rrf_merge_deduplicates(self, tmp_faiss, embedder):
        bm25 = BM25Index()
        semantic = SemanticRetriever(embedder, tmp_faiss)
        retriever = HybridRetriever(semantic, bm25)

        chunk_a = _make_chunk("shared chunk", idx=0)
        chunk_b = _make_chunk("unique chunk", idx=1)
        chunk_a_dup = _make_chunk("shared chunk", idx=0)  # same id
        chunk_a_dup.id = chunk_a.id

        merged = retriever._rrf_merge([chunk_a, chunk_b], [chunk_a_dup])
        ids = [c.id for c in merged]
        assert len(ids) == len(set(ids)), "Duplicate chunks should be deduplicated"

    def test_alpha_pure_semantic(self, tmp_faiss, embedder):
        bm25 = BM25Index()
        semantic = SemanticRetriever(embedder, tmp_faiss)
        retriever = HybridRetriever(semantic, bm25, alpha=1.0)
        # Only semantic contributes; BM25 results ignored
        chunk_a = _make_chunk("semantic result", idx=0)
        chunk_b = _make_chunk("bm25 result", idx=1)
        merged = retriever._rrf_merge([chunk_a], [chunk_b])
        assert merged[0].id == chunk_a.id

    def test_alpha_pure_keyword(self, tmp_faiss, embedder):
        bm25 = BM25Index()
        semantic = SemanticRetriever(embedder, tmp_faiss)
        retriever = HybridRetriever(semantic, bm25, alpha=0.0)
        chunk_a = _make_chunk("semantic result", idx=0)
        chunk_b = _make_chunk("bm25 result", idx=1)
        merged = retriever._rrf_merge([chunk_a], [chunk_b])
        assert merged[0].id == chunk_b.id


# ══════════════════════════════════════════════════════════════════════════════
# CrossEncoderReranker
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self):
        reranker = CrossEncoderReranker()
        result = await reranker.rerank("query", [], top_n=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_reranks_with_mock_model(self):
        reranker = CrossEncoderReranker()
        # Inject a mock model so no download happens
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.9, 0.2, 0.6]))
        reranker._model = mock_model

        chunks = [
            _make_chunk("highly relevant document", 0),
            _make_chunk("unrelated content", 1),
            _make_chunk("somewhat relevant", 2),
        ]
        result = await reranker.rerank("query", chunks, top_n=3)
        assert len(result) == 3
        # Highest score (0.9 → chunk 0) should be first
        assert result[0][0].id == "chunk_0"

    @pytest.mark.asyncio
    async def test_top_n_limits_output(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.8, 0.5, 0.3, 0.1]))
        reranker._model = mock_model

        chunks = [_make_chunk(f"chunk {i}", i) for i in range(4)]
        result = await reranker.rerank("query", chunks, top_n=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_top_n_capped_at_chunk_count(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.5, 0.3]))
        reranker._model = mock_model

        chunks = [_make_chunk("a", 0), _make_chunk("b", 1)]
        result = await reranker.rerank("query", chunks, top_n=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_output_is_chunk_score_pairs(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.7]))
        reranker._model = mock_model

        chunks = [_make_chunk("content", 0)]
        result = await reranker.rerank("query", chunks, top_n=1)
        assert isinstance(result[0][0], DocumentChunk)
        assert isinstance(result[0][1], float)

    @pytest.mark.asyncio
    async def test_scores_are_descending(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.3, 0.9, 0.1, 0.7]))
        reranker._model = mock_model

        chunks = [_make_chunk(f"doc {i}", i) for i in range(4)]
        result = await reranker.rerank("query", chunks, top_n=4)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_lazy_load_skipped_when_model_present(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.5]))
        reranker._model = mock_model  # pre-inject

        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            chunks = [_make_chunk("test", 0)]
            await reranker.rerank("q", chunks, top_n=1)
            mock_ce.assert_not_called()  # should not reload


# ══════════════════════════════════════════════════════════════════════════════
# SemanticChunker
# ══════════════════════════════════════════════════════════════════════════════

class TestSemanticChunker:
    @pytest.fixture
    def embedder(self):
        return _mock_embedder()

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self, embedder):
        chunker = SemanticChunker(embedder)
        result = await chunker.chunk("")
        assert result == []

    @pytest.mark.asyncio
    async def test_single_sentence_returns_one_chunk(self, embedder):
        chunker = SemanticChunker(embedder)
        result = await chunker.chunk("This is a single sentence.")
        assert len(result) == 1
        assert result[0].content.strip() != ""

    @pytest.mark.asyncio
    async def test_chunk_index_is_sequential(self, embedder):
        chunker = SemanticChunker(embedder, breakpoint_threshold=0.99)
        text = " ".join(f"Sentence number {i}." for i in range(10))
        result = await chunker.chunk(text)
        assert [c.chunk_index for c in result] == list(range(len(result)))

    @pytest.mark.asyncio
    async def test_content_preserved(self, embedder):
        chunker = SemanticChunker(embedder)
        text = "Machine learning is powerful. Python is popular."
        result = await chunker.chunk(text)
        all_content = " ".join(c.content for c in result)
        # All words should be present somewhere in the chunks
        assert "Machine" in all_content
        assert "Python" in all_content

    @pytest.mark.asyncio
    async def test_threshold_zero_produces_many_chunks(self, embedder):
        chunker = SemanticChunker(embedder, breakpoint_threshold=0.0)
        sentences = [f"Topic {i} is about subject {i}." for i in range(5)]
        text = " ".join(sentences)
        result = await chunker.chunk(text)
        # threshold=0 means every adjacent pair triggers a split
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_threshold_never_fires_gives_one_chunk(self, embedder):
        # Cosine similarity is bounded to [-1, 1]; threshold=-2.0 never fires.
        chunker = SemanticChunker(embedder, breakpoint_threshold=-2.0)
        text = "First sentence. Second sentence. Third sentence."
        result = await chunker.chunk(text)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_oversized_group_is_split(self, embedder):
        chunker = SemanticChunker(embedder, max_chunk_tokens=3)
        # Provide text that will form one big group but exceed 3 tokens
        text = "word1 word2 word3 word4 word5 word6 word7."
        result = await chunker.chunk(text)
        for c in result:
            assert c.token_count <= 6  # tolerance for subword tokenisation

    @pytest.mark.asyncio
    async def test_token_count_set(self, embedder):
        chunker = SemanticChunker(embedder)
        result = await chunker.chunk("Hello world. This is fine.")
        for c in result:
            assert c.token_count > 0

    @pytest.mark.asyncio
    async def test_returns_chunk_objects(self, embedder):
        from orchestrator.rag.chunker import Chunk
        chunker = SemanticChunker(embedder)
        result = await chunker.chunk("One sentence. Two sentences.")
        for c in result:
            assert isinstance(c, Chunk)


# ══════════════════════════════════════════════════════════════════════════════
# RAGPipeline with injected components
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGPipelineInjection:
    @pytest.fixture
    def embedder(self):
        return _mock_embedder()

    @pytest.fixture
    def tmp_index(self, tmp_path):
        return FAISSIndex(dimension=384, index_path=str(tmp_path / "faiss"))

    @pytest.mark.asyncio
    async def test_semantic_chunker_used_during_ingest(self, embedder, tmp_index):
        chunker = SemanticChunker(embedder)
        pipeline = RAGPipeline(embedder=embedder, index=tmp_index, chunker=chunker)
        count = await pipeline.ingest("Hello world. This is a test.")
        assert count >= 1

    @pytest.mark.asyncio
    async def test_bm25_updated_during_ingest(self, embedder, tmp_index):
        bm25 = BM25Index()
        pipeline = RAGPipeline(embedder=embedder, index=tmp_index, bm25_index=bm25)
        await pipeline.ingest("python programming language", source="test")
        assert bm25.total_docs >= 1

    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_used(self, embedder, tmp_index):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.8]))
        reranker._model = mock_model

        pipeline = RAGPipeline(embedder=embedder, index=tmp_index, reranker=reranker)
        await pipeline.ingest("python is great", source="test")
        result = await pipeline.retrieve("python")
        assert isinstance(result, RAGResult)

    @pytest.mark.asyncio
    async def test_hybrid_retriever_used(self, embedder, tmp_index):
        bm25 = BM25Index()
        semantic = SemanticRetriever(embedder, tmp_index)
        hybrid = HybridRetriever(semantic, bm25)
        pipeline = RAGPipeline(
            embedder=embedder, index=tmp_index,
            retriever=hybrid, bm25_index=bm25
        )
        await pipeline.ingest("machine learning deep learning", source="test")
        result = await pipeline.retrieve("machine learning")
        assert isinstance(result, RAGResult)

    @pytest.mark.asyncio
    async def test_bm25_persist_called_during_ingest(self, embedder, tmp_index, tmp_path):
        bm25 = BM25Index(persist_path=str(tmp_path / "bm25.json"))
        pipeline = RAGPipeline(embedder=embedder, index=tmp_index, bm25_index=bm25)
        await pipeline.ingest("test document content", source="test")
        assert Path(tmp_path / "bm25.json").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Speculative decoding in InferenceEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestSpeculativeDecodingWiring:
    """Verify InferenceEngine routes through SpeculativeDecoder when set."""

    def _make_engine(self, llama_client, speculative=None):
        from orchestrator.runtime.inference_engine import InferenceEngine
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        from orchestrator.router.routing_engine import RoutingEngine
        from orchestrator.experts.expert_manager import ExpertManager
        from orchestrator.fusion.fusion_engine import AdaptiveFusionEngine
        from orchestrator.runtime.adaptive_compute import AdaptiveComputeController
        from orchestrator.memory.memory_manager import MemoryManager
        from orchestrator.memory.l1_cache import L1ConversationCache
        from orchestrator.memory.l2_cache import L2SummaryCache
        from orchestrator.memory.l3_cache import L3VectorMemory
        from orchestrator.memory.l4_cache import L4KnowledgeBase

        embedder = _mock_embedder()
        tmp_index = FAISSIndex(dimension=384, index_path="/tmp/test_spec_faiss")
        rag = RAGPipeline(embedder=embedder, index=tmp_index)

        l1 = L1ConversationCache()
        l2 = L2SummaryCache(db_path=":memory:")
        l3 = L3VectorMemory(embedder=embedder, index=FAISSIndex(
            dimension=384, index_path="/tmp/test_spec_mem_faiss"))
        l4 = L4KnowledgeBase(db_path=":memory:")
        memory = MemoryManager(l1=l1, l2=l2, l3=l3, l4=l4)

        return InferenceEngine(
            llama_client=llama_client,
            memory_manager=memory,
            rag_pipeline=rag,
            intent_classifier=HybridIntentClassifier(embedder=embedder),
            routing_engine=RoutingEngine(),
            expert_manager=ExpertManager(),
            fusion_engine=AdaptiveFusionEngine(),
            compute_controller=AdaptiveComputeController(),
            speculative_decoder=speculative,
        )

    @pytest.mark.asyncio
    async def test_no_speculative_uses_llama_client(self):
        llama = MagicMock()
        llama.chat = AsyncMock(return_value="direct llama response")
        engine = self._make_engine(llama, speculative=None)

        from orchestrator.core.types import FusedContext, GenerationParams, ContextItem
        fused = FusedContext(
            system_prompt="Be helpful.",
            context_items=[],
            conversation_turns=[Message(role=MessageRole.USER, content="hi")],
            total_token_count=10,
            token_budget_used=0.01,
        )
        params = GenerationParams(max_tokens=32, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1)
        resp = await engine.generate(fused, params)
        assert resp.content == "direct llama response"
        llama.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_speculative_decoder_uses_it(self):
        from orchestrator.runtime.speculative import SpeculativeDecoder

        llama = MagicMock()
        llama.chat = AsyncMock(return_value="should not be called")

        draft = MagicMock()
        draft.chat = AsyncMock(return_value="draft tokens")
        speculative = SpeculativeDecoder(verifier_client=llama, draft_client=draft, k_tokens=3)
        # patch _speculative_generate to return a known string
        speculative._speculative_generate = AsyncMock(return_value="speculative response")

        engine = self._make_engine(llama, speculative=speculative)

        from orchestrator.core.types import FusedContext, GenerationParams
        fused = FusedContext(
            system_prompt="Be helpful.",
            context_items=[],
            conversation_turns=[Message(role=MessageRole.USER, content="hi")],
            total_token_count=10,
            token_budget_used=0.01,
        )
        params = GenerationParams(max_tokens=32, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1)
        resp = await engine.generate(fused, params)
        assert resp.content == "speculative response"
        # llama.chat should NOT be called directly
        llama.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_speculative_none_disables_it(self):
        """When speculative_decoder=None, engine falls back to llama.chat."""
        llama = MagicMock()
        llama.chat = AsyncMock(return_value="fallback response")
        engine = self._make_engine(llama, speculative=None)
        assert engine._speculative is None
        assert engine._llama is llama
