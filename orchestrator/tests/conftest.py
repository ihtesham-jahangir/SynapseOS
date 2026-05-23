"""
Shared pytest fixtures for the SynapseOS test suite.

Provides:
  - mock_embedder: BGEEmbedder stub returning deterministic random vectors
  - mock_llama_client: LlamaClient stub returning canned responses
  - tmp_rag_pipeline: RAGPipeline backed by mock embedder + temp FAISS index
  - sample_intent / sample_fused_context: pre-built domain objects for reuse
"""
from __future__ import annotations

import asyncio
import tempfile
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from orchestrator.core.types import (
    ContextItem,
    DocumentChunk,
    ExpertGuidance,
    ExpertType,
    FusedContext,
    GenerationParams,
    Intent,
    IntentType,
    MemoryItem,
    MemoryLevel,
    MemoryResult,
    Message,
    MessageRole,
    RAGResult,
)
from orchestrator.rag.indexer import FAISSIndex
from orchestrator.rag.pipeline import RAGPipeline


# ── Embedding mock ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    """
    Drop-in BGEEmbedder that returns deterministic unit vectors.
    No model weights are loaded – safe to use in any unit test.
    """
    embedder = MagicMock()
    embedder.dimension = 384

    _settings = MagicMock()
    _settings.query_instruction = (
        "Represent this sentence for searching relevant passages: "
    )
    _settings.batch_size = 32
    _settings.normalize = True
    _settings.device = "cpu"
    embedder._settings = _settings

    def _vec(text: str) -> List[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(384).astype(np.float32)
        return (v / (np.linalg.norm(v) + 1e-10)).tolist()

    async def embed(texts: List[str]) -> List[List[float]]:
        return [_vec(t) for t in texts]

    async def embed_query(query: str) -> List[float]:
        return _vec(f"query:{query}")

    async def embed_single(text: str) -> List[float]:
        return _vec(text)

    embedder.embed = embed
    embedder.embed_query = embed_query
    embedder.embed_single = embed_single
    return embedder


# ── LlamaClient mock ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_llama_client():
    """LlamaClient that returns canned text without hitting any server."""
    client = MagicMock()
    client.chat = AsyncMock(return_value="This is a mock LLM response.")
    client.health_check = AsyncMock(return_value=True)
    client.close = AsyncMock()

    async def _stream(*args, **kwargs) -> AsyncGenerator[str, None]:
        for token in ["Hello", " from", " mock", " streamer"]:
            yield token

    client.stream_chat = _stream
    return client


# ── RAG pipeline with temp index ─────────────────────────────────────────────

@pytest.fixture
def tmp_rag_pipeline(mock_embedder, tmp_path):
    """
    RAGPipeline with mock embedder and a temporary FAISS index.
    Uses a small chunk_size (50 tokens) so tests can create multiple chunks
    without massive input text.
    """
    from orchestrator.rag.chunker import RecursiveTextChunker
    from orchestrator.rag.retriever import SemanticRetriever
    from orchestrator.rag.reranker import EmbeddingReranker

    index = FAISSIndex(dimension=384, index_path=str(tmp_path / "test_faiss"))
    pipeline = RAGPipeline(embedder=mock_embedder, index=index)
    # Override chunker with a small chunk size so tests don't need huge documents
    pipeline._chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
    pipeline._retriever = SemanticRetriever(mock_embedder, index)
    pipeline._reranker = EmbeddingReranker(mock_embedder)
    return pipeline


# ── Common domain objects ─────────────────────────────────────────────────────

@pytest.fixture
def coding_intent() -> Intent:
    return Intent(
        intent_type=IntentType.CODING,
        confidence=0.92,
        requires_expert=True,
    )


@pytest.fixture
def conversation_intent() -> Intent:
    return Intent(intent_type=IntentType.CONVERSATION, confidence=0.80)


@pytest.fixture
def math_intent() -> Intent:
    return Intent(intent_type=IntentType.MATH, confidence=0.95, requires_expert=True)


@pytest.fixture
def empty_memory_result() -> MemoryResult:
    return MemoryResult(items=[], total_tokens=0, levels_queried=[], query_time_ms=0.0)


@pytest.fixture
def empty_rag_result() -> RAGResult:
    return RAGResult(chunks=[], scores=[], total_tokens=0, query_time_ms=0.0)


@pytest.fixture
def sample_fused_context() -> FusedContext:
    return FusedContext(
        system_prompt="You are a helpful AI assistant.",
        context_items=[
            ContextItem(
                content="Python was created by Guido van Rossum.",
                source="knowledge",
                composite_score=0.9,
                token_count=8,
            )
        ],
        conversation_turns=[
            Message(role=MessageRole.USER, content="What is Python?")
        ],
        total_token_count=50,
        token_budget_used=0.05,
    )


@pytest.fixture
def sample_gen_params() -> GenerationParams:
    return GenerationParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
    )
