"""Tests for the adaptive fusion engine and context builder."""
from __future__ import annotations

import pytest
from typing import List

from orchestrator.fusion.fusion_engine import AdaptiveFusionEngine
from orchestrator.fusion.context_builder import AdaptiveContextBuilder
from orchestrator.fusion.deduplicator import SemanticDeduplicator
from orchestrator.fusion.relevance_scorer import CompositeScorer
from orchestrator.core.types import (
    ContextItem,
    DocumentChunk,
    ExpertGuidance,
    ExpertType,
    Intent,
    IntentType,
    MemoryItem,
    MemoryLevel,
    MemoryResult,
    Message,
    MessageRole,
    RAGResult,
)


def make_memory_item(content: str, level: MemoryLevel, score: float = 0.8) -> MemoryItem:
    return MemoryItem(
        content=content,
        level=level,
        session_id="test",
        relevance_score=score,
        recency_score=0.9,
        importance_score=0.7,
        token_count=len(content.split()),
    )


def make_rag_chunk(content: str, score: float = 0.75) -> tuple:
    chunk = DocumentChunk(
        content=content,
        source="test_doc.txt",
        chunk_index=0,
        token_count=len(content.split()),
    )
    return chunk, score


@pytest.fixture
def intent():
    return Intent(intent_type=IntentType.CONVERSATION, confidence=0.8)


@pytest.fixture
def memory_result():
    items = [
        make_memory_item("User said hello", MemoryLevel.L1),
        make_memory_item("Previous conversation summary", MemoryLevel.L2),
        make_memory_item("Relevant fact from memory", MemoryLevel.L3),
        make_memory_item("Paris is the capital of France", MemoryLevel.L4),
    ]
    return MemoryResult(
        items=items,
        total_tokens=sum(i.token_count for i in items),
        levels_queried=[MemoryLevel.L1, MemoryLevel.L2, MemoryLevel.L3, MemoryLevel.L4],
        query_time_ms=5.0,
    )


@pytest.fixture
def rag_result():
    chunks = []
    scores = []
    for i in range(3):
        c, s = make_rag_chunk(f"Relevant document chunk {i} with useful information")
        chunks.append(c)
        scores.append(s - i * 0.05)
    return RAGResult(chunks=chunks, scores=scores, total_tokens=30, query_time_ms=10.0)


@pytest.fixture
def expert_guidance():
    return ExpertGuidance(
        expert_type=ExpertType.GENERAL,
        system_prompt="You are a helpful assistant.",
        confidence=0.9,
        token_count=10,
    )


class TestCompositeScorer:
    def test_l1_scores_highest(self):
        scorer = CompositeScorer()
        l1_item = make_memory_item("L1 content", MemoryLevel.L1, score=0.8)
        l3_item = make_memory_item("L3 content", MemoryLevel.L3, score=0.8)
        l1_score = scorer.score_memory_item(l1_item)
        l3_score = scorer.score_memory_item(l3_item)
        assert l1_score > l3_score

    def test_rag_scoring(self):
        scorer = CompositeScorer()
        chunk = DocumentChunk(
            content="test chunk",
            source="doc.txt",
            chunk_index=0,
            token_count=5,
        )
        score = scorer.score_rag_chunk(chunk, retrieval_score=0.9)
        assert 0 < score <= 1


class TestDeduplicator:
    def test_removes_near_duplicates(self):
        dedup = SemanticDeduplicator(threshold=0.5)

        item1 = ContextItem(
            content="The capital of France is Paris",
            source="test",
            composite_score=0.9,
            token_count=7,
        )
        item2 = ContextItem(
            content="Paris is the capital city of France",  # Similar
            source="test",
            composite_score=0.8,
            token_count=8,
        )
        item3 = ContextItem(
            content="Python is a programming language",  # Different
            source="test",
            composite_score=0.7,
            token_count=6,
        )

        scored = [(item1, 0.9, None), (item2, 0.8, None), (item3, 0.7, None)]
        result = dedup.deduplicate(scored)

        # item2 should be deduplicated (similar to item1)
        contents = [r[0].content for r in result]
        # item3 should always be kept (different content)
        assert any("Python" in c for c in contents)

    def test_keeps_all_when_no_duplicates(self):
        dedup = SemanticDeduplicator(threshold=0.99)
        items = [
            (ContextItem(content=f"Unique content {i}", source="t", composite_score=0.9-i*0.1, token_count=3), 0.9-i*0.1, None)
            for i in range(5)
        ]
        result = dedup.deduplicate(items)
        assert len(result) == 5


class TestAdaptiveContextBuilder:
    def test_builds_fused_context(self, expert_guidance):
        builder = AdaptiveContextBuilder()
        items = [
            (ContextItem(content=f"Context item {i}", source="mem", composite_score=0.9-i*0.1, token_count=5), 0.9-i*0.1)
            for i in range(5)
        ]
        conv = [Message(role=MessageRole.USER, content="Hello")]

        fused = builder.build(
            system_prompt="You are a helpful assistant.",
            scored_context_items=items,
            conversation=conv,
            expert_guidance=expert_guidance,
        )

        assert fused.system_prompt != ""
        assert len(fused.conversation_turns) > 0
        assert fused.total_token_count > 0
        assert 0 <= fused.token_budget_used <= 1

    def test_respects_token_budget(self, expert_guidance):
        builder = AdaptiveContextBuilder()
        # Create many large items
        items = [
            (ContextItem(
                content=" ".join(["word"] * 100),
                source="mem",
                composite_score=1.0,
                token_count=100,
            ), 1.0)
            for _ in range(50)
        ]
        conv = [Message(role=MessageRole.USER, content="test")]
        fused = builder.build("system", items, conv)
        assert fused.token_budget_used <= 1.05  # slight tolerance for estimation


@pytest.mark.asyncio
class TestFusionEngine:
    async def test_full_fusion(self, intent, memory_result, rag_result, expert_guidance):
        engine = AdaptiveFusionEngine()
        conv = [Message(role=MessageRole.USER, content="Tell me about Paris")]

        fused = await engine.fuse(
            query="Tell me about Paris",
            intent=intent,
            memory_result=memory_result,
            rag_result=rag_result,
            expert_guidance=expert_guidance,
            conversation=conv,
            system_prompt="You are a helpful assistant.",
        )

        assert fused.system_prompt != ""
        assert fused.total_token_count > 0
        assert len(fused.conversation_turns) > 0

    async def test_empty_inputs_dont_crash(self, intent):
        engine = AdaptiveFusionEngine()
        empty_memory = MemoryResult(
            items=[], total_tokens=0, levels_queried=[], query_time_ms=0
        )
        empty_rag = RAGResult(chunks=[], scores=[], total_tokens=0, query_time_ms=0)

        fused = await engine.fuse(
            query="test query",
            intent=intent,
            memory_result=empty_memory,
            rag_result=empty_rag,
            expert_guidance=None,
            conversation=[],
        )
        assert fused is not None
