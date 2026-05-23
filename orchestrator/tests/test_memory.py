"""Tests for the hierarchical memory system."""
from __future__ import annotations

import asyncio
import tempfile
import pytest

from orchestrator.memory.l1_cache import L1ConversationCache
from orchestrator.memory.l2_cache import L2SummaryCache
from orchestrator.memory.l4_cache import L4KnowledgeBase
from orchestrator.core.types import Message, MessageRole, MemoryItem, MemoryLevel


@pytest.fixture
def l1():
    return L1ConversationCache(max_turns=5, max_tokens=500)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def l2(tmp_db):
    return L2SummaryCache(db_path=tmp_db)


@pytest.fixture
def l4(tmp_db):
    return L4KnowledgeBase(db_path=tmp_db)


class TestL1Cache:
    @pytest.mark.asyncio
    async def test_push_and_retrieve(self, l1):
        msg = Message(role=MessageRole.USER, content="Hello, how are you?")
        await l1.push_turn("session1", msg)
        turns = await l1.get_turns("session1")
        assert len(turns) == 1
        assert turns[0].content == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_max_turns_eviction(self, l1):
        for i in range(8):  # More than max_turns=5
            msg = Message(role=MessageRole.USER, content=f"Message {i}")
            await l1.push_turn("session1", msg)
        turns = await l1.get_turns("session1")
        assert len(turns) <= 5

    @pytest.mark.asyncio
    async def test_clear_session(self, l1):
        msg = Message(role=MessageRole.USER, content="test")
        await l1.push_turn("session1", msg)
        await l1.clear_session("session1")
        turns = await l1.get_turns("session1")
        assert len(turns) == 0

    @pytest.mark.asyncio
    async def test_session_isolation(self, l1):
        await l1.push_turn("session1", Message(role=MessageRole.USER, content="S1"))
        await l1.push_turn("session2", Message(role=MessageRole.USER, content="S2"))
        s1 = await l1.get_turns("session1")
        s2 = await l1.get_turns("session2")
        assert s1[0].content == "S1"
        assert s2[0].content == "S2"

    @pytest.mark.asyncio
    async def test_retrieve_returns_memory_items(self, l1):
        await l1.push_turn("s1", Message(role=MessageRole.USER, content="test content"))
        items = await l1.retrieve("test", "s1")
        assert len(items) > 0
        assert items[0].level == MemoryLevel.L1


class TestL2Cache:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, l2):
        import uuid
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content="Summary of conversation about Python programming",
            level=MemoryLevel.L2,
            session_id="s1",
            metadata={"turn_start": 0, "turn_end": 10},
        )
        await l2.store(item)
        results = await l2.retrieve("Python", "s1")
        assert len(results) >= 1
        assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_clear_session(self, l2):
        import uuid
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content="test summary",
            level=MemoryLevel.L2,
            session_id="s1",
            metadata={"turn_start": 0, "turn_end": 5},
        )
        await l2.store(item)
        await l2.clear_session("s1")
        results = await l2.retrieve("test", "s1")
        assert len(results) == 0


class TestL4KnowledgeBase:
    @pytest.mark.asyncio
    async def test_add_and_retrieve_knowledge(self, l4):
        doc_id = await l4.add_knowledge(
            content="The capital of France is Paris",
            category="geography",
            keywords="france capital city",
        )
        assert doc_id is not None
        results = await l4.retrieve("capital France", "any_session")
        assert len(results) >= 1
        assert "Paris" in results[0].content

    @pytest.mark.asyncio
    async def test_high_priority_returned_first(self, l4):
        await l4.add_knowledge("Low priority fact", priority=0.3)
        await l4.add_knowledge("High priority fact", priority=0.95)
        results = await l4.retrieve("priority fact", "s1")
        if len(results) >= 2:
            assert results[0].importance_score >= results[1].importance_score

    @pytest.mark.asyncio
    async def test_list_categories(self, l4):
        await l4.add_knowledge("Test content", category="science")
        await l4.add_knowledge("Test content 2", category="history")
        cats = await l4.list_categories()
        assert "science" in cats
        assert "history" in cats
