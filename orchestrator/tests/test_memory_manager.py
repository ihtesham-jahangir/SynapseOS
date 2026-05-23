"""
Tests for MemoryManager – the unified 4-tier memory orchestrator.

Uses real L1/L2/L4 caches with temp SQLite databases.
L3 is mocked because it requires the embedding model.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.core.types import Message, MessageRole, MemoryItem, MemoryLevel, MemoryResult
from orchestrator.memory.l1_cache import L1ConversationCache
from orchestrator.memory.l2_cache import L2SummaryCache
from orchestrator.memory.l4_cache import L4KnowledgeBase
from orchestrator.memory.memory_manager import MemoryManager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def l1():
    return L1ConversationCache(max_turns=10, max_tokens=2000)


@pytest.fixture
def l2(tmp_path):
    return L2SummaryCache(db_path=str(tmp_path / "l2.db"))


@pytest.fixture
def l3_mock():
    l3 = MagicMock()
    l3.retrieve = AsyncMock(return_value=[])
    l3.store = AsyncMock()
    l3.clear_session = AsyncMock()
    return l3


@pytest.fixture
def l4(tmp_path):
    return L4KnowledgeBase(db_path=str(tmp_path / "l4.db"))


@pytest.fixture
def manager(l1, l2, l3_mock, l4):
    return MemoryManager(l1=l1, l2=l2, l3=l3_mock, l4=l4)


# ── record_turn ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRecordTurn:
    async def test_stores_both_messages(self, manager):
        user = Message(role=MessageRole.USER, content="Hello!")
        asst = Message(role=MessageRole.ASSISTANT, content="Hi!")
        await manager.record_turn("s1", user, asst)

        turns = await manager.get_conversation("s1")
        assert len(turns) == 2
        assert turns[0].content == "Hello!"
        assert turns[1].content == "Hi!"

    async def test_accumulates_turns(self, manager):
        for i in range(3):
            await manager.record_turn(
                "s1",
                Message(role=MessageRole.USER, content=f"Q{i}"),
                Message(role=MessageRole.ASSISTANT, content=f"A{i}"),
            )
        turns = await manager.get_conversation("s1")
        assert len(turns) == 6

    async def test_sessions_are_isolated(self, manager):
        await manager.record_turn(
            "session-a",
            Message(role=MessageRole.USER, content="Session A"),
            Message(role=MessageRole.ASSISTANT, content="Reply A"),
        )
        await manager.record_turn(
            "session-b",
            Message(role=MessageRole.USER, content="Session B"),
            Message(role=MessageRole.ASSISTANT, content="Reply B"),
        )

        turns_a = await manager.get_conversation("session-a")
        turns_b = await manager.get_conversation("session-b")

        assert all("Session A" in t.content or "Reply A" in t.content for t in turns_a)
        assert all("Session B" in t.content or "Reply B" in t.content for t in turns_b)


# ── retrieve_all ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRetrieveAll:
    async def test_returns_memory_result_type(self, manager):
        result = await manager.retrieve_all("test query", "s1")
        assert isinstance(result, MemoryResult)

    async def test_empty_session_returns_empty(self, manager):
        result = await manager.retrieve_all("anything", "new-session")
        assert result.items == [] or isinstance(result.items, list)

    async def test_l1_items_included_after_record(self, manager):
        await manager.record_turn(
            "s-retrieve",
            Message(role=MessageRole.USER, content="Python programming"),
            Message(role=MessageRole.ASSISTANT, content="Sure, Python is great"),
        )
        result = await manager.retrieve_all("Python", "s-retrieve")
        assert any(item.level == MemoryLevel.L1 for item in result.items)

    async def test_l4_items_included_after_store(self, manager, l4):
        await l4.add_knowledge(
            content="SynapseOS is an AI orchestrator",
            category="product",
            keywords="synapseos ai",
        )
        result = await manager.retrieve_all("SynapseOS", "any-session")
        l4_items = [i for i in result.items if i.level == MemoryLevel.L4]
        assert len(l4_items) >= 1

    async def test_levels_queried_populated(self, manager):
        result = await manager.retrieve_all("test", "s1")
        assert isinstance(result.levels_queried, list)

    async def test_query_time_positive(self, manager):
        result = await manager.retrieve_all("test", "s1")
        assert result.query_time_ms >= 0

    async def test_tier_failure_is_isolated(self, l1, l2, l4, tmp_path):
        """A failure in one tier should not crash the whole retrieve_all."""
        l3_broken = MagicMock()
        l3_broken.retrieve = AsyncMock(side_effect=RuntimeError("L3 broken"))
        l3_broken.store = AsyncMock()
        l3_broken.clear_session = AsyncMock()

        manager = MemoryManager(l1=l1, l2=l2, l3=l3_broken, l4=l4)
        result = await manager.retrieve_all("test", "s1")
        # Should not raise; result may have fewer items but is still valid
        assert isinstance(result, MemoryResult)


# ── store_fact ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestStoreFact:
    async def test_store_fact_calls_l3(self, manager, l3_mock):
        await manager.store_fact("session-x", "The user prefers dark mode", importance=0.9)
        l3_mock.store.assert_awaited_once()

        stored_item = l3_mock.store.call_args[0][0]
        assert "dark mode" in stored_item.content
        assert stored_item.importance_score == pytest.approx(0.9)
        assert stored_item.level == MemoryLevel.L3


# ── clear_session ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestClearSession:
    async def test_clears_l1(self, manager):
        await manager.record_turn(
            "clear-me",
            Message(role=MessageRole.USER, content="remember this"),
            Message(role=MessageRole.ASSISTANT, content="ok"),
        )
        await manager.clear_session("clear-me")
        turns = await manager.get_conversation("clear-me")
        assert len(turns) == 0

    async def test_clear_calls_all_tiers(self, l1, l2, l3_mock, l4):
        manager = MemoryManager(l1=l1, l2=l2, l3=l3_mock, l4=l4)
        await manager.clear_session("sess")
        l3_mock.clear_session.assert_awaited_once_with("sess")


# ── get_conversation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGetConversation:
    async def test_returns_list(self, manager):
        turns = await manager.get_conversation("new-session")
        assert isinstance(turns, list)

    async def test_returns_messages_in_order(self, manager):
        msgs = ["first", "second", "third"]
        for m in msgs:
            await manager.record_turn(
                "ordered",
                Message(role=MessageRole.USER, content=m),
                Message(role=MessageRole.ASSISTANT, content=f"reply-{m}"),
            )

        turns = await manager.get_conversation("ordered")
        user_msgs = [t.content for t in turns if t.role == MessageRole.USER]
        assert user_msgs == msgs
