"""
Tests for the InferenceEngine – the core orchestration pipeline.

All LLM calls are mocked so no running server is required.
Tests verify pipeline correctness, error propagation, and conversation handling.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.core.types import (
    ChatRequest,
    Intent,
    IntentType,
    Message,
    MessageRole,
    MemoryResult,
    RAGResult,
    FusedContext,
    ContextItem,
    GenerationParams,
    InferenceResponse,
)
from orchestrator.core.exceptions import LlamaServerError, LlamaTimeoutError
from orchestrator.runtime.inference_engine import InferenceEngine
from orchestrator.router.intent_classifier import HybridIntentClassifier
from orchestrator.router.routing_engine import RoutingEngine
from orchestrator.experts.expert_manager import ExpertManager
from orchestrator.fusion.fusion_engine import AdaptiveFusionEngine
from orchestrator.runtime.adaptive_compute import AdaptiveComputeController


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_engine(llama_client, memory_manager=None, rag_pipeline=None):
    """Build an InferenceEngine with all expensive components mocked."""
    if memory_manager is None:
        memory_manager = MagicMock()
        memory_manager.get_conversation = AsyncMock(return_value=[])
        memory_manager.retrieve_all = AsyncMock(
            return_value=MemoryResult(
                items=[], total_tokens=0, levels_queried=[], query_time_ms=0
            )
        )
        memory_manager.record_turn = AsyncMock()

    if rag_pipeline is None:
        rag_pipeline = MagicMock()
        rag_pipeline.retrieve = AsyncMock(
            return_value=RAGResult(chunks=[], scores=[], total_tokens=0, query_time_ms=0)
        )

    return InferenceEngine(
        llama_client=llama_client,
        memory_manager=memory_manager,
        rag_pipeline=rag_pipeline,
        intent_classifier=HybridIntentClassifier(embedder=None),
        routing_engine=RoutingEngine(),
        expert_manager=ExpertManager(),
        fusion_engine=AdaptiveFusionEngine(),
        compute_controller=AdaptiveComputeController(),
    )


def _chat_request(text: str, session_id: str = "test-session", **kwargs) -> ChatRequest:
    return ChatRequest(
        session_id=session_id,
        messages=[Message(role=MessageRole.USER, content=text)],
        **kwargs,
    )


# ── process_request ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestProcessRequest:
    async def test_returns_chat_response(self, mock_llama_client):
        engine = _make_engine(mock_llama_client)
        request = _chat_request("Hello, how are you?")
        response = await engine.process_request(request)

        assert response.session_id == "test-session"
        assert response.content == "This is a mock LLM response."
        assert response.tokens_generated > 0
        assert response.total_time_ms >= 0

    async def test_user_message_reaches_llm(self, mock_llama_client):
        """Verify the user query is present in the messages sent to the LLM."""
        engine = _make_engine(mock_llama_client)
        request = _chat_request("What is the capital of France?")
        await engine.process_request(request)

        call_args = mock_llama_client.chat.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        assert any("France" in c for c in user_contents), (
            "User query was never passed to the LLM"
        )

    async def test_max_tokens_respected(self, mock_llama_client):
        """Request-level max_tokens acts as an upper bound on generation."""
        engine = _make_engine(mock_llama_client)
        # Use a high max_tokens; the AdaptiveComputeController will cap by intent limit
        request = _chat_request("Hello", max_tokens=512)
        await engine.process_request(request)

        call_args = mock_llama_client.chat.call_args
        sent_max = call_args.kwargs.get(
            "max_tokens", call_args.args[1] if len(call_args.args) > 1 else None
        )
        assert sent_max is not None
        # CONVERSATION intent cap is 256; request's 512 should be capped
        assert sent_max <= 512

    async def test_llama_timeout_raises(self, mock_llama_client):
        mock_llama_client.chat.side_effect = LlamaTimeoutError()
        engine = _make_engine(mock_llama_client)
        with pytest.raises(LlamaTimeoutError):
            await engine.process_request(_chat_request("ping"))

    async def test_llama_server_error_raises(self, mock_llama_client):
        mock_llama_client.chat.side_effect = LlamaServerError("connection refused")
        engine = _make_engine(mock_llama_client)
        with pytest.raises(LlamaServerError):
            await engine.process_request(_chat_request("ping"))

    async def test_multi_turn_history_merged(self, mock_llama_client):
        """Messages from the request body must be merged into conversation context."""
        engine = _make_engine(mock_llama_client)
        request = ChatRequest(
            session_id="session-history",
            messages=[
                Message(role=MessageRole.USER, content="My name is Alice"),
                Message(role=MessageRole.ASSISTANT, content="Hi Alice!"),
                Message(role=MessageRole.USER, content="What is my name?"),
            ],
        )
        await engine.process_request(request)

        call_args = mock_llama_client.chat.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
        all_content = " ".join(m["content"] for m in messages)
        assert "Alice" in all_content, "Prior conversation history was not passed to LLM"

    async def test_response_includes_metadata(self, mock_llama_client):
        engine = _make_engine(mock_llama_client)
        response = await engine.process_request(_chat_request("test"))
        assert response.metadata is not None
        assert isinstance(response.metadata, dict)

    async def test_different_sessions_are_isolated(self, mock_llama_client):
        """Session A's memory manager must not be called with session B's data."""
        record_a: list = []
        record_b: list = []

        def make_mem(records):
            m = MagicMock()
            m.get_conversation = AsyncMock(return_value=[])
            m.retrieve_all = AsyncMock(
                return_value=MemoryResult(
                    items=[], total_tokens=0, levels_queried=[], query_time_ms=0
                )
            )
            async def _record_turn(sid, user_msg, asst_msg):
                records.append(sid)
            m.record_turn = _record_turn
            return m

        mem_a = make_mem(record_a)
        mem_b = make_mem(record_b)
        engine_a = _make_engine(mock_llama_client, memory_manager=mem_a)
        engine_b = _make_engine(mock_llama_client, memory_manager=mem_b)

        await engine_a.process_request(_chat_request("Session A query", session_id="sess-a"))
        await engine_b.process_request(_chat_request("Session B query", session_id="sess-b"))

        # engine_a's memory only recorded sess-a, and engine_b's only sess-b
        assert all(sid == "sess-a" for sid in record_a)
        assert all(sid == "sess-b" for sid in record_b)


# ── generate ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGenerate:
    async def test_returns_inference_response(self, mock_llama_client, sample_fused_context, sample_gen_params):
        engine = _make_engine(mock_llama_client)
        resp = await engine.generate(sample_fused_context, sample_gen_params)

        assert resp.content == "This is a mock LLM response."
        assert resp.tokens_generated > 0
        assert resp.total_time_ms >= 0
        assert resp.time_to_first_token_ms >= 0

    async def test_propagates_llama_error(self, mock_llama_client, sample_fused_context, sample_gen_params):
        mock_llama_client.chat.side_effect = LlamaServerError("down")
        engine = _make_engine(mock_llama_client)
        with pytest.raises(LlamaServerError):
            await engine.generate(sample_fused_context, sample_gen_params)


# ── stream ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestStream:
    async def test_yields_tokens(self, mock_llama_client, sample_fused_context, sample_gen_params):
        engine = _make_engine(mock_llama_client)
        tokens = []
        async for token in engine.stream(sample_fused_context, sample_gen_params):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    async def test_reconstructed_content_nonempty(self, mock_llama_client, sample_fused_context, sample_gen_params):
        engine = _make_engine(mock_llama_client)
        tokens = [t async for t in engine.stream(sample_fused_context, sample_gen_params)]
        assert "".join(tokens).strip() != ""


# ── health_check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHealthCheck:
    async def test_returns_true_when_llama_alive(self, mock_llama_client):
        engine = _make_engine(mock_llama_client)
        assert await engine.health_check() is True

    async def test_returns_false_when_llama_down(self, mock_llama_client):
        mock_llama_client.health_check.return_value = False
        engine = _make_engine(mock_llama_client)
        assert await engine.health_check() is False
