"""
Tests for AdaptiveComputeController.

Verifies token budget enforcement, temperature adjustment,
stop-sequence merging, and per-intent max_token caps.
"""
from __future__ import annotations

import pytest

from orchestrator.core.types import (
    ContextItem,
    FusedContext,
    GenerationParams,
    Intent,
    IntentType,
    Message,
    MessageRole,
)
from orchestrator.runtime.adaptive_compute import (
    AdaptiveComputeController,
    _INTENT_MAX_TOKENS,
    _INTENT_STOP_SEQUENCES,
)


def _intent(itype: IntentType, confidence: float = 0.8) -> Intent:
    return Intent(intent_type=itype, confidence=confidence)


def _fused(token_count: int = 100) -> FusedContext:
    return FusedContext(
        system_prompt="You are a helpful assistant.",
        context_items=[],
        conversation_turns=[Message(role=MessageRole.USER, content="test")],
        total_token_count=token_count,
        token_budget_used=token_count / 4096,
    )


def _base(max_tokens: int = 512, temperature: float = 0.7) -> GenerationParams:
    return GenerationParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
    )


@pytest.fixture
def controller() -> AdaptiveComputeController:
    return AdaptiveComputeController()


# ── max_tokens logic ──────────────────────────────────────────────────────────

class TestMaxTokens:
    def test_coding_caps_at_intent_limit(self, controller):
        params = controller.optimize(_intent(IntentType.CODING), _fused(), _base(max_tokens=2048))
        assert params.max_tokens <= _INTENT_MAX_TOKENS[IntentType.CODING]

    def test_math_caps_at_intent_limit(self, controller):
        params = controller.optimize(_intent(IntentType.MATH), _fused(), _base(max_tokens=2048))
        assert params.max_tokens <= _INTENT_MAX_TOKENS[IntentType.MATH]

    def test_conversation_caps_at_intent_limit(self, controller):
        params = controller.optimize(_intent(IntentType.CONVERSATION), _fused(), _base(max_tokens=2048))
        assert params.max_tokens <= _INTENT_MAX_TOKENS[IntentType.CONVERSATION]

    def test_large_context_reduces_max_tokens(self, controller):
        # 4000-token context leaves very little room
        large_context = _fused(token_count=4000)
        params = controller.optimize(_intent(IntentType.CODING), large_context, _base(max_tokens=1024))
        assert params.max_tokens <= 1024

    def test_max_tokens_never_below_minimum(self, controller):
        huge_context = _fused(token_count=4090)
        params = controller.optimize(_intent(IntentType.CODING), huge_context, _base(max_tokens=512))
        assert params.max_tokens >= 128  # enforced minimum

    def test_user_max_tokens_not_exceeded(self, controller):
        params = controller.optimize(_intent(IntentType.CONVERSATION), _fused(), _base(max_tokens=50))
        assert params.max_tokens <= 50


# ── temperature logic ─────────────────────────────────────────────────────────

class TestTemperature:
    def test_high_confidence_reduces_temperature(self, controller):
        base = _base(temperature=0.7)
        low_conf = controller.optimize(_intent(IntentType.CODING, confidence=0.5), _fused(), base)
        high_conf = controller.optimize(_intent(IntentType.CODING, confidence=0.95), _fused(), base)
        assert high_conf.temperature < low_conf.temperature

    def test_temperature_never_below_floor(self, controller):
        base = _base(temperature=0.1)
        params = controller.optimize(_intent(IntentType.CODING, confidence=0.99), _fused(), base)
        assert params.temperature >= 0.05

    def test_temperature_unchanged_below_threshold(self, controller):
        base = _base(temperature=0.7)
        params = controller.optimize(_intent(IntentType.CODING, confidence=0.5), _fused(), base)
        assert params.temperature == pytest.approx(0.7)


# ── stop sequences ────────────────────────────────────────────────────────────

class TestStopSequences:
    def test_coding_stop_sequences_added(self, controller):
        params = controller.optimize(_intent(IntentType.CODING), _fused(), _base())
        coding_stops = _INTENT_STOP_SEQUENCES.get(IntentType.CODING, [])
        for stop in coding_stops:
            assert stop in params.stop_sequences

    def test_no_duplicate_stop_sequences(self, controller):
        params = controller.optimize(_intent(IntentType.CONVERSATION), _fused(), _base())
        assert len(params.stop_sequences) == len(set(params.stop_sequences))

    def test_default_stops_always_present(self, controller):
        from orchestrator.config.settings import get_settings
        default_stops = get_settings().generation.stop_sequences
        params = controller.optimize(_intent(IntentType.MATH), _fused(), _base())
        for s in default_stops:
            assert s in params.stop_sequences


# ── other params preserved ────────────────────────────────────────────────────

class TestParamPreservation:
    def test_top_p_preserved(self, controller):
        base = _base()
        base.top_p = 0.88
        params = controller.optimize(_intent(IntentType.MATH), _fused(), base)
        assert params.top_p == pytest.approx(0.88)

    def test_top_k_preserved(self, controller):
        base = _base()
        base.top_k = 50
        params = controller.optimize(_intent(IntentType.MATH), _fused(), base)
        assert params.top_k == 50

    def test_repeat_penalty_preserved(self, controller):
        base = _base()
        base.repeat_penalty = 1.15
        params = controller.optimize(_intent(IntentType.MATH), _fused(), base)
        assert params.repeat_penalty == pytest.approx(1.15)

    def test_no_base_params_uses_defaults(self, controller):
        params = controller.optimize(_intent(IntentType.CONVERSATION), _fused(), None)
        assert params is not None
        assert params.max_tokens > 0
        assert 0 < params.temperature <= 2.0


# ── should_use_rag ─────────────────────────────────────────────────────────────

class TestShouldUseRAG:
    def test_retrieval_uses_rag(self):
        intent = Intent(intent_type=IntentType.RETRIEVAL, confidence=0.9, requires_rag=True)
        assert AdaptiveComputeController.should_use_rag(intent) is True

    def test_reasoning_uses_rag(self):
        intent = Intent(intent_type=IntentType.REASONING, confidence=0.9)
        assert AdaptiveComputeController.should_use_rag(intent) is True

    def test_conversation_no_rag(self):
        intent = Intent(intent_type=IntentType.CONVERSATION, confidence=0.9)
        assert AdaptiveComputeController.should_use_rag(intent) is False

    def test_coding_no_rag_by_default(self):
        intent = Intent(intent_type=IntentType.CODING, confidence=0.9)
        assert AdaptiveComputeController.should_use_rag(intent) is False
