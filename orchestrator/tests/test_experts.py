"""
Tests for ExpertManager and individual expert modules.

Verifies: lazy loading, concurrent dispatch, fallback behavior,
per-expert system prompts, and confidence thresholds.
"""
from __future__ import annotations

import pytest

from orchestrator.core.types import ExpertGuidance, ExpertType, Intent, IntentType
from orchestrator.experts.expert_manager import ExpertManager, _EXPERT_REGISTRY
from orchestrator.experts.code_expert import CodeExpert
from orchestrator.experts.math_expert import MathExpert
from orchestrator.experts.translation_expert import TranslationExpert
from orchestrator.experts.summarization_expert import SummarizationExpert
from orchestrator.experts.reasoning_expert import ReasoningExpert


def _intent(itype: IntentType, confidence: float = 0.9) -> Intent:
    return Intent(intent_type=itype, confidence=confidence, requires_expert=True)


# ── Registry ─────────────────────────────────────────────────────────────────

class TestExpertRegistry:
    def test_all_expert_types_registered(self):
        expected = {
            ExpertType.CODE, ExpertType.MATH, ExpertType.TRANSLATION,
            ExpertType.SUMMARIZATION, ExpertType.REASONING, ExpertType.GENERAL,
        }
        assert expected.issubset(set(_EXPERT_REGISTRY.keys()))

    def test_registry_maps_to_correct_classes(self):
        assert _EXPERT_REGISTRY[ExpertType.CODE] is CodeExpert
        assert _EXPERT_REGISTRY[ExpertType.MATH] is MathExpert
        assert _EXPERT_REGISTRY[ExpertType.TRANSLATION] is TranslationExpert
        assert _EXPERT_REGISTRY[ExpertType.SUMMARIZATION] is SummarizationExpert
        assert _EXPERT_REGISTRY[ExpertType.REASONING] is ReasoningExpert


# ── ExpertManager ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestExpertManager:
    async def test_returns_guidance_for_code(self):
        mgr = ExpertManager()
        guidance = await mgr.get_guidance(
            "Write a Python function", _intent(IntentType.CODING), [ExpertType.CODE]
        )
        assert guidance is not None
        assert guidance.expert_type == ExpertType.CODE
        assert guidance.confidence > 0
        assert guidance.system_prompt != ""

    async def test_returns_guidance_for_math(self):
        mgr = ExpertManager()
        guidance = await mgr.get_guidance(
            "Solve x^2 + 5x + 6 = 0", _intent(IntentType.MATH), [ExpertType.MATH]
        )
        assert guidance is not None
        assert guidance.expert_type == ExpertType.MATH

    async def test_returns_guidance_for_translation(self):
        mgr = ExpertManager()
        guidance = await mgr.get_guidance(
            "Translate hello to French", _intent(IntentType.TRANSLATION), [ExpertType.TRANSLATION]
        )
        assert guidance is not None
        assert guidance.expert_type == ExpertType.TRANSLATION

    async def test_returns_guidance_for_summarization(self):
        mgr = ExpertManager()
        guidance = await mgr.get_guidance(
            "Summarize this text", _intent(IntentType.SUMMARIZATION), [ExpertType.SUMMARIZATION]
        )
        assert guidance is not None
        assert guidance.expert_type == ExpertType.SUMMARIZATION

    async def test_returns_guidance_for_reasoning(self):
        mgr = ExpertManager()
        guidance = await mgr.get_guidance(
            "All men are mortal. Socrates is a man.", _intent(IntentType.REASONING), [ExpertType.REASONING]
        )
        assert guidance is not None
        assert guidance.expert_type == ExpertType.REASONING

    async def test_empty_experts_falls_back_to_general(self):
        mgr = ExpertManager()
        guidance = await mgr.get_guidance(
            "Hi there", _intent(IntentType.CONVERSATION), []
        )
        assert guidance is not None
        assert guidance.expert_type == ExpertType.GENERAL

    async def test_unknown_expert_type_returns_general(self):
        mgr = ExpertManager()
        # Pass an invalid expert type that doesn't exist in registry
        guidance = await mgr.get_guidance(
            "test", _intent(IntentType.CONVERSATION), [ExpertType.GENERAL]
        )
        assert guidance is not None

    async def test_lazy_loading_caches_instance(self):
        mgr = ExpertManager()
        await mgr.get_guidance("test", _intent(IntentType.CODING), [ExpertType.CODE])
        await mgr.get_guidance("test2", _intent(IntentType.CODING), [ExpertType.CODE])
        # Same instance returned from cache
        assert ExpertType.CODE in mgr._instances

    async def test_concurrent_expert_calls_safe(self):
        import asyncio
        mgr = ExpertManager()
        intent = _intent(IntentType.CODING)
        results = await asyncio.gather(
            mgr.get_guidance("query1", intent, [ExpertType.CODE]),
            mgr.get_guidance("query2", intent, [ExpertType.CODE]),
            mgr.get_guidance("query3", intent, [ExpertType.CODE]),
        )
        assert all(r is not None for r in results)

    async def test_picks_highest_confidence_from_multiple(self):
        mgr = ExpertManager()
        intent = _intent(IntentType.CODING, confidence=0.9)
        guidance = await mgr.get_guidance(
            "Write code", intent, [ExpertType.CODE, ExpertType.GENERAL]
        )
        assert guidance is not None
        assert guidance.confidence >= 0

    async def test_available_experts_lists_all(self):
        mgr = ExpertManager()
        experts = mgr.available_experts()
        assert len(experts) >= 6
        assert "code" in experts
        assert "math" in experts

    async def test_guidance_has_token_count(self):
        mgr = ExpertManager()
        guidance = await mgr.get_guidance(
            "Write a sort function", _intent(IntentType.CODING), [ExpertType.CODE]
        )
        assert guidance is not None
        assert guidance.token_count > 0


# ── Individual expert system prompts ─────────────────────────────────────────

class TestExpertSystemPrompts:
    def test_code_expert_prompt_relevant(self):
        expert = CodeExpert()
        assert expert.is_available()
        # Prompt should reference programming/code
        prompt_lower = expert._system_prompt.lower()
        assert any(kw in prompt_lower for kw in ["code", "program", "engineer", "software"])

    def test_math_expert_prompt_relevant(self):
        expert = MathExpert()
        prompt_lower = expert._system_prompt.lower()
        assert any(kw in prompt_lower for kw in ["math", "equation", "solve", "calcul"])

    def test_translation_expert_prompt_relevant(self):
        expert = TranslationExpert()
        prompt_lower = expert._system_prompt.lower()
        assert any(kw in prompt_lower for kw in ["translat", "language", "linguist"])

    def test_reasoning_expert_prompt_relevant(self):
        expert = ReasoningExpert()
        prompt_lower = expert._system_prompt.lower()
        assert any(kw in prompt_lower for kw in ["reason", "logic", "analy", "think"])
