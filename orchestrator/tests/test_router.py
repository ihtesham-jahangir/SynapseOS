"""Tests for the intent classifier and routing engine."""
from __future__ import annotations

import pytest

from orchestrator.router.intent_classifier import HybridIntentClassifier
from orchestrator.router.routing_engine import RoutingEngine
from orchestrator.core.types import IntentType, ExpertType


@pytest.fixture
def classifier():
    # No embedder = keyword-only mode
    return HybridIntentClassifier(embedder=None)


@pytest.fixture
def router():
    return RoutingEngine()


class TestKeywordClassifier:
    @pytest.mark.asyncio
    async def test_coding_intent(self, classifier):
        intent = await classifier.classify("Write a Python function to sort a list")
        assert intent.intent_type == IntentType.CODING
        assert intent.confidence > 0.6

    @pytest.mark.asyncio
    async def test_math_intent(self, classifier):
        intent = await classifier.classify("Solve this equation: 2x + 5 = 11")
        assert intent.intent_type == IntentType.MATH
        assert intent.confidence > 0.6

    @pytest.mark.asyncio
    async def test_translation_intent(self, classifier):
        intent = await classifier.classify("Translate 'hello world' to French")
        assert intent.intent_type == IntentType.TRANSLATION

    @pytest.mark.asyncio
    async def test_summarization_intent(self, classifier):
        intent = await classifier.classify("Summarize this article for me")
        assert intent.intent_type == IntentType.SUMMARIZATION

    @pytest.mark.asyncio
    async def test_caching(self, classifier):
        text = "Write a Python function"
        # First call
        intent1 = await classifier.classify(text)
        # Second call – should hit cache
        intent2 = await classifier.classify(text)
        assert intent1.intent_type == intent2.intent_type
        assert intent1.confidence == intent2.confidence

    @pytest.mark.asyncio
    async def test_low_confidence_falls_back(self, classifier):
        # Generic phrase with no strong signals
        intent = await classifier.classify("yes please")
        # Should return something, not crash
        assert intent.intent_type is not None
        assert 0 <= intent.confidence <= 1


class TestRoutingEngine:
    def test_coding_routes_to_code_expert(self, router, classifier):
        from orchestrator.core.types import Intent
        intent = Intent(
            intent_type=IntentType.CODING,
            confidence=0.9,
            requires_expert=True,
        )
        decision = router.route(intent)
        assert ExpertType.CODE in decision.active_experts
        assert decision.generation_override.temperature < 0.5  # coding → low temp

    def test_rag_enabled_for_retrieval(self, router):
        from orchestrator.core.types import Intent
        intent = Intent(intent_type=IntentType.RETRIEVAL, confidence=0.85)
        decision = router.route(intent)
        assert decision.run_rag is True

    def test_low_confidence_routes_to_conversation(self, router):
        from orchestrator.core.types import Intent
        intent = Intent(intent_type=IntentType.CODING, confidence=0.3)
        decision = router.route(intent)
        # Low confidence should redirect
        assert ExpertType.GENERAL in decision.active_experts
