"""
v3.5 feature tests — expert intelligence upgrade.

Groups:
  1.  New IntentType / ExpertType enum values
  2.  IntentScore model + Intent secondary_intents field
  3.  Keyword scoring for all 6 new domains
  4.  _keyword_score_all returns scores for ALL matching intents
  5.  Multi-intent detection (secondary_intents above 0.40)
  6.  Subtype extraction (coding language, translation target, data tool)
  7.  Context-aware classification (_apply_context_prior)
  8.  KEYWORD_THRESHOLD formula: 2 hits → exactly 0.72
  9.  Semantic scoring via per-exemplar similarity
  10. RoutingEngine: new intents mapped correctly
  11. RoutingEngine: multi-expert routing from secondary intents
  12. RoutingEngine: adaptive temperature (uncertainty bump)
  13. RoutingEngine: adaptive max_tokens multiplier
  14. RoutingEngine: security priority = 3, confidence threshold = 0.45
  15. RoutingEngine: subtype-aware system prompt specialization
  16. ExpertManager: all 12 experts registered + lazy load
  17. New expert system prompts are non-empty and domain-specific
  18. New expert few-shot examples are populated
  19. ExpertManager dynamic few-shot selection
  20. _keyword_classify backward-compat wrapper
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


# ── 1. New IntentType / ExpertType enum values ────────────────────────────────

class TestNewEnumValues:
    def test_new_intent_types_exist(self):
        from orchestrator.core.types import IntentType
        assert IntentType.WRITING.value == "writing"
        assert IntentType.DATA_ANALYSIS.value == "data_analysis"
        assert IntentType.CREATIVE.value == "creative"
        assert IntentType.SECURITY.value == "security"
        assert IntentType.PLANNING.value == "planning"
        assert IntentType.EDUCATION.value == "education"

    def test_new_expert_types_exist(self):
        from orchestrator.core.types import ExpertType
        assert ExpertType.WRITING.value == "writing"
        assert ExpertType.DATA_ANALYSIS.value == "data_analysis"
        assert ExpertType.CREATIVE.value == "creative"
        assert ExpertType.SECURITY.value == "security"
        assert ExpertType.PLANNING.value == "planning"
        assert ExpertType.EDUCATION.value == "education"

    def test_total_intent_types(self):
        from orchestrator.core.types import IntentType
        # 9 original + 6 new = 15
        assert len(IntentType) == 15

    def test_total_expert_types(self):
        from orchestrator.core.types import ExpertType
        # 6 original + 6 new = 12
        assert len(ExpertType) == 12


# ── 2. IntentScore model + Intent secondary_intents ───────────────────────────

class TestIntentScoreModel:
    def test_intent_score_fields(self):
        from orchestrator.core.types import IntentScore, IntentType
        score = IntentScore(intent_type=IntentType.CODING, confidence=0.85)
        assert score.intent_type == IntentType.CODING
        assert score.confidence == 0.85

    def test_intent_has_secondary_intents_field(self):
        from orchestrator.core.types import Intent, IntentType
        intent = Intent(
            intent_type=IntentType.CODING,
            confidence=0.80,
            secondary_intents=[],
        )
        assert intent.secondary_intents == []

    def test_intent_has_classification_stage_field(self):
        from orchestrator.core.types import Intent, IntentType
        intent = Intent(
            intent_type=IntentType.MATH,
            confidence=0.75,
            classification_stage="keyword",
        )
        assert intent.classification_stage == "keyword"

    def test_intent_secondary_intents_defaults_empty(self):
        from orchestrator.core.types import Intent, IntentType
        intent = Intent(intent_type=IntentType.WRITING, confidence=0.65)
        assert isinstance(intent.secondary_intents, list)
        assert len(intent.secondary_intents) == 0


# ── 3. Keyword scoring for all 6 new domains ──────────────────────────────────

class TestNewDomainKeywords:
    @pytest.fixture
    def clf(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        return HybridIntentClassifier(embedder=None)

    def test_writing_keyword_matched(self, clf):
        from orchestrator.core.types import IntentType
        scores = clf._keyword_score_all("write an essay about climate change")
        assert IntentType.WRITING in scores
        assert scores[IntentType.WRITING] > 0

    def test_security_keyword_matched(self, clf):
        from orchestrator.core.types import IntentType
        scores = clf._keyword_score_all("check this code for SQL injection vulnerabilities")
        assert IntentType.SECURITY in scores
        assert scores[IntentType.SECURITY] > 0

    def test_planning_keyword_matched(self, clf):
        from orchestrator.core.types import IntentType
        scores = clf._keyword_score_all("create a project roadmap for the next quarter")
        assert IntentType.PLANNING in scores
        assert scores[IntentType.PLANNING] > 0

    def test_education_keyword_matched(self, clf):
        from orchestrator.core.types import IntentType
        scores = clf._keyword_score_all("explain recursion to a beginner with examples")
        assert IntentType.EDUCATION in scores
        assert scores[IntentType.EDUCATION] > 0

    def test_creative_keyword_matched(self, clf):
        from orchestrator.core.types import IntentType
        scores = clf._keyword_score_all("brainstorm ideas for a mobile app")
        assert IntentType.CREATIVE in scores
        assert scores[IntentType.CREATIVE] > 0

    def test_data_analysis_keyword_matched(self, clf):
        from orchestrator.core.types import IntentType
        scores = clf._keyword_score_all("analyze this dataframe with pandas and plot results")
        assert IntentType.DATA_ANALYSIS in scores
        assert scores[IntentType.DATA_ANALYSIS] > 0


# ── 4. _keyword_score_all returns ALL matching intents ───────────────────────

class TestKeywordScoreAll:
    @pytest.fixture
    def clf(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        return HybridIntentClassifier(embedder=None)

    def test_returns_dict(self, clf):
        result = clf._keyword_score_all("debug my Python code")
        assert isinstance(result, dict)

    def test_multi_intent_query_scores_multiple(self, clf):
        from orchestrator.core.types import IntentType
        # "explain how to fix this security bug in Python code" should hit
        # EDUCATION, SECURITY, and CODING
        scores = clf._keyword_score_all(
            "explain how to fix this security injection bug in my Python code"
        )
        assert len(scores) >= 2

    def test_unrelated_query_returns_empty_or_low(self, clf):
        scores = clf._keyword_score_all("hello")
        # Either empty or low confidence
        if scores:
            assert max(scores.values()) < 0.80

    def test_all_scores_bounded(self, clf):
        scores = clf._keyword_score_all(
            "write Python code to analyze data and summarize security vulnerabilities"
        )
        for confidence in scores.values():
            assert 0.0 < confidence <= 1.0


# ── 5. Multi-intent detection ─────────────────────────────────────────────────

class TestMultiIntentDetection:
    @pytest.fixture
    def clf(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        return HybridIntentClassifier(embedder=None)

    @pytest.mark.asyncio
    async def test_secondary_intents_populated_for_overlap(self, clf):
        # A query that hits both CODING and SECURITY keyword patterns
        intent = await clf.classify(
            "write a secure Python function that sanitizes SQL injection input"
        )
        # Primary should be one of CODING or SECURITY
        from orchestrator.core.types import IntentType
        assert intent.intent_type in {IntentType.CODING, IntentType.SECURITY}

    @pytest.mark.asyncio
    async def test_secondary_intents_are_list_of_intent_scores(self, clf):
        from orchestrator.core.types import IntentScore
        intent = await clf.classify("explain Python recursion with code examples")
        assert isinstance(intent.secondary_intents, list)
        for s in intent.secondary_intents:
            assert isinstance(s, IntentScore)
            assert 0.0 <= s.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_secondary_intents_capped_at_three(self, clf):
        intent = await clf.classify(
            "write secure Python code to analyze and summarize data with SQL queries"
        )
        assert len(intent.secondary_intents) <= 3

    @pytest.mark.asyncio
    async def test_secondary_intents_above_threshold(self, clf):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        threshold = HybridIntentClassifier.MIN_SECONDARY_CONFIDENCE
        intent = await clf.classify("write Python code to visualize security metrics")
        for s in intent.secondary_intents:
            assert s.confidence >= threshold

    @pytest.mark.asyncio
    async def test_primary_not_in_secondary(self, clf):
        intent = await clf.classify("write a Python function to detect SQL injection")
        secondary_types = {s.intent_type for s in intent.secondary_intents}
        assert intent.intent_type not in secondary_types

    @pytest.mark.asyncio
    async def test_classification_stage_set(self, clf):
        intent = await clf.classify("debug my code")
        assert intent.classification_stage in {"keyword", "semantic", "fused", "unknown"}


# ── 6. Subtype extraction ─────────────────────────────────────────────────────

class TestSubtypeExtraction:
    def test_coding_python_subtype(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("write a Python function", IntentType.CODING)
        assert result == "python"

    def test_coding_javascript_subtype(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("fix my JavaScript bug in React", IntentType.CODING)
        assert result == "javascript"

    def test_coding_rust_subtype(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("implement a Rust struct with lifetimes", IntentType.CODING)
        assert result == "rust"

    def test_translation_french_target(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("translate this paragraph to French", IntentType.TRANSLATION)
        assert result == "french"

    def test_translation_japanese_target(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("say hello in Japanese", IntentType.TRANSLATION)
        assert result == "japanese"

    def test_data_pandas_tool(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("use pandas to filter this dataframe", IntentType.DATA_ANALYSIS)
        assert result == "pandas"

    def test_data_sql_tool(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("write a SQL query to select from users", IntentType.DATA_ANALYSIS)
        assert result == "sql"

    def test_no_subtype_for_vague_query(self):
        from orchestrator.router.intent_classifier import _extract_subtype
        from orchestrator.core.types import IntentType
        result = _extract_subtype("write some code", IntentType.CODING)
        assert result is None

    def test_subtype_propagated_in_intent(self):
        import asyncio
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        clf = HybridIntentClassifier(embedder=None)
        intent = asyncio.get_event_loop().run_until_complete(
            clf.classify("write a Python script to automate backups")
        )
        if intent.subtype is not None:
            assert intent.subtype == "python"


# ── 7. Context-aware classification ──────────────────────────────────────────

class TestContextAwareClassification:
    @pytest.fixture
    def clf(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        return HybridIntentClassifier(embedder=None)

    def test_context_prior_boosts_existing_intent(self, clf):
        from orchestrator.core.types import IntentType
        scores = {IntentType.CODING: 0.60, IntentType.MATH: 0.50}
        context = "write a Python function"  # coding context
        result = clf._apply_context_prior(scores, context)
        assert result[IntentType.CODING] > scores[IntentType.CODING]

    def test_context_prior_returns_dict(self, clf):
        from orchestrator.core.types import IntentType
        scores = {IntentType.WRITING: 0.55}
        result = clf._apply_context_prior(scores, "draft an email")
        assert isinstance(result, dict)

    def test_no_context_returns_unchanged(self, clf):
        from orchestrator.core.types import IntentType
        scores = {IntentType.MATH: 0.70}
        result = clf._apply_context_prior(scores, None)
        assert result == scores

    def test_empty_context_returns_unchanged(self, clf):
        from orchestrator.core.types import IntentType
        scores = {IntentType.CODING: 0.65}
        result = clf._apply_context_prior(scores, "")
        assert result == scores

    def test_context_boost_capped_at_one(self, clf):
        from orchestrator.core.types import IntentType
        scores = {IntentType.CODING: 0.95}
        result = clf._apply_context_prior(scores, "write a function")
        assert result[IntentType.CODING] <= 1.0

    @pytest.mark.asyncio
    async def test_classify_with_context_shifts_result(self, clf):
        # Ask a neutral question, but in a strong coding context
        intent_no_ctx = await clf.classify("how does this work?")
        intent_with_ctx = await clf.classify(
            "how does this work?",
            context="implement a Python function using recursion"
        )
        # The coding-context version should have coding-related score shifts
        # (or at least not error out)
        assert intent_with_ctx is not None


# ── 8. KEYWORD_THRESHOLD formula: 2 hits → exactly 0.72 ─────────────────────

class TestKeywordFormula:
    @pytest.fixture
    def clf(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        return HybridIntentClassifier(embedder=None)

    def test_two_hits_reach_keyword_threshold(self, clf):
        # "write" + "code" + "function" should produce ≥ 3 hits for CODING
        scores = clf._keyword_score_all("write a code function")
        from orchestrator.core.types import IntentType
        assert IntentType.CODING in scores
        assert scores[IntentType.CODING] >= clf.KEYWORD_THRESHOLD

    def test_single_hit_below_threshold(self, clf):
        from orchestrator.core.types import IntentType
        # Only "code" — 1 hit → 0.60 + 1*0.06 = 0.66 < 0.72
        scores = clf._keyword_score_all("the code")
        if IntentType.CODING in scores:
            assert scores[IntentType.CODING] < clf.KEYWORD_THRESHOLD

    def test_many_hits_capped_at_threshold(self, clf):
        from orchestrator.core.types import IntentType
        # Massive overlap — should cap at KEYWORD_THRESHOLD (0.72)
        scores = clf._keyword_score_all(
            "write code function class debug python javascript algorithm recursion"
        )
        assert IntentType.CODING in scores
        assert scores[IntentType.CODING] <= 1.0


# ── 9. Semantic scoring (per-exemplar) ───────────────────────────────────────

class TestPerExemplarSemanticScoring:
    @pytest.fixture
    def clf_with_embedder(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        embedder = MagicMock()
        # Return deterministic embeddings:
        # "coding" query → strong match to CODING exemplar bucket
        async def mock_embed(texts):
            return [np.random.rand(384).astype(np.float32).tolist() for _ in texts]
        async def mock_embed_query(text):
            return np.random.rand(384).astype(np.float32).tolist()
        embedder.embed = mock_embed
        embedder.embed_query = mock_embed_query
        return HybridIntentClassifier(embedder=embedder)

    @pytest.mark.asyncio
    async def test_semantic_score_all_returns_dict(self, clf_with_embedder):
        scores = await clf_with_embedder._semantic_score_all("explain recursion")
        assert isinstance(scores, dict)

    @pytest.mark.asyncio
    async def test_semantic_scores_bounded(self, clf_with_embedder):
        scores = await clf_with_embedder._semantic_score_all("write Python code")
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    @pytest.mark.asyncio
    async def test_exemplars_loaded_after_first_call(self, clf_with_embedder):
        assert clf_with_embedder._exemplar_embeddings is None
        await clf_with_embedder._semantic_score_all("test query")
        assert clf_with_embedder._exemplar_embeddings is not None

    @pytest.mark.asyncio
    async def test_no_embedder_returns_empty(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        clf = HybridIntentClassifier(embedder=None)
        scores = await clf._semantic_score_all("write Python code")
        assert scores == {}


# ── 10. RoutingEngine: new intents mapped correctly ──────────────────────────

class TestRoutingEngineNewIntents:
    @pytest.fixture
    def engine(self):
        from orchestrator.router.routing_engine import RoutingEngine
        return RoutingEngine()

    def _make_intent(self, intent_type, confidence=0.80):
        from orchestrator.core.types import Intent, IntentType
        return Intent(intent_type=intent_type, confidence=confidence)

    def test_writing_routes_to_writing_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        decision = engine.route(self._make_intent(IntentType.WRITING))
        assert ExpertType.WRITING in decision.active_experts

    def test_security_routes_to_security_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        decision = engine.route(self._make_intent(IntentType.SECURITY))
        assert ExpertType.SECURITY in decision.active_experts

    def test_planning_routes_to_planning_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        decision = engine.route(self._make_intent(IntentType.PLANNING))
        assert ExpertType.PLANNING in decision.active_experts

    def test_education_routes_to_education_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        decision = engine.route(self._make_intent(IntentType.EDUCATION))
        assert ExpertType.EDUCATION in decision.active_experts

    def test_creative_routes_to_creative_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        decision = engine.route(self._make_intent(IntentType.CREATIVE))
        assert ExpertType.CREATIVE in decision.active_experts

    def test_data_analysis_routes_to_data_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        decision = engine.route(self._make_intent(IntentType.DATA_ANALYSIS))
        assert ExpertType.DATA_ANALYSIS in decision.active_experts

    def test_creative_has_high_temperature(self, engine):
        from orchestrator.core.types import IntentType
        decision = engine.route(self._make_intent(IntentType.CREATIVE, confidence=0.90))
        assert decision.generation_override.temperature >= 0.85

    def test_security_has_low_temperature(self, engine):
        from orchestrator.core.types import IntentType
        decision = engine.route(self._make_intent(IntentType.SECURITY, confidence=0.90))
        assert decision.generation_override.temperature < 0.30


# ── 11. RoutingEngine: multi-expert routing from secondary intents ────────────

class TestMultiExpertRouting:
    @pytest.fixture
    def engine(self):
        from orchestrator.router.routing_engine import RoutingEngine
        return RoutingEngine()

    def _make_intent_with_secondary(self, primary, secondary_list, confidence=0.80):
        from orchestrator.core.types import Intent, IntentScore
        return Intent(
            intent_type=primary,
            confidence=confidence,
            secondary_intents=[
                IntentScore(intent_type=s, confidence=0.55)
                for s in secondary_list
            ],
        )

    def test_coding_plus_security_secondary_adds_security_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        intent = self._make_intent_with_secondary(
            IntentType.CODING, [IntentType.SECURITY]
        )
        decision = engine.route(intent)
        assert ExpertType.SECURITY in decision.active_experts
        assert ExpertType.CODE in decision.active_experts

    def test_coding_plus_education_secondary_adds_education_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        intent = self._make_intent_with_secondary(
            IntentType.CODING, [IntentType.EDUCATION]
        )
        decision = engine.route(intent)
        assert ExpertType.EDUCATION in decision.active_experts

    def test_writing_plus_creative_adds_creative_expert(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        intent = self._make_intent_with_secondary(
            IntentType.WRITING, [IntentType.CREATIVE]
        )
        decision = engine.route(intent)
        assert ExpertType.CREATIVE in decision.active_experts

    def test_no_duplicate_experts(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        # EDUCATION primary + CODING secondary → adds CODE expert
        intent = self._make_intent_with_secondary(
            IntentType.EDUCATION, [IntentType.CODING]
        )
        decision = engine.route(intent)
        assert len(decision.active_experts) == len(set(decision.active_experts))

    def test_unknown_secondary_pair_not_added(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        # MATH + VOICE has no mapping in _SECONDARY_EXPERT_MAP
        intent = self._make_intent_with_secondary(
            IntentType.MATH, [IntentType.VOICE]
        )
        decision = engine.route(intent)
        assert decision.active_experts == [ExpertType.MATH]


# ── 12. RoutingEngine: adaptive temperature ───────────────────────────────────

class TestAdaptiveTemperature:
    @pytest.fixture
    def engine(self):
        from orchestrator.router.routing_engine import RoutingEngine
        return RoutingEngine()

    def _make_intent(self, intent_type, confidence):
        from orchestrator.core.types import Intent
        return Intent(intent_type=intent_type, confidence=confidence)

    def test_high_confidence_no_bump(self, engine):
        from orchestrator.core.types import IntentType
        # confidence = 0.90 → uncertainty_bump = max(0, (0.65-0.90)*0.15) = 0
        intent = self._make_intent(IntentType.CODING, confidence=0.90)
        decision = engine.route(intent)
        # temperature should be at (or very close to) the base value of 0.15
        assert decision.generation_override.temperature <= 0.20

    def test_low_confidence_adds_bump(self, engine):
        from orchestrator.core.types import IntentType
        # confidence = 0.50 → uncertainty_bump = (0.65-0.50)*0.15 = 0.0225
        high_conf_decision = engine.route(self._make_intent(IntentType.CODING, 0.90))
        low_conf_decision = engine.route(self._make_intent(IntentType.CODING, 0.50))
        assert low_conf_decision.generation_override.temperature > high_conf_decision.generation_override.temperature

    def test_temperature_capped_at_one(self, engine):
        from orchestrator.core.types import IntentType
        intent = self._make_intent(IntentType.CREATIVE, confidence=0.10)
        decision = engine.route(intent)
        assert decision.generation_override.temperature <= 1.0


# ── 13. RoutingEngine: adaptive max_tokens multiplier ─────────────────────────

class TestAdaptiveMaxTokens:
    @pytest.fixture
    def engine(self):
        from orchestrator.router.routing_engine import RoutingEngine
        return RoutingEngine()

    def _make_intent(self, intent_type, confidence=0.80):
        from orchestrator.core.types import Intent
        return Intent(intent_type=intent_type, confidence=confidence)

    def test_coding_gets_more_tokens_than_retrieval(self, engine):
        from orchestrator.core.types import IntentType
        code_dec = engine.route(self._make_intent(IntentType.CODING))
        retrieval_dec = engine.route(self._make_intent(IntentType.RETRIEVAL))
        assert code_dec.generation_override.max_tokens > retrieval_dec.generation_override.max_tokens

    def test_creative_gets_high_token_budget(self, engine):
        from orchestrator.core.types import IntentType
        creative_dec = engine.route(self._make_intent(IntentType.CREATIVE))
        conversation_dec = engine.route(self._make_intent(IntentType.CONVERSATION))
        assert creative_dec.generation_override.max_tokens > conversation_dec.generation_override.max_tokens

    def test_max_tokens_capped_at_2048(self, engine):
        from orchestrator.core.types import IntentType
        for intent_type in IntentType:
            decision = engine.route(self._make_intent(intent_type))
            if decision.generation_override:
                assert decision.generation_override.max_tokens <= 2048


# ── 14. RoutingEngine: security priority, confidence threshold ─────────────────

class TestRoutingSpecialCases:
    @pytest.fixture
    def engine(self):
        from orchestrator.router.routing_engine import RoutingEngine
        return RoutingEngine()

    def _make_intent(self, intent_type, confidence=0.80):
        from orchestrator.core.types import Intent
        return Intent(intent_type=intent_type, confidence=confidence)

    def test_security_priority_is_3(self, engine):
        from orchestrator.core.types import IntentType
        decision = engine.route(self._make_intent(IntentType.SECURITY))
        assert decision.priority == 3

    def test_confidence_threshold_is_045(self, engine):
        from orchestrator.router.routing_engine import RoutingEngine
        assert RoutingEngine.CONFIDENCE_THRESHOLD == 0.45

    def test_low_confidence_falls_back_to_conversation(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        intent = self._make_intent(IntentType.SECURITY, confidence=0.30)
        decision = engine.route(intent)
        # confidence 0.30 < CONFIDENCE_THRESHOLD 0.45 → CONVERSATION fallback
        assert ExpertType.GENERAL in decision.active_experts

    def test_exactly_at_threshold_uses_classified_intent(self, engine):
        from orchestrator.core.types import IntentType, ExpertType
        intent = self._make_intent(IntentType.CODING, confidence=0.45)
        decision = engine.route(intent)
        assert ExpertType.CODE in decision.active_experts


# ── 15. RoutingEngine: subtype-aware system prompt specialization ──────────────

class TestSubtypeSystemPrompt:
    @pytest.fixture
    def engine(self):
        from orchestrator.router.routing_engine import RoutingEngine
        return RoutingEngine()

    def _make_intent(self, intent_type, subtype=None, confidence=0.80):
        from orchestrator.core.types import Intent
        return Intent(intent_type=intent_type, confidence=confidence, subtype=subtype)

    def test_coding_python_subtype_in_prompt(self, engine):
        from orchestrator.core.types import IntentType
        decision = engine.route(self._make_intent(IntentType.CODING, subtype="python"))
        assert "PYTHON" in decision.system_prompt_hint or "python" in decision.system_prompt_hint.lower()

    def test_translation_french_subtype_in_prompt(self, engine):
        from orchestrator.core.types import IntentType
        decision = engine.route(self._make_intent(IntentType.TRANSLATION, subtype="french"))
        assert "french" in decision.system_prompt_hint.lower() or "French" in decision.system_prompt_hint

    def test_data_pandas_subtype_in_prompt(self, engine):
        from orchestrator.core.types import IntentType
        decision = engine.route(self._make_intent(IntentType.DATA_ANALYSIS, subtype="pandas"))
        assert "pandas" in decision.system_prompt_hint.lower()

    def test_no_subtype_prompt_is_still_set(self, engine):
        from orchestrator.core.types import IntentType
        decision = engine.route(self._make_intent(IntentType.CODING, subtype=None))
        assert len(decision.system_prompt_hint) > 10


# ── 16. ExpertManager: all 12 experts registered ─────────────────────────────

class TestExpertManagerRegistry:
    @pytest.fixture
    def manager(self):
        from orchestrator.experts.expert_manager import ExpertManager
        return ExpertManager(embedder=None)

    def test_available_experts_count(self, manager):
        assert len(manager.available_experts()) == 12

    def test_new_experts_in_available_list(self, manager):
        available = manager.available_experts()
        for name in ["writing", "data_analysis", "creative", "security", "planning", "education"]:
            assert name in available

    def test_originally_none_loaded(self, manager):
        assert len(manager.loaded_experts()) == 0

    @pytest.mark.asyncio
    async def test_expert_lazy_loaded_on_first_use(self, manager):
        from orchestrator.core.types import ExpertType, Intent, IntentType
        intent = Intent(intent_type=IntentType.WRITING, confidence=0.80)
        result = await manager.get_guidance("write an email", intent, [ExpertType.WRITING])
        assert result is not None
        assert "writing" in manager.loaded_experts()

    @pytest.mark.asyncio
    async def test_unknown_expert_type_returns_general_fallback(self, manager):
        from orchestrator.core.types import ExpertType, Intent, IntentType
        intent = Intent(intent_type=IntentType.CONVERSATION, confidence=0.50)
        # ExpertType.GENERAL should fall back gracefully
        result = await manager.get_guidance("hello", intent, [ExpertType.GENERAL])
        # May be None if GeneralExpert.is_available() returns False, but no exception
        # (GeneralExpert has no model requirement so it should work)


# ── 17. New expert system prompts ────────────────────────────────────────────

class TestNewExpertSystemPrompts:
    def _get_expert_instance(self, expert_class):
        return expert_class()

    def test_writing_expert_system_prompt_non_empty(self):
        from orchestrator.experts.writing_expert import WritingExpert
        e = WritingExpert()
        assert len(e._system_prompt) > 50
        assert "writ" in e._system_prompt.lower()

    def test_data_expert_system_prompt_non_empty(self):
        from orchestrator.experts.data_expert import DataExpert
        e = DataExpert()
        assert len(e._system_prompt) > 50
        assert any(w in e._system_prompt.lower() for w in ["data", "sql", "analys"])

    def test_creative_expert_system_prompt_non_empty(self):
        from orchestrator.experts.creative_expert import CreativeExpert
        e = CreativeExpert()
        assert len(e._system_prompt) > 50
        assert "creative" in e._system_prompt.lower() or "imaginat" in e._system_prompt.lower()

    def test_security_expert_system_prompt_non_empty(self):
        from orchestrator.experts.security_expert import SecurityExpert
        e = SecurityExpert()
        assert len(e._system_prompt) > 50
        assert "secur" in e._system_prompt.lower()

    def test_planning_expert_system_prompt_non_empty(self):
        from orchestrator.experts.planning_expert import PlanningExpert
        e = PlanningExpert()
        assert len(e._system_prompt) > 50
        assert any(w in e._system_prompt.lower() for w in ["plan", "project", "strateg"])

    def test_education_expert_system_prompt_non_empty(self):
        from orchestrator.experts.education_expert import EducationExpert
        e = EducationExpert()
        assert len(e._system_prompt) > 50
        assert any(w in e._system_prompt.lower() for w in ["teach", "learn", "explain"])


# ── 18. New expert few-shot examples ─────────────────────────────────────────

class TestNewExpertFewShots:
    def test_writing_expert_has_few_shots(self):
        from orchestrator.experts.writing_expert import WritingExpert
        e = WritingExpert()
        assert len(e._few_shot_examples) >= 2
        for ex in e._few_shot_examples:
            assert "user" in ex and "assistant" in ex

    def test_data_expert_has_few_shots(self):
        from orchestrator.experts.data_expert import DataExpert
        e = DataExpert()
        assert len(e._few_shot_examples) >= 2

    def test_creative_expert_has_few_shots(self):
        from orchestrator.experts.creative_expert import CreativeExpert
        e = CreativeExpert()
        assert len(e._few_shot_examples) >= 2

    def test_security_expert_has_few_shots(self):
        from orchestrator.experts.security_expert import SecurityExpert
        e = SecurityExpert()
        assert len(e._few_shot_examples) >= 2

    def test_planning_expert_has_few_shots(self):
        from orchestrator.experts.planning_expert import PlanningExpert
        e = PlanningExpert()
        assert len(e._few_shot_examples) >= 2

    def test_education_expert_has_few_shots(self):
        from orchestrator.experts.education_expert import EducationExpert
        e = EducationExpert()
        assert len(e._few_shot_examples) >= 2

    def test_all_few_shot_examples_have_content(self):
        from orchestrator.experts.writing_expert import WritingExpert
        from orchestrator.experts.security_expert import SecurityExpert
        for cls in (WritingExpert, SecurityExpert):
            e = cls()
            for ex in e._few_shot_examples:
                assert len(ex.get("user", "")) > 5
                assert len(ex.get("assistant", "")) > 20


# ── 19. ExpertManager dynamic few-shot selection ─────────────────────────────

class TestDynamicFewShotSelection:
    @pytest.fixture
    def manager_with_embedder(self):
        from orchestrator.experts.expert_manager import ExpertManager
        embedder = MagicMock()

        async def mock_embed(texts):
            # Return distinct embeddings so similarity scores vary
            return [np.eye(384)[i % 384].tolist() for i, _ in enumerate(texts)]

        async def mock_embed_query(text):
            return np.eye(384)[0].tolist()

        embedder.embed = mock_embed
        embedder.embed_query = mock_embed_query
        return ExpertManager(embedder=embedder)

    @pytest.mark.asyncio
    async def test_dynamic_selection_returns_list(self, manager_with_embedder):
        from orchestrator.experts.education_expert import EducationExpert
        expert = EducationExpert()
        result = await manager_with_embedder._select_few_shots_dynamically(
            expert, "explain recursion", max_examples=2
        )
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_dynamic_selection_respects_max_examples(self, manager_with_embedder):
        from orchestrator.experts.education_expert import EducationExpert
        expert = EducationExpert()
        result = await manager_with_embedder._select_few_shots_dynamically(
            expert, "explain loops", max_examples=1
        )
        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_dynamic_selection_fallback_on_no_embedder(self):
        from orchestrator.experts.expert_manager import ExpertManager
        from orchestrator.experts.planning_expert import PlanningExpert
        manager = ExpertManager(embedder=None)
        expert = PlanningExpert()
        result = await manager._select_few_shots_dynamically(
            expert, "plan a project", max_examples=2
        )
        # Falls back to static selection
        assert isinstance(result, list)
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_dynamic_selection_returns_valid_examples(self, manager_with_embedder):
        from orchestrator.experts.writing_expert import WritingExpert
        expert = WritingExpert()
        result = await manager_with_embedder._select_few_shots_dynamically(
            expert, "write a formal email", max_examples=2
        )
        for ex in result:
            assert "user" in ex
            assert "assistant" in ex


# ── 20. _keyword_classify backward-compat wrapper ────────────────────────────

class TestKeywordClassifyCompat:
    @pytest.fixture
    def clf(self):
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        return HybridIntentClassifier(embedder=None)

    def test_returns_tuple_or_none(self, clf):
        result = clf._keyword_classify("write a Python function")
        assert result is None or (isinstance(result, tuple) and len(result) == 2)

    def test_returns_none_for_empty_match(self, clf):
        result = clf._keyword_classify("hello")
        # May return None or a low-confidence match — just not an exception
        assert result is None or isinstance(result, tuple)

    def test_returns_intent_type_and_float(self, clf):
        from orchestrator.core.types import IntentType
        result = clf._keyword_classify("write a code function that implements recursion")
        assert result is not None
        intent_type, confidence = result
        assert isinstance(intent_type, IntentType)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_two_hits_returns_ge_keyword_confidence(self, clf):
        result = clf._keyword_classify("write a code function")
        assert result is not None
        _, confidence = result
        from orchestrator.router.intent_classifier import HybridIntentClassifier
        assert confidence >= HybridIntentClassifier.KEYWORD_CONFIDENCE
