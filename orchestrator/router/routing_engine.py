"""
Dynamic routing engine — v3.5.

Maps Intent → set of active processing paths:
  - which experts to activate (multiple for hybrid queries)
  - whether to run RAG
  - generation parameter overrides (adaptive to intent + confidence)
  - priority tier

v3.5 additions:
  - Multi-expert routing: secondary intents can activate additional experts
  - Adaptive token budget: scaled by intent complexity
  - Adaptive temperature: adjusted per intent + confidence uncertainty
  - Context-sensitive system prompts: subtype-aware specialization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from orchestrator.core.types import ExpertType, Intent, IntentType, GenerationParams
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class RoutingDecision:
    """Fully resolved routing plan for a single request."""
    intent: Intent
    active_experts: List[ExpertType] = field(default_factory=list)
    run_rag: bool = False
    query_l3: bool = True
    query_l4: bool = True
    generation_override: Optional[GenerationParams] = None
    system_prompt_hint: str = ""
    priority: int = 1  # 1=normal, 2=high, 3=critical


# ─── Primary routing table ────────────────────────────────────────────────────
# temperature: lower = more deterministic (code, math); higher = more creative
# max_tokens_multiplier: relative to DEFAULT_MAX_TOKENS setting
# secondary_expert: additional expert for hybrid queries

_ROUTING_TABLE = {
    IntentType.CODING: {
        "experts": [ExpertType.CODE],
        "run_rag": False,
        "temperature": 0.15,
        "max_tokens_multiplier": 1.5,  # code often needs more tokens
        "priority": 2,
        "system_hint": (
            "You are an expert software engineer. Produce clean, correct, "
            "idiomatic code. Always include the complete implementation."
        ),
    },
    IntentType.MATH: {
        "experts": [ExpertType.MATH],
        "run_rag": False,
        "temperature": 0.05,
        "max_tokens_multiplier": 1.2,
        "priority": 2,
        "system_hint": (
            "You are a precise mathematician. Show every step. "
            "Verify arithmetic. Present the final answer prominently."
        ),
    },
    IntentType.TRANSLATION: {
        "experts": [ExpertType.TRANSLATION],
        "run_rag": False,
        "temperature": 0.25,
        "max_tokens_multiplier": 1.0,
        "priority": 1,
        "system_hint": (
            "You are an expert multilingual translator. "
            "Preserve tone, meaning, and cultural nuance faithfully."
        ),
    },
    IntentType.SUMMARIZATION: {
        "experts": [ExpertType.SUMMARIZATION],
        "run_rag": True,
        "temperature": 0.35,
        "max_tokens_multiplier": 0.8,  # summaries should be concise
        "priority": 1,
        "system_hint": (
            "You are a skilled summarizer. Be concise and comprehensive. "
            "Preserve key facts, numbers, and named entities exactly."
        ),
    },
    IntentType.RETRIEVAL: {
        "experts": [],
        "run_rag": True,
        "temperature": 0.40,
        "max_tokens_multiplier": 0.7,
        "priority": 1,
        "system_hint": (
            "You are a knowledgeable assistant. Answer based on the provided context. "
            "If the context does not contain the answer, say so clearly."
        ),
    },
    IntentType.REASONING: {
        "experts": [ExpertType.REASONING],
        "run_rag": True,
        "temperature": 0.55,
        "max_tokens_multiplier": 1.6,  # reasoning needs space
        "priority": 2,
        "system_hint": (
            "You are a rigorous analytical thinker. Think step by step. "
            "Consider multiple perspectives. State your assumptions. "
            "End with a clear, justified conclusion."
        ),
    },
    IntentType.CONVERSATION: {
        "experts": [ExpertType.GENERAL],
        "run_rag": False,
        "temperature": 0.75,
        "max_tokens_multiplier": 0.8,
        "priority": 1,
        "system_hint": "You are a helpful, harmless, and honest AI assistant.",
    },
    IntentType.VOICE: {
        "experts": [],
        "run_rag": False,
        "temperature": 0.40,
        "max_tokens_multiplier": 1.0,
        "priority": 1,
        "system_hint": "Process the audio/voice content as instructed.",
    },
    # ── v3.5 new intents ──────────────────────────────────────────────────────
    IntentType.WRITING: {
        "experts": [ExpertType.WRITING],
        "run_rag": False,
        "temperature": 0.65,
        "max_tokens_multiplier": 1.4,
        "priority": 1,
        "system_hint": (
            "You are a professional writer and editor with expertise in diverse styles — "
            "technical, business, creative, and academic. Match the user's requested "
            "tone and audience. Produce polished, publication-ready text."
        ),
    },
    IntentType.DATA_ANALYSIS: {
        "experts": [ExpertType.DATA_ANALYSIS],
        "run_rag": True,
        "temperature": 0.20,
        "max_tokens_multiplier": 1.5,
        "priority": 2,
        "system_hint": (
            "You are a senior data scientist and analyst. Produce correct, efficient "
            "code and clear analytical insights. Show your reasoning for analytical "
            "conclusions. Always handle edge cases in code."
        ),
    },
    IntentType.CREATIVE: {
        "experts": [ExpertType.CREATIVE],
        "run_rag": False,
        "temperature": 0.90,  # high creativity
        "max_tokens_multiplier": 1.6,
        "priority": 1,
        "system_hint": (
            "You are a creative director and imaginative thinker. Generate original, "
            "unexpected ideas. Avoid clichés. Build on concepts to create rich, "
            "detailed creative work. Think laterally and make surprising connections."
        ),
    },
    IntentType.SECURITY: {
        "experts": [ExpertType.SECURITY],
        "run_rag": True,
        "temperature": 0.10,  # very precise for security
        "max_tokens_multiplier": 1.5,
        "priority": 3,  # highest priority
        "system_hint": (
            "You are a senior security engineer and ethical hacker. Identify real "
            "vulnerabilities with precise technical detail. Reference CVEs and OWASP "
            "standards where relevant. Provide actionable remediation steps."
        ),
    },
    IntentType.PLANNING: {
        "experts": [ExpertType.PLANNING],
        "run_rag": False,
        "temperature": 0.45,
        "max_tokens_multiplier": 1.4,
        "priority": 1,
        "system_hint": (
            "You are an experienced project manager and strategic planner. "
            "Break work into concrete, actionable steps with clear ownership. "
            "Consider dependencies, risks, and resource constraints."
        ),
    },
    IntentType.EDUCATION: {
        "experts": [ExpertType.EDUCATION],
        "run_rag": True,
        "temperature": 0.55,
        "max_tokens_multiplier": 1.5,
        "priority": 1,
        "system_hint": (
            "You are a patient, expert teacher who adapts explanations to the learner's level. "
            "Use analogies, examples, and step-by-step breakdowns. "
            "Check for understanding and anticipate follow-up questions."
        ),
    },
    IntentType.UNKNOWN: {
        "experts": [ExpertType.GENERAL],
        "run_rag": False,
        "temperature": 0.70,
        "max_tokens_multiplier": 1.0,
        "priority": 1,
        "system_hint": "You are a helpful AI assistant.",
    },
}

# Experts that pair well together for cross-domain queries
_SECONDARY_EXPERT_MAP: dict = {
    (IntentType.CODING, IntentType.SECURITY): ExpertType.SECURITY,
    (IntentType.CODING, IntentType.EDUCATION): ExpertType.EDUCATION,
    (IntentType.CODING, IntentType.DATA_ANALYSIS): ExpertType.DATA_ANALYSIS,
    (IntentType.DATA_ANALYSIS, IntentType.CODING): ExpertType.CODE,
    (IntentType.REASONING, IntentType.PLANNING): ExpertType.PLANNING,
    (IntentType.WRITING, IntentType.CREATIVE): ExpertType.CREATIVE,
    (IntentType.EDUCATION, IntentType.CODING): ExpertType.CODE,
    (IntentType.PLANNING, IntentType.REASONING): ExpertType.REASONING,
    (IntentType.SECURITY, IntentType.CODING): ExpertType.CODE,
}

_BASE_SYSTEM_PROMPT = (
    "You are SynapseOS, an expert AI assistant. "
    "Answer the user's request directly, accurately, and concisely."
)


class RoutingEngine:
    """
    Translates a classified Intent into a concrete RoutingDecision.

    v3.5: multi-expert routing for hybrid queries; adaptive token budget
    and temperature; subtype-aware system prompt specialization.
    """

    CONFIDENCE_THRESHOLD = 0.45  # lowered: new intents have well-separated embeddings

    def __init__(self) -> None:
        self._cfg = get_settings().generation

    def route(self, intent: Intent) -> RoutingDecision:
        effective_intent = intent.intent_type

        # Fall back to CONVERSATION if classifier is uncertain
        if intent.confidence < self.CONFIDENCE_THRESHOLD:
            effective_intent = IntentType.CONVERSATION
            log.debug(
                "Low confidence — routing to CONVERSATION",
                original=intent.intent_type.value,
                confidence=intent.confidence,
            )

        table_entry = _ROUTING_TABLE.get(effective_intent, _ROUTING_TABLE[IntentType.UNKNOWN])

        # ── Adaptive generation parameters ────────────────────────────────────
        base_max = self._cfg.max_tokens
        multiplier = table_entry.get("max_tokens_multiplier", 1.0)
        adaptive_max = min(int(base_max * multiplier), 2048)

        # When confidence is low, nudge temperature up slightly for diversity
        base_temp = table_entry.get("temperature", self._cfg.temperature)
        uncertainty_bump = max(0.0, (0.65 - intent.confidence) * 0.15)
        adaptive_temp = min(1.0, base_temp + uncertainty_bump)

        gen_params = GenerationParams(
            max_tokens=adaptive_max,
            temperature=round(adaptive_temp, 3),
            top_p=self._cfg.top_p,
            top_k=self._cfg.top_k,
            repeat_penalty=self._cfg.repeat_penalty,
        )

        # ── System prompt with subtype specialization ──────────────────────────
        system_hint = table_entry.get("system_hint", "")
        if intent.subtype:
            system_hint = _specialize_prompt(system_hint, effective_intent, intent.subtype)
        system_prompt = f"{_BASE_SYSTEM_PROMPT}\n\n{system_hint}".strip()

        # ── Multi-expert selection ─────────────────────────────────────────────
        primary_experts: List[ExpertType] = list(table_entry.get("experts", []))

        # Check secondary intents for additional expert activation
        for secondary in intent.secondary_intents:
            key = (effective_intent, secondary.intent_type)
            extra_expert = _SECONDARY_EXPERT_MAP.get(key)
            if extra_expert and extra_expert not in primary_experts:
                primary_experts.append(extra_expert)
                log.debug(
                    "Adding secondary expert for hybrid intent",
                    primary=effective_intent.value,
                    secondary=secondary.intent_type.value,
                    expert=extra_expert.value,
                )

        decision = RoutingDecision(
            intent=intent,
            active_experts=primary_experts,
            run_rag=table_entry.get("run_rag", False) or intent.requires_rag,
            query_l3=True,
            query_l4=True,
            generation_override=gen_params,
            system_prompt_hint=system_prompt,
            priority=table_entry.get("priority", 1),
        )

        log.debug(
            "Routing decision",
            intent=effective_intent.value,
            experts=[e.value for e in decision.active_experts],
            run_rag=decision.run_rag,
            priority=decision.priority,
            max_tokens=adaptive_max,
            temperature=round(adaptive_temp, 3),
        )
        return decision


def _specialize_prompt(base_hint: str, intent: IntentType, subtype: str) -> str:
    """Inject subtype context into the system prompt hint."""
    if intent == IntentType.CODING:
        return base_hint + f"\n\nThe user is working in **{subtype.upper()}**. Use {subtype}-idiomatic patterns."
    if intent == IntentType.TRANSLATION:
        return base_hint + f"\n\nTarget language: **{subtype.title()}**. Preserve cultural nuance."
    if intent == IntentType.DATA_ANALYSIS:
        return base_hint + f"\n\nPrimary tool: **{subtype}**. Produce tool-idiomatic, optimized code."
    return base_hint
