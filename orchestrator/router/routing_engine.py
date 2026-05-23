"""
Dynamic routing engine.

Maps Intent → set of active processing paths:
  - which experts to activate
  - whether to run RAG
  - whether to query L3/L4 memory
  - generation parameter overrides

This is the "traffic controller" of the orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

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


# ─── Routing table ────────────────────────────────────────────────────────────
# Maps (intent_type) → routing config

_ROUTING_TABLE = {
    IntentType.CODING: {
        "experts": [ExpertType.CODE],
        "run_rag": False,
        "temperature": 0.2,
        "system_hint": "You are an expert software engineer. Produce clean, correct, idiomatic code.",
    },
    IntentType.MATH: {
        "experts": [ExpertType.MATH],
        "run_rag": False,
        "temperature": 0.1,
        "system_hint": "You are a precise mathematician. Show step-by-step reasoning.",
    },
    IntentType.TRANSLATION: {
        "experts": [ExpertType.TRANSLATION],
        "run_rag": False,
        "temperature": 0.3,
        "system_hint": "You are an expert multilingual translator. Preserve tone and meaning.",
    },
    IntentType.SUMMARIZATION: {
        "experts": [ExpertType.SUMMARIZATION],
        "run_rag": True,
        "temperature": 0.4,
        "system_hint": "You are a skilled summarizer. Be concise, accurate, and comprehensive.",
    },
    IntentType.RETRIEVAL: {
        "experts": [],
        "run_rag": True,
        "temperature": 0.5,
        "system_hint": "You are a knowledgeable assistant. Answer based on the provided context.",
    },
    IntentType.REASONING: {
        "experts": [ExpertType.REASONING],
        "run_rag": True,
        "temperature": 0.6,
        "system_hint": "You are a careful analytical reasoner. Think step by step.",
    },
    IntentType.CONVERSATION: {
        "experts": [ExpertType.GENERAL],
        "run_rag": False,
        "temperature": 0.75,
        "system_hint": "You are a helpful, harmless, and honest AI assistant.",
    },
    IntentType.VOICE: {
        "experts": [],
        "run_rag": False,
        "temperature": 0.5,
        "system_hint": "Process the audio/voice content as instructed.",
    },
    IntentType.UNKNOWN: {
        "experts": [ExpertType.GENERAL],
        "run_rag": False,
        "temperature": 0.7,
        "system_hint": "You are a helpful AI assistant.",
    },
}

_BASE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question directly and concisely."
)


class RoutingEngine:
    """
    Translates a classified Intent into a concrete RoutingDecision.

    Also handles confidence-based fallback:
    - High confidence: use primary routing
    - Low confidence: fall back to conversation with RAG
    """

    CONFIDENCE_THRESHOLD = 0.60

    def __init__(self) -> None:
        self._cfg = get_settings().generation

    def route(self, intent: Intent) -> RoutingDecision:
        effective_intent = intent.intent_type

        # Fall back to conversation if classifier is uncertain
        if intent.confidence < self.CONFIDENCE_THRESHOLD:
            effective_intent = IntentType.CONVERSATION
            log.debug(
                "Low confidence intent, routing to CONVERSATION",
                original=intent.intent_type.value,
                confidence=intent.confidence,
            )

        table_entry = _ROUTING_TABLE.get(effective_intent, _ROUTING_TABLE[IntentType.UNKNOWN])

        # Build generation params with intent-specific temperature
        gen_params = GenerationParams(
            max_tokens=self._cfg.max_tokens,
            temperature=table_entry.get("temperature", self._cfg.temperature),
            top_p=self._cfg.top_p,
            top_k=self._cfg.top_k,
            repeat_penalty=self._cfg.repeat_penalty,
        )

        # Compose system prompt
        system_hint = table_entry.get("system_hint", "")
        system_prompt = f"{_BASE_SYSTEM_PROMPT}\n\n{system_hint}".strip()

        decision = RoutingDecision(
            intent=intent,
            active_experts=table_entry.get("experts", []),
            run_rag=table_entry.get("run_rag", False) or intent.requires_rag,
            query_l3=True,
            query_l4=True,
            generation_override=gen_params,
            system_prompt_hint=system_prompt,
        )

        log.debug(
            "Routing decision",
            intent=effective_intent.value,
            experts=[e.value for e in decision.active_experts],
            run_rag=decision.run_rag,
        )
        return decision
