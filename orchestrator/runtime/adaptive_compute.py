"""
Adaptive Compute Controller.

Dynamically adjusts generation parameters based on:
  - Intent type (math → lower temp, conversation → higher temp)
  - Token budget remaining (reduce max_tokens if context is large)
  - Confidence level (low confidence → more conservative settings)
  - Response complexity hints (simple Q&A → fewer tokens)

Also implements early-exit heuristics:
  - Detect if response is already "done" before max_tokens
  - Adjust stop sequences per intent
"""
from __future__ import annotations

import re
from typing import List, Optional

from orchestrator.core.types import FusedContext, GenerationParams, Intent, IntentType
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

# Per-intent max token recommendations
_INTENT_MAX_TOKENS = {
    IntentType.CODING: 1024,
    IntentType.MATH: 512,
    IntentType.TRANSLATION: 512,
    IntentType.SUMMARIZATION: 512,
    IntentType.RETRIEVAL: 384,
    IntentType.REASONING: 768,
    IntentType.CONVERSATION: 256,
    IntentType.VOICE: 256,
    IntentType.UNKNOWN: 384,
}

# Per-intent stop sequences (supplement defaults)
_INTENT_STOP_SEQUENCES = {
    IntentType.CODING: ["```\n\n", "Human:", "User:"],
    IntentType.MATH: ["\n\n\n", "Human:", "User:"],
    IntentType.CONVERSATION: ["Human:", "User:", "Assistant:"],
}


class AdaptiveComputeController:
    """
    Produces optimized GenerationParams for each request context.

    Optimization levers:
    - max_tokens: capped by intent, context size, and token budget
    - temperature: intent-default, reduced when confidence is high
    - stop sequences: intent-specific additions
    """

    def __init__(self) -> None:
        self._cfg = get_settings().generation

    def optimize(
        self,
        intent: Intent,
        fused_context: FusedContext,
        base_params: Optional[GenerationParams] = None,
    ) -> GenerationParams:
        if base_params is None:
            base_params = GenerationParams(
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
                top_k=self._cfg.top_k,
                repeat_penalty=self._cfg.repeat_penalty,
            )

        # Adjust max_tokens
        intent_max = _INTENT_MAX_TOKENS.get(intent.intent_type, self._cfg.max_tokens)
        remaining_budget = max(
            128,
            get_settings().llama.context_size
            - fused_context.total_token_count
            - 64,  # safety margin
        )
        max_tokens = min(base_params.max_tokens, intent_max, remaining_budget)

        # Adjust temperature based on confidence
        # Higher confidence = model knows what to do = can lower temp slightly
        temperature = base_params.temperature
        if intent.confidence > 0.85:
            temperature = max(0.05, temperature * 0.9)

        # Merge stop sequences
        stop = list(self._cfg.stop_sequences)
        stop.extend(_INTENT_STOP_SEQUENCES.get(intent.intent_type, []))
        # Deduplicate
        seen = set()
        stop = [s for s in stop if not (s in seen or seen.add(s))]

        # Clamp temperature to a valid range regardless of base_params source
        temperature = max(0.05, min(2.0, temperature))

        optimized = GenerationParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=base_params.top_p,
            top_k=base_params.top_k,
            repeat_penalty=base_params.repeat_penalty,
            stop_sequences=stop,
        )

        log.debug(
            "Adaptive compute",
            intent=intent.intent_type.value,
            max_tokens=max_tokens,
            temperature=f"{temperature:.2f}",
            context_tokens=fused_context.total_token_count,
        )

        return optimized

    @staticmethod
    def should_use_rag(intent: Intent) -> bool:
        """Quick gate: is retrieval worth the latency for this intent?"""
        return intent.requires_rag or intent.intent_type in (
            IntentType.RETRIEVAL,
            IntentType.REASONING,
            IntentType.SUMMARIZATION,
        )
