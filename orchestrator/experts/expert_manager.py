"""
Expert Manager – lazy-loads, caches, and dispatches expert modules.

Experts are only instantiated on first use (lazy init).
Multiple experts can run concurrently for multi-intent queries.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Type

from orchestrator.core.types import ExpertGuidance, ExpertType, Intent
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.async_utils import gather_with_fallback

from .base_expert import ExpertBase
from .code_expert import CodeExpert
from .math_expert import MathExpert
from .translation_expert import TranslationExpert
from .summarization_expert import SummarizationExpert
from .reasoning_expert import ReasoningExpert

log = get_logger(__name__)

_EXPERT_REGISTRY: Dict[ExpertType, Type[ExpertBase]] = {
    ExpertType.CODE: CodeExpert,
    ExpertType.MATH: MathExpert,
    ExpertType.TRANSLATION: TranslationExpert,
    ExpertType.SUMMARIZATION: SummarizationExpert,
    ExpertType.REASONING: ReasoningExpert,
}


class GeneralExpert(ExpertBase):
    expert_type = "general"
    _system_prompt = (
        "You are a helpful AI assistant. "
        "Respond clearly and concisely. If you are unsure about something, say so."
    )


_EXPERT_REGISTRY[ExpertType.GENERAL] = GeneralExpert


class ExpertManager:
    """
    Manages expert lifecycle:
    - Lazy instantiation
    - Async concurrent execution
    - Confidence-weighted expert selection
    - Graceful degradation on failure
    """

    def __init__(self) -> None:
        self._instances: Dict[ExpertType, ExpertBase] = {}
        self._lock = asyncio.Lock()

    async def _get_expert(self, expert_type: ExpertType) -> Optional[ExpertBase]:
        if expert_type in self._instances:
            return self._instances[expert_type]

        async with self._lock:
            if expert_type in self._instances:
                return self._instances[expert_type]

            cls = _EXPERT_REGISTRY.get(expert_type)
            if cls is None:
                log.warning("Unknown expert type", expert=expert_type.value)
                return None

            expert = cls()
            if not expert.is_available():
                log.warning("Expert not available", expert=expert_type.value)
                return None

            self._instances[expert_type] = expert
            log.debug("Expert loaded", expert=expert_type.value)
            return expert

    async def get_guidance(
        self,
        query: str,
        intent: Intent,
        expert_types: List[ExpertType],
    ) -> Optional[ExpertGuidance]:
        """
        Run all requested experts concurrently and return the highest-confidence guidance.
        Falls back to GENERAL if all requested experts fail.
        """
        if not expert_types:
            expert_types = [ExpertType.GENERAL]

        async def _run_expert(etype: ExpertType) -> Optional[ExpertGuidance]:
            expert = await self._get_expert(etype)
            if expert is None:
                return None
            try:
                return await expert.get_guidance(query, intent)
            except Exception as exc:
                log.error("Expert execution failed", expert=etype.value, error=str(exc))
                return None

        results = await gather_with_fallback(
            *[_run_expert(et) for et in expert_types],
            fallbacks=[None] * len(expert_types),
        )

        # Pick best non-None result
        valid = [r for r in results if r is not None]
        if not valid:
            # Last resort: general expert
            fallback = await self._get_expert(ExpertType.GENERAL)
            if fallback:
                return await fallback.get_guidance(query, intent)
            return None

        return max(valid, key=lambda g: g.confidence)

    def available_experts(self) -> List[str]:
        return [e.value for e in _EXPERT_REGISTRY]
