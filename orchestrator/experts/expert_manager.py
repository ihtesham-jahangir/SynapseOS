"""
Expert Manager – v3.5.

Manages expert lifecycle:
- Lazy instantiation (experts loaded on first use)
- Async concurrent multi-expert execution
- Confidence-weighted expert merging
- Graceful degradation on failure
- Dynamic few-shot selection via embedding similarity (when embedder available)
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple, Type

from orchestrator.core.types import ExpertGuidance, ExpertType, Intent
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.async_utils import gather_with_fallback

from .base_expert import ExpertBase
from .code_expert import CodeExpert
from .math_expert import MathExpert
from .translation_expert import TranslationExpert
from .summarization_expert import SummarizationExpert
from .reasoning_expert import ReasoningExpert
from .writing_expert import WritingExpert
from .data_expert import DataExpert
from .creative_expert import CreativeExpert
from .security_expert import SecurityExpert
from .planning_expert import PlanningExpert
from .education_expert import EducationExpert

log = get_logger(__name__)

_EXPERT_REGISTRY: Dict[ExpertType, Type[ExpertBase]] = {
    ExpertType.CODE: CodeExpert,
    ExpertType.MATH: MathExpert,
    ExpertType.TRANSLATION: TranslationExpert,
    ExpertType.SUMMARIZATION: SummarizationExpert,
    ExpertType.REASONING: ReasoningExpert,
    # v3.5 new experts
    ExpertType.WRITING: WritingExpert,
    ExpertType.DATA_ANALYSIS: DataExpert,
    ExpertType.CREATIVE: CreativeExpert,
    ExpertType.SECURITY: SecurityExpert,
    ExpertType.PLANNING: PlanningExpert,
    ExpertType.EDUCATION: EducationExpert,
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
    - Async concurrent execution for multi-expert queries
    - Confidence-weighted expert selection
    - Dynamic few-shot selection when an embedder is provided
    - Graceful degradation on failure
    """

    def __init__(self, embedder=None) -> None:
        self._instances: Dict[ExpertType, ExpertBase] = {}
        self._lock = asyncio.Lock()
        self._embedder = embedder  # optional: enables dynamic few-shot selection

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

    async def _select_few_shots_dynamically(
        self,
        expert: ExpertBase,
        query: str,
        max_examples: int = 2,
    ) -> List[Dict]:
        """
        Use embedding similarity to pick the most relevant few-shot examples
        from the expert's pool. Falls back to the default static selection
        when no embedder is available or on error.
        """
        examples = expert._few_shot_examples
        if not examples or len(examples) <= max_examples or self._embedder is None:
            return examples[:max_examples]

        try:
            import numpy as np
            query_emb = np.array(
                await self._embedder.embed_query(query), dtype=np.float32
            )
            texts = [e.get("user", "") for e in examples]
            embs = np.array(await self._embedder.embed(texts), dtype=np.float32)

            scores = embs @ query_emb / (
                (np.linalg.norm(embs, axis=1) * np.linalg.norm(query_emb)) + 1e-10
            )
            top_idx = scores.argsort()[-max_examples:][::-1]
            return [examples[i] for i in sorted(top_idx)]
        except Exception as exc:
            log.debug("Dynamic few-shot selection failed, using static", error=str(exc))
            return examples[:max_examples]

    async def get_guidance(
        self,
        query: str,
        intent: Intent,
        expert_types: List[ExpertType],
    ) -> Optional[ExpertGuidance]:
        """
        Run all requested experts concurrently and return the highest-confidence guidance.
        Falls back to GENERAL if all requested experts fail.

        When multiple experts are requested (multi-expert routing):
        - All run concurrently
        - The highest-confidence result wins as the primary guidance
        - Its few-shot examples are dynamically selected from the best expert
        """
        if not expert_types:
            expert_types = [ExpertType.GENERAL]

        async def _run_expert(etype: ExpertType) -> Optional[ExpertGuidance]:
            expert = await self._get_expert(etype)
            if expert is None:
                return None
            try:
                guidance = await expert.get_guidance(query, intent)
                # Override few-shot examples with dynamically selected ones
                if self._embedder is not None and expert._few_shot_examples:
                    guidance.few_shot_examples = await self._select_few_shots_dynamically(
                        expert, query, max_examples=2
                    )
                return guidance
            except Exception as exc:
                log.error("Expert execution failed", expert=etype.value, error=str(exc))
                return None

        results = await gather_with_fallback(
            *[_run_expert(et) for et in expert_types],
            fallbacks=[None] * len(expert_types),
        )

        # Pick best non-None result by confidence
        valid = [r for r in results if r is not None]
        if not valid:
            fallback = await self._get_expert(ExpertType.GENERAL)
            if fallback:
                return await fallback.get_guidance(query, intent)
            return None

        best = max(valid, key=lambda g: g.confidence)

        if len(valid) > 1:
            log.debug(
                "Multi-expert result selected",
                winner=best.expert_type.value,
                all_experts=[g.expert_type.value for g in valid],
            )

        return best

    def available_experts(self) -> List[str]:
        return [e.value for e in _EXPERT_REGISTRY]

    def loaded_experts(self) -> List[str]:
        return [e.value for e in self._instances]
