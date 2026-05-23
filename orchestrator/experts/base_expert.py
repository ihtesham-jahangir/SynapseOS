"""Abstract base for all expert modules."""
from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

from orchestrator.core.base import BaseExpert
from orchestrator.core.types import ExpertGuidance, ExpertType, Intent
from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class ExpertBase(BaseExpert):
    """
    Provides expert-specific system prompts and few-shot examples.
    Experts do NOT run a separate model (optional for lightweight setup).
    They contribute structured guidance that the fusion engine injects
    into TinyLlama's context, effectively specializing its behavior
    without any additional model.

    To use an actual secondary model (Phi-2, TinyCoder, etc.),
    override get_guidance() and call the model there.
    """

    expert_type: str = "base"
    _system_prompt: str = ""
    _few_shot_examples: List[Dict[str, str]] = []

    def is_available(self) -> bool:
        return True  # Pure-prompt experts are always available

    async def get_guidance(self, query: str, intent: Intent) -> ExpertGuidance:
        system_prompt = self._build_system_prompt(query, intent)
        few_shots = self._select_few_shots(query)

        total_tokens = count_tokens(system_prompt) + sum(
            count_tokens(e.get("user", "") + e.get("assistant", ""))
            for e in few_shots
        )

        return ExpertGuidance(
            expert_type=ExpertType(self.expert_type),
            system_prompt=system_prompt,
            few_shot_examples=few_shots,
            confidence=intent.confidence,
            token_count=total_tokens,
        )

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        return self._system_prompt

    def _select_few_shots(self, query: str, max_examples: int = 2) -> List[Dict[str, str]]:
        return self._few_shot_examples[:max_examples]
