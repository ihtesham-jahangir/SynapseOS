"""
Adaptive Context Builder.

Takes the deduplicated, ranked list of context items and fits them
into the available token budget, then assembles the final FusedContext
that gets passed to TinyLlama.

Budget allocation strategy (token reserve hierarchy):
  Total budget = context_size - generation_max_tokens
  ├── System prompt reserve (256 tokens)
  ├── Expert guidance (up to 512 tokens)
  ├── Conversation reserve (last N turns, 512 tokens)
  └── Retrieved context (remaining budget)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from orchestrator.core.types import (
    ContextItem,
    ExpertGuidance,
    FusedContext,
    MemoryItem,
    Message,
)
from orchestrator.config.settings import get_settings
from orchestrator.utils.token_counter import count_tokens, truncate_to_budget
from orchestrator.utils.text_utils import format_context_block
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class AdaptiveContextBuilder:
    """
    Assembles FusedContext within strict token budgets.

    The builder operates in three phases:
    1. Reserve mandatory slots (system prompt, recent conversation)
    2. Fill optional context slots (memory, RAG) in priority order
    3. Inject expert guidance into system prompt
    """

    def __init__(self) -> None:
        cfg = get_settings().fusion
        self._total_budget = cfg.context_token_budget
        self._system_reserve = cfg.system_token_reserve
        self._conv_reserve = cfg.conversation_token_reserve

    def build(
        self,
        system_prompt: str,
        scored_context_items: List[Tuple[ContextItem, float]],  # (item, score)
        conversation: List[Message],
        expert_guidance: Optional[ExpertGuidance] = None,
    ) -> FusedContext:
        # Phase 1: account for mandatory slots
        system_prompt_extended = self._inject_expert_into_system(system_prompt, expert_guidance)
        system_tokens = count_tokens(system_prompt_extended)

        conv_tokens, trimmed_conv = self._trim_conversation(conversation)

        reserved = system_tokens + conv_tokens
        context_budget = self._total_budget - reserved

        # Phase 2: fill context slots greedily by score
        selected_items: List[ContextItem] = []
        used_context_tokens = 0

        for item, _score in scored_context_items:
            if used_context_tokens + item.token_count > context_budget:
                # Try truncating the item to fit
                remaining = context_budget - used_context_tokens
                if remaining > 50:  # only if enough room for meaningful content
                    truncated = truncate_to_budget(item.content, remaining)
                    item = item.model_copy(
                        update={
                            "content": truncated,
                            "token_count": count_tokens(truncated),
                        }
                    )
                    selected_items.append(item)
                    used_context_tokens += item.token_count
                break
            selected_items.append(item)
            used_context_tokens += item.token_count

        total_tokens = system_tokens + conv_tokens + used_context_tokens
        budget_fraction = total_tokens / max(self._total_budget, 1)

        log.debug(
            "Context built",
            system_tokens=system_tokens,
            conv_tokens=conv_tokens,
            context_tokens=used_context_tokens,
            context_items=len(selected_items),
            budget_used=f"{budget_fraction:.1%}",
        )

        return FusedContext(
            system_prompt=system_prompt_extended,
            context_items=selected_items,
            conversation_turns=trimmed_conv,
            expert_guidance=expert_guidance,
            total_token_count=total_tokens,
            token_budget_used=budget_fraction,
        )

    def _inject_expert_into_system(
        self,
        base_system: str,
        expert: Optional[ExpertGuidance],
    ) -> str:
        if expert is None:
            return base_system
        expert_section = f"\n\n### Expert Mode: {expert.expert_type.value.title()}\n{expert.system_prompt}"
        combined = base_system + expert_section
        # Hard-cap system prompt
        return truncate_to_budget(combined, self._system_reserve + expert.token_count)

    def _trim_conversation(
        self,
        messages: List[Message],
    ) -> Tuple[int, List[Message]]:
        """
        Keep as many recent messages as fit in the conversation reserve.
        Always preserves the last user message (mandatory).
        """
        if not messages:
            return 0, []

        budget = self._conv_reserve
        kept: List[Message] = []
        used = 0

        for msg in reversed(messages):
            tok = count_tokens(msg.content)
            if used + tok > budget and kept:
                break
            kept.insert(0, msg)
            used += tok

        return used, kept
