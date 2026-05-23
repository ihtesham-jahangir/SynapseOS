"""
Context compressor – uses TinyLlama to summarize overflow conversation turns.
Triggered by MemoryManager when L1 exceeds its threshold.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from orchestrator.core.types import Message, MessageRole, MemoryItem, MemoryLevel
from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_SUMMARIZE_PROMPT = """Summarize the following conversation turns into a concise paragraph
that preserves all key facts, decisions, and context. Be dense and factual.

Conversation:
{turns}

Summary:"""


class ContextCompressor:
    """
    Compresses a list of conversation turns into a single summary
    by calling the TinyLlama runtime (passed in to avoid circular imports).
    """

    def __init__(self, llama_client) -> None:
        self._client = llama_client

    async def compress_turns(
        self,
        turns: List[Message],
        session_id: str,
        turn_start: int = 0,
        turn_end: Optional[int] = None,
    ) -> MemoryItem:
        """Generate a summary MemoryItem for a slice of conversation turns."""
        turn_end = turn_end or len(turns)
        formatted = "\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in turns
        )
        prompt = _SUMMARIZE_PROMPT.format(turns=formatted)

        try:
            summary = await self._client.complete(
                prompt=prompt,
                max_tokens=256,
                temperature=0.3,
            )
        except Exception as exc:
            # Fall back to extractive summary if LLM fails
            log.warning("Summarization LLM call failed, using extractive", error=str(exc))
            summary = self._extractive_summary(turns)

        return MemoryItem(
            id=str(uuid.uuid4()),
            content=summary.strip(),
            level=MemoryLevel.L2,
            session_id=session_id,
            importance_score=0.7,
            token_count=count_tokens(summary),
            metadata={"turn_start": turn_start, "turn_end": turn_end},
        )

    @staticmethod
    def _extractive_summary(turns: List[Message], max_chars: int = 500) -> str:
        """Fallback: concatenate first sentence of each turn."""
        parts = []
        for t in turns:
            first_sent = t.content.split(".")[0][:100]
            parts.append(f"{t.role.value}: {first_sent}")
        summary = " | ".join(parts)
        return summary[:max_chars]
