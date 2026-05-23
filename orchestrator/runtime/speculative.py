"""
Speculative Decoding support (optional).

Architecture: draft model proposes K tokens → verifier (TinyLlama) accepts
or rejects them in parallel → accepted tokens committed.

This module provides the framework. Full speculative decoding requires:
  - A fast draft model (e.g. 50M-param distilled model)
  - Batch verification support in llama.cpp (--parallel flag)

When a draft model is not configured, falls back to standard generation.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List, Optional

from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class SpeculativeDecoder:
    """
    Speculative decoding wrapper.

    In production mode (with a draft model):
      - Draft model proposes K tokens per step (default K=5)
      - Verifier checks all K in one forward pass
      - Accepted tokens committed, rejected → revert to last accepted

    Current implementation: pass-through to verifier only.
    Enable full mode by providing a draft_client pointing to
    a secondary llama.cpp server running the smaller draft model.
    """

    def __init__(
        self,
        verifier_client,
        draft_client=None,
        k_tokens: int = 5,
    ) -> None:
        self._verifier = verifier_client
        self._draft = draft_client
        self._k = k_tokens
        self._enabled = draft_client is not None

        if self._enabled:
            log.info("Speculative decoding enabled", k_tokens=k_tokens)
        else:
            log.info("Speculative decoding disabled (no draft model configured)")

    async def generate(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        if not self._enabled:
            return await self._verifier.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        # Full speculative decoding path
        return await self._speculative_generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def _speculative_generate(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> str:
        """
        Speculative generation loop.

        Note: True batch verification requires llama.cpp server-level support.
        This implementation approximates it via parallel async calls.
        """
        generated_tokens: List[str] = []
        remaining = max_tokens

        while remaining > 0:
            # Phase 1: Draft K tokens
            draft_response = await self._draft.chat(
                messages=messages,
                max_tokens=min(self._k, remaining),
                temperature=temperature * 1.1,  # Slightly higher temp for draft
            )
            draft_tokens = draft_response.split()

            if not draft_tokens:
                break

            # Phase 2: Verify with main model
            # Build extended context for verification
            verify_messages = messages + [{"role": "assistant", "content": " ".join(generated_tokens)}]
            verify_response = await self._verifier.chat(
                messages=verify_messages,
                max_tokens=len(draft_tokens) + 1,
                temperature=temperature,
            )
            verify_tokens = verify_response.split()

            # Accept matching prefix
            accepted = 0
            for d, v in zip(draft_tokens, verify_tokens):
                if d.lower() == v.lower():
                    accepted += 1
                    generated_tokens.append(d)
                else:
                    # Accept verifier's token at divergence point
                    if v:
                        generated_tokens.append(v)
                    break

            remaining -= max(accepted, 1)

            if not verify_tokens or verify_response.endswith((".", "?", "!")):
                break

        return " ".join(generated_tokens)

    async def stream_generate(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Streaming always uses the verifier directly (draft not streamed)."""
        async for token in self._verifier.stream_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield token
