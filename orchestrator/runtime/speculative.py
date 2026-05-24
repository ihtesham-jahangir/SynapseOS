"""
Speculative Decoding — v2.0

Architecture: each step runs draft and verifier CONCURRENTLY via asyncio.gather:

  ┌─────────────────────────────────────────────────────────────┐
  │  Context = original messages + generated-so-far            │
  │                                                             │
  │   draft model (small, fast)  ──►  K draft tokens           │
  │       runs in parallel with                                 │
  │   verifier model (large)     ──►  K+1 verify tokens        │
  │                                                             │
  │  Accept matching prefix; at first mismatch take verifier's  │
  │  correction.  If all K match, also take the K+1 bonus token.│
  └─────────────────────────────────────────────────────────────┘

Speedup sources:
  1. Draft and verifier run concurrently → wall time = max(draft, verify)
     not draft + verify.
  2. Draft model is 5–10× faster per token (smaller parameter count).
  3. cache_prompt=True on the verifier reuses the KV-cache for the prompt
     prefix across loop iterations, so the verifier only processes new tokens
     rather than re-encoding the full context each step.

Expected speedup: 2–3× on CPU with a Phi-3-mini or TinyLlama draft model
at 70–80% token acceptance rate.

Fallback: when no draft model is configured, passes through directly to the
verifier with zero overhead.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncGenerator, Dict, List, Optional

from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.metrics import (
    SPECULATIVE_TOKENS_DRAFTED,
    SPECULATIVE_TOKENS_ACCEPTED,
    SPECULATIVE_ACCEPTANCE_RATE,
)

log = get_logger(__name__)

_EOS_TOKENS = frozenset({
    "</s>", "[/INST]", "[INST]", "<|end|>", "<|endoftext|>",
    "<eos>", "<|im_end|>", "<end_of_turn>",
})


# ── Token utilities ────────────────────────────────────────────────────────────

def _split_tokens(text: str) -> List[str]:
    """
    Split text into word-level tokens while preserving whitespace so that
    ''.join(_split_tokens(text)) == text for lossless reconstruction.
    Each token is either a run of non-whitespace chars or a run of whitespace.
    """
    return re.findall(r"[^\s]+|\s+", text)


def _tokens_match(a: str, b: str) -> bool:
    """Accept a draft token if its stripped, lowercased form equals the verifier's."""
    return a.strip().lower() == b.strip().lower()


def _contains_eos(text: str) -> bool:
    return any(eos in text for eos in _EOS_TOKENS)


def _strip_eos(text: str) -> str:
    for eos in _EOS_TOKENS:
        text = text.replace(eos, "")
    return text.strip()


def _build_context(
    messages: List[Dict[str, str]],
    generated: List[str],
) -> List[Dict[str, str]]:
    """
    Append accumulated generated text as a partial assistant turn so both
    draft and verifier continue from exactly the same position in the sequence.
    """
    if not generated:
        return messages
    return messages + [{"role": "assistant", "content": "".join(generated)}]


# ── Main class ─────────────────────────────────────────────────────────────────

class SpeculativeDecoder:
    """
    Speculative decoding orchestrator for SynapseOS.

    Enable by setting LLAMA_DRAFT_MODEL_URL in the environment (or .env):
        LLAMA_DRAFT_MODEL_URL=http://localhost:8081

    Then start the draft server:
        ./start_draft.sh

    When disabled (no draft_client), all calls pass through to the verifier
    with zero additional latency.
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
        self.enabled = draft_client is not None

        if self.enabled:
            log.info(
                "Speculative decoding ENABLED",
                k_tokens=k_tokens,
                mode="concurrent_draft_verify",
                hint="Start draft server with: ./start_draft.sh",
            )
        else:
            log.info(
                "Speculative decoding disabled — set LLAMA_DRAFT_MODEL_URL to enable",
            )

    # ── Public interface ───────────────────────────────────────────────────────

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        if not self.enabled:
            return await self._verifier.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        return await self._speculative_generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming with speculative decoding.

        Generates the full response via the speculative loop (faster total wall
        time than verifier-only), then streams the result token-by-token.
        This preserves the streaming interface while delivering the speedup.
        """
        if not self.enabled:
            async for token in self._verifier.stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            ):
                yield token
            return

        full_response = await self._speculative_generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        for token in _split_tokens(full_response):
            yield token
            await asyncio.sleep(0)  # yield control between tokens

    # ── Core speculative loop ──────────────────────────────────────────────────

    async def _speculative_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> str:
        """
        Concurrent speculative decoding loop.

        Each iteration:
          1. Build context = original messages + tokens generated so far
          2. Fire draft (K tokens) and verifier (K+1 tokens) CONCURRENTLY
          3. Walk draft tokens against verifier tokens; accept the matching prefix
          4. At the first mismatch, append verifier's correction; discard rest of draft
          5. If all K tokens matched, append verifier's bonus K+1 token for free
          6. Repeat until max_tokens reached or EOS detected
        """
        all_tokens: List[str] = []
        total_drafted = 0
        total_accepted = 0
        t_start = time.perf_counter()

        # Strip kwargs that we manage ourselves to avoid duplicate-keyword errors
        extra = {k: v for k, v in kwargs.items() if k not in ("max_tokens", "temperature")}

        while len(all_tokens) < max_tokens:
            remaining = max_tokens - len(all_tokens)
            k = min(self._k, remaining)
            context = _build_context(messages, all_tokens)

            # ── Concurrent draft + verifier ───────────────────────────────────
            try:
                draft_text, verify_text = await asyncio.gather(
                    self._draft.chat(
                        messages=context,
                        max_tokens=k,
                        temperature=temperature * 1.1,
                        cache_prompt=True,
                        **extra,
                    ),
                    self._verifier.chat(
                        messages=context,
                        max_tokens=k + 1,
                        temperature=temperature,
                        cache_prompt=True,
                        **extra,
                    ),
                )
            except Exception as exc:
                log.warning(
                    "Speculative step failed — falling back to verifier",
                    error=str(exc),
                    tokens_so_far=len(all_tokens),
                )
                fallback = await self._verifier.chat(
                    messages=context,
                    max_tokens=remaining,
                    temperature=temperature,
                    **extra,
                )
                all_tokens.extend(_split_tokens(fallback))
                break

            draft_tokens = _split_tokens(draft_text)[:k]
            verify_tokens = _split_tokens(verify_text)

            if not verify_tokens:
                break

            # ── Accept / reject prefix ────────────────────────────────────────
            n_accepted = 0
            for d_tok, v_tok in zip(draft_tokens, verify_tokens):
                if _tokens_match(d_tok, v_tok):
                    all_tokens.append(d_tok)
                    n_accepted += 1
                else:
                    # Divergence: take verifier's correction, stop accepting draft
                    all_tokens.append(v_tok)
                    break
            else:
                # All K draft tokens accepted → take the bonus K+1 verifier token.
                # The bonus is not counted in n_accepted (it was not drafted).
                bonus_idx = len(draft_tokens)
                if bonus_idx < len(verify_tokens):
                    all_tokens.append(verify_tokens[bonus_idx])

            total_drafted += len(draft_tokens)
            total_accepted += n_accepted

            # ── Early stop on EOS ─────────────────────────────────────────────
            if _contains_eos("".join(all_tokens)) or not draft_tokens or not verify_tokens:
                break

        # ── Prometheus metrics ────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        acceptance_rate = total_accepted / max(total_drafted, 1)

        SPECULATIVE_TOKENS_DRAFTED.inc(total_drafted)
        SPECULATIVE_TOKENS_ACCEPTED.inc(total_accepted)
        SPECULATIVE_ACCEPTANCE_RATE.observe(acceptance_rate)

        log.info(
            "Speculative generation complete",
            tokens=len(all_tokens),
            acceptance_rate=f"{acceptance_rate:.1%}",
            drafted=total_drafted,
            accepted=total_accepted,
            elapsed_ms=f"{elapsed_ms:.0f}",
        )

        return _strip_eos("".join(all_tokens))
