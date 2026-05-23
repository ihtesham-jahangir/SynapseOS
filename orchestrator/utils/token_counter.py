"""
Token counting utilities using tiktoken (cl100k_base).
TinyLlama's tokenizer is different, but for budget tracking this is close
enough without pulling in the full HF tokenizer (keeps cold-start fast).
"""
from __future__ import annotations

import functools
from typing import Dict, List, Optional

import tiktoken

from .logging_utils import get_logger

log = get_logger(__name__)

_ENCODING_CACHE: Dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str = "cl100k_base") -> tiktoken.Encoding:
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.get_encoding(model)
        except Exception:
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODING_CACHE[model]


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    if not text:
        return 0
    enc = _get_encoding(model)
    return len(enc.encode(text, disallowed_special=()))


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "cl100k_base") -> int:
    """Estimate tokens for a list of {role, content} dicts."""
    total = 0
    for msg in messages:
        # Per-message overhead (role tokens + separators ≈ 4)
        total += 4
        total += count_tokens(msg.get("content", ""), model)
        total += count_tokens(msg.get("role", ""), model)
    total += 2  # priming tokens
    return total


def truncate_to_budget(text: str, budget: int, model: str = "cl100k_base") -> str:
    """Truncate text to fit within token budget, preserving whole words."""
    if count_tokens(text, model) <= budget:
        return text
    enc = _get_encoding(model)
    tokens = enc.encode(text, disallowed_special=())
    truncated = tokens[:budget]
    return enc.decode(truncated)


def split_by_token_budget(
    texts: List[str],
    total_budget: int,
    model: str = "cl100k_base",
) -> List[str]:
    """
    Greedily select texts from the list until the token budget is exhausted.
    Returns as many complete texts as fit, in order.
    """
    selected: List[str] = []
    used = 0
    for text in texts:
        cost = count_tokens(text, model)
        if used + cost <= total_budget:
            selected.append(text)
            used += cost
        else:
            break
    return selected
