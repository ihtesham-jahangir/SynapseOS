"""
Intent classification engine.

Two-stage pipeline:
  Stage 1 – fast keyword/regex matching (microseconds, no model call)
  Stage 2 – semantic similarity to intent exemplars via BGE embeddings
             (triggered only when stage 1 confidence is below threshold)

This keeps cold-path latency near-zero for common patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from orchestrator.core.base import BaseIntentClassifier
from orchestrator.core.types import Intent, IntentType
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.async_utils import AsyncLRUCache

log = get_logger(__name__)

# ─── Keyword rules (Stage 1) ─────────────────────────────────────────────────

_KEYWORD_RULES: Dict[IntentType, List[str]] = {
    IntentType.CODING: [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bscript\b", r"\bdebug\b",
        r"\bprogramm", r"\bimport\b", r"\bdef \w+\(", r"\bsyntax\b",
        r"\bbug\b", r"\berror\b.*line", r"\bimplementi?\b", r"\brefactor\b",
        r"\bgithub\b", r"\bapi endpoint", r"\bunit test", r"\bcompile\b",
    ],
    IntentType.MATH: [
        r"\bsolve\b", r"\bcalculate\b", r"\bequation\b", r"\bintegral\b",
        r"\bderivative\b", r"\bmatrix\b", r"\bprob(?:ability)?\b", r"\bstatistic",
        r"\bformula\b", r"\balgebra\b", r"\bcalculus\b", r"\bcompute\b",
        r"\d+\s*[\+\-\*\/\^]\s*\d+", r"\bsum of\b", r"\baverage\b",
    ],
    IntentType.TRANSLATION: [
        r"\btranslat", r"\bin french\b", r"\bin spanish\b", r"\bin german\b",
        r"\bin japanese\b", r"\bin chinese\b", r"\bin arabic\b",
        r"\bto english\b", r"\bfrom english\b", r"\blanguage\b",
    ],
    IntentType.SUMMARIZATION: [
        r"\bsummar", r"\bbriefly\b", r"\btldr\b", r"\bshorten\b",
        r"\bkey points\b", r"\bmain idea", r"\bcondense\b", r"\bsynopsis\b",
        r"\boverview\b", r"\bgist\b",
    ],
    IntentType.RETRIEVAL: [
        r"\bfind\b", r"\bsearch\b", r"\blook up\b", r"\bwhat is\b",
        r"\bwho is\b", r"\bwhen did\b", r"\bwhere is\b", r"\bhow does\b",
        r"\btell me about\b", r"\bexplain\b",
    ],
    IntentType.REASONING: [
        r"\bwhy\b", r"\banalyze\b", r"\bcompare\b", r"\bevaluate\b",
        r"\bpros and cons\b", r"\bargument\b", r"\blogic\b", r"\bdeduce\b",
        r"\binfer\b", r"\bcritique\b", r"\bassess\b", r"\bthink\b.*step",
    ],
    IntentType.VOICE: [
        r"\btranscrib", r"\bspeak\b", r"\baudio\b", r"\bspeech\b",
        r"\brecogniz", r"\bvoice\b", r"\bwav\b", r"\bmp3\b",
    ],
}

# ─── Exemplar sentences for semantic fallback ─────────────────────────────────

_INTENT_EXEMPLARS: Dict[IntentType, List[str]] = {
    IntentType.CODING: [
        "Write a Python function to sort a list",
        "Fix this JavaScript bug",
        "How do I implement a binary search tree?",
        "Generate unit tests for this code",
    ],
    IntentType.MATH: [
        "Solve this quadratic equation",
        "What is the integral of x squared?",
        "Calculate the probability of drawing an ace",
        "Find the eigenvalues of this matrix",
    ],
    IntentType.TRANSLATION: [
        "Translate this paragraph to French",
        "How do you say hello in Japanese?",
        "Convert this English text to Spanish",
    ],
    IntentType.SUMMARIZATION: [
        "Summarize this article in three bullet points",
        "Give me the key takeaways from this document",
        "Create a brief overview of this content",
    ],
    IntentType.RETRIEVAL: [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What are the symptoms of diabetes?",
    ],
    IntentType.REASONING: [
        "What are the pros and cons of remote work?",
        "Analyze the causes of the financial crisis",
        "Compare object-oriented and functional programming",
    ],
    IntentType.CONVERSATION: [
        "Hello, how are you?",
        "Tell me a joke",
        "What can you help me with?",
        "Thanks for your help",
    ],
    IntentType.VOICE: [
        "Transcribe this audio file",
        "Convert my speech to text",
        "Generate audio from this text",
    ],
}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


class HybridIntentClassifier(BaseIntentClassifier):
    """
    Two-stage intent classifier:
      1. Regex/keyword fast-path (no model, ~0ms)
      2. BGE semantic similarity fallback (~50ms on CPU)
    """

    KEYWORD_CONFIDENCE = 0.88
    SEMANTIC_THRESHOLD = 0.55

    def __init__(self, embedder=None) -> None:
        self._embedder = embedder  # BGEEmbedder, injected
        self._exemplar_embeddings: Optional[Dict[IntentType, np.ndarray]] = None
        self._cache: AsyncLRUCache = AsyncLRUCache(maxsize=512)
        # Pre-compile patterns
        self._patterns: Dict[IntentType, List[re.Pattern]] = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in _KEYWORD_RULES.items()
        }

    async def _ensure_exemplars(self) -> None:
        """Lazy-load exemplar embeddings (only once)."""
        if self._exemplar_embeddings is not None or self._embedder is None:
            return

        all_texts: List[str] = []
        intent_order: List[IntentType] = []
        counts: List[int] = []

        for intent, examples in _INTENT_EXEMPLARS.items():
            all_texts.extend(examples)
            intent_order.extend([intent] * len(examples))
            counts.append(len(examples))

        embeddings = await self._embedder.embed(all_texts)
        arr = np.array(embeddings, dtype=np.float32)

        self._exemplar_embeddings = {}
        idx = 0
        for intent, count in zip(_INTENT_EXEMPLARS.keys(), counts):
            group = arr[idx : idx + count]
            # Mean-pool exemplar embeddings
            self._exemplar_embeddings[intent] = group.mean(axis=0)
            idx += count

        log.info("Intent exemplar embeddings loaded", intents=len(self._exemplar_embeddings))

    def _keyword_classify(self, text: str) -> Optional[Tuple[IntentType, float]]:
        scores: Dict[IntentType, int] = {}
        for intent, patterns in self._patterns.items():
            hits = sum(1 for p in patterns if p.search(text))
            if hits > 0:
                scores[intent] = hits

        if not scores:
            return None

        best = max(scores, key=scores.__getitem__)
        # Normalize: more hits = higher confidence
        raw = scores[best]
        confidence = min(self.KEYWORD_CONFIDENCE, 0.60 + raw * 0.06)
        return best, confidence

    async def _semantic_classify(self, text: str) -> Optional[Tuple[IntentType, float]]:
        if self._embedder is None:
            return None
        await self._ensure_exemplars()

        query_emb = np.array(
            await self._embedder.embed_query(text), dtype=np.float32
        )

        best_intent: Optional[IntentType] = None
        best_score = -1.0
        for intent, exemplar_emb in self._exemplar_embeddings.items():
            sim = _cosine_sim(query_emb, exemplar_emb)
            if sim > best_score:
                best_score = sim
                best_intent = intent

        if best_score < self.SEMANTIC_THRESHOLD:
            return IntentType.CONVERSATION, 0.5

        return best_intent, best_score

    async def classify(self, text: str, context: Optional[str] = None) -> Intent:
        # Try cache first
        cache_key = text[:200]
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Stage 1: keyword fast-path
        keyword_result = self._keyword_classify(text)
        if keyword_result and keyword_result[1] >= self.KEYWORD_CONFIDENCE:
            intent_type, confidence = keyword_result
        else:
            # Stage 2: semantic fallback (degrades gracefully if embedder unavailable)
            try:
                semantic_result = await self._semantic_classify(text)
            except Exception:
                semantic_result = None
            if semantic_result:
                intent_type, confidence = semantic_result
            elif keyword_result:
                intent_type, confidence = keyword_result
            else:
                intent_type, confidence = IntentType.CONVERSATION, 0.5

        intent = Intent(
            intent_type=intent_type,
            confidence=confidence,
            requires_expert=intent_type in (
                IntentType.CODING, IntentType.MATH, IntentType.TRANSLATION
            ),
            requires_rag=intent_type in (
                IntentType.RETRIEVAL, IntentType.REASONING, IntentType.SUMMARIZATION
            ),
        )

        await self._cache.set(cache_key, intent)
        log.debug(
            "Intent classified",
            intent=intent_type.value,
            confidence=f"{confidence:.2f}",
            stage="keyword" if keyword_result and keyword_result[1] >= self.KEYWORD_CONFIDENCE else "semantic",
        )
        return intent
