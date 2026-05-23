"""
Document chunking strategies for RAG ingestion.

Two chunkers with the same ``chunk(text)`` interface:
  RecursiveTextChunker — fast, sync, splits at paragraph/sentence/word boundaries
  SemanticChunker      — async, splits at embedding-space topic boundaries

SemanticChunker.chunk() is a coroutine; call it with ``await``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.text_utils import clean_text, sentence_split


@dataclass
class Chunk:
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int


class RecursiveTextChunker:
    """
    Hierarchical chunker that tries to split on paragraphs → sentences → words.
    Falls back to harder splits only when necessary.
    Produces overlapping chunks for context continuity.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[Chunk]:
        text = clean_text(text)
        raw_chunks = self._split(text, self.SEPARATORS)
        chunks: List[Chunk] = []
        pos = 0
        for i, c in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    content=c,
                    chunk_index=i,
                    start_char=pos,
                    end_char=pos + len(c),
                    token_count=count_tokens(c),
                )
            )
            pos += max(1, len(c) - self._overlap_chars(c))
        return chunks

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the separator hierarchy."""
        if not text.strip():
            return []

        separator = separators[0] if separators else ""

        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        good: List[str] = []
        pending: List[str] = []
        pending_tokens = 0

        for part in parts:
            part_tokens = count_tokens(part)

            if part_tokens > self.chunk_size and len(separators) > 1:
                # Recursively split oversized parts
                sub_chunks = self._split(part, separators[1:])
                for sc in sub_chunks:
                    sc_tokens = count_tokens(sc)
                    if pending_tokens + sc_tokens > self.chunk_size and pending:
                        good.append(separator.join(pending))
                        pending = self._overlap_slice(pending)
                        pending_tokens = count_tokens(separator.join(pending))
                    pending.append(sc)
                    pending_tokens += sc_tokens
            else:
                if pending_tokens + part_tokens > self.chunk_size and pending:
                    good.append(separator.join(pending))
                    pending = self._overlap_slice(pending)
                    pending_tokens = count_tokens(separator.join(pending))
                pending.append(part)
                pending_tokens += part_tokens

        if pending:
            good.append(separator.join(pending))

        return [g for g in good if g.strip()]

    def _overlap_slice(self, parts: List[str]) -> List[str]:
        """Keep the tail of parts that fits within chunk_overlap tokens."""
        result: List[str] = []
        total = 0
        for p in reversed(parts):
            t = count_tokens(p)
            if total + t > self.chunk_overlap:
                break
            result.insert(0, p)
            total += t
        return result

    def _overlap_chars(self, chunk: str) -> int:
        tokens = count_tokens(chunk)
        if tokens <= self.chunk_overlap:
            return 0
        ratio = self.chunk_overlap / max(tokens, 1)
        return int(len(chunk) * ratio)


class SemanticChunker:
    """
    Splits text at embedding-space topic boundaries.

    Embeds every sentence, then inserts a chunk boundary wherever the
    cosine similarity between adjacent sentences drops below
    ``breakpoint_threshold``.  Groups that exceed ``max_chunk_tokens``
    are further split by word count.

    ``chunk()`` is an async method — call it with ``await``.
    """

    def __init__(
        self,
        embedder: Any,            # BGEEmbedder — Any to avoid heavy import
        breakpoint_threshold: float = 0.75,
        max_chunk_tokens: int = 512,
    ) -> None:
        self._embedder = embedder
        self._threshold = breakpoint_threshold
        self._max_tokens = max_chunk_tokens

    async def chunk(self, text: str) -> List[Chunk]:
        text = clean_text(text)
        sentences = sentence_split(text)

        if not sentences:
            return []
        if len(sentences) <= 2:
            # Too few sentences for meaningful boundary detection
            return self._build_chunks(sentences, [])

        embeddings = await self._embedder.embed(sentences)

        breakpoints: List[int] = []
        for i in range(len(embeddings) - 1):
            sim = _cosine(embeddings[i], embeddings[i + 1])
            if sim < self._threshold:
                breakpoints.append(i + 1)

        chunks = self._build_chunks(sentences, breakpoints)
        # Reassign sequential indices
        for idx, c in enumerate(chunks):
            c.chunk_index = idx
        return chunks

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_chunks(
        self,
        sentences: List[str],
        breakpoints: List[int],
    ) -> List[Chunk]:
        groups: List[List[str]] = []
        prev = 0
        for bp in breakpoints:
            groups.append(sentences[prev:bp])
            prev = bp
        groups.append(sentences[prev:])

        result: List[Chunk] = []
        char_pos = 0
        group_idx = 0
        for group in groups:
            content = " ".join(group)
            sub = self._token_split(content, group_idx, char_pos)
            result.extend(sub)
            group_idx += len(sub)
            char_pos += len(content) + 1
        return result

    def _token_split(
        self,
        text: str,
        start_idx: int,
        char_pos: int,
    ) -> List[Chunk]:
        tok = count_tokens(text)
        if tok <= self._max_tokens:
            return [
                Chunk(
                    content=text,
                    chunk_index=start_idx,
                    start_char=char_pos,
                    end_char=char_pos + len(text),
                    token_count=tok,
                )
            ]
        # Oversized: split at word midpoint recursively
        words = text.split()
        mid = len(words) // 2
        left = " ".join(words[:mid])
        right = " ".join(words[mid:])
        left_chunks = self._token_split(left, start_idx, char_pos)
        right_chunks = self._token_split(
            right,
            start_idx + len(left_chunks),
            char_pos + len(left) + 1,
        )
        return left_chunks + right_chunks


def _cosine(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-10
    return float(np.dot(va, vb) / denom)
