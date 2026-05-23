"""General-purpose text manipulation helpers."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace sequences into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Remove control characters and normalize unicode."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return normalize_whitespace(text)


def sentence_split(text: str) -> List[str]:
    """Naive sentence splitter that avoids breaking abbreviations."""
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)].rstrip() + suffix


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract (language, code) pairs from markdown fenced blocks."""
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.strip(), code.strip()) for lang, code in matches]


def remove_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()


def has_code(text: str) -> bool:
    return bool(re.search(r"```|`[^`]+`|def |class |import |#include", text))


def detect_language_hint(text: str) -> Optional[str]:
    """Very lightweight language detection from keyword patterns."""
    checks = {
        "python": r"\bdef \w+\(|import \w+|print\(",
        "javascript": r"\bfunction \w+\(|const |let |var |=>\s*{",
        "java": r"\bpublic class |System\.out\.print|void main",
        "cpp": r"#include <|std::|cout <<",
        "rust": r"\bfn \w+\(|let mut |use std::",
        "sql": r"\bSELECT\b|\bFROM\b|\bWHERE\b",
        "bash": r"^\s*#!/bin/bash|^\s*\$\s+\w",
    }
    for lang, pattern in checks.items():
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            return lang
    return None


def chunk_text_by_sentences(
    text: str,
    max_chars: int = 2000,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks respecting sentence boundaries.
    Used for document ingestion chunking.
    """
    sentences = sentence_split(text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep last portion
            overlap: List[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s)
            current_chunk = overlap
            current_len = overlap_len
        current_chunk.append(sentence)
        current_len += s_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def format_context_block(source: str, content: str) -> str:
    """Format a piece of retrieved context for prompt injection."""
    return f"[Source: {source}]\n{content.strip()}"
