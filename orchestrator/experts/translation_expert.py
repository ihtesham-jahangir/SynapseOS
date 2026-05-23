"""Translation expert – guides TinyLlama for multilingual tasks."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from orchestrator.core.types import Intent
from .base_expert import ExpertBase


_LANG_PATTERNS = {
    "French": r"\b(?:french|français|fr)\b",
    "Spanish": r"\b(?:spanish|español|es)\b",
    "German": r"\b(?:german|deutsch|de)\b",
    "Italian": r"\b(?:italian|italiano|it)\b",
    "Japanese": r"\b(?:japanese|日本語|ja)\b",
    "Chinese": r"\b(?:chinese|mandarin|中文|zh)\b",
    "Arabic": r"\b(?:arabic|عربي|ar)\b",
    "Portuguese": r"\b(?:portuguese|português|pt)\b",
    "Russian": r"\b(?:russian|русский|ru)\b",
}


class TranslationExpert(ExpertBase):
    expert_type = "translation"

    _system_prompt = """You are an expert multilingual translator with native-level
fluency in dozens of languages. When translating:
- Preserve the original meaning, tone, and nuance
- Use natural-sounding phrasing in the target language
- Respect cultural context and idiomatic expressions
- Note any terms that don't have direct equivalents
- If the target language is unspecified, ask for clarification"""

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Translate 'The early bird catches the worm' to French",
            "assistant": "\"L'avenir appartient à ceux qui se lèvent tôt\" (lit: 'The future belongs to those who rise early') — this is the French idiomatic equivalent.",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        target_lang = self._detect_target_language(query)
        if target_lang:
            return (
                self._system_prompt
                + f"\n\nTarget language detected: **{target_lang}**. "
                f"Ensure output is idiomatic {target_lang}."
            )
        return self._system_prompt

    @staticmethod
    def _detect_target_language(query: str) -> Optional[str]:
        for lang, pattern in _LANG_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                return lang
        return None
