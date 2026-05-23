"""Code expert – guides TinyLlama for software engineering tasks."""
from __future__ import annotations

from typing import Dict, List

from orchestrator.core.types import ExpertGuidance, Intent
from orchestrator.utils.text_utils import detect_language_hint
from .base_expert import ExpertBase


class CodeExpert(ExpertBase):
    expert_type = "code"

    _system_prompt = """You are an expert software engineer with deep knowledge across
multiple languages and paradigms. When writing code:
- Always produce syntactically correct, runnable code
- Follow language-specific best practices and idiomatic style
- Add minimal inline comments only where logic is non-obvious
- Handle edge cases and errors appropriately
- Prefer clarity over cleverness
- For debugging: identify the root cause before proposing a fix"""

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Write a Python function to find duplicate elements in a list",
            "assistant": """```python
def find_duplicates(lst: list) -> list:
    seen = set()
    return [x for x in lst if x in seen or seen.add(x)]  # type: ignore
```
This runs in O(n) time using a set for O(1) lookups.""",
        },
        {
            "user": "Fix this Python code: for i in range(10) print(i)",
            "assistant": "The loop body needs a colon: `for i in range(10): print(i)`",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        lang = detect_language_hint(query)
        if lang:
            lang_note = f"\nThe user appears to be working in **{lang}**. Match their language."
        else:
            lang_note = ""
        return self._system_prompt + lang_note
