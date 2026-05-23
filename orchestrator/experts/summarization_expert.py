"""Summarization expert."""
from __future__ import annotations

from typing import Dict, List

from .base_expert import ExpertBase


class SummarizationExpert(ExpertBase):
    expert_type = "summarization"

    _system_prompt = """You are a world-class editor and summarizer.
When summarizing content:
- Identify and preserve all key facts, decisions, and insights
- Use the most compact representation that loses no critical information
- Structure output clearly: use bullet points for lists of facts, prose for narrative
- Preserve original numbers, names, dates, and technical terms exactly
- Target length: 20-30% of the original unless instructed otherwise
- Avoid padding, filler phrases, and redundant statements"""

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Summarize: Machine learning is a subset of AI that gives systems the ability to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "assistant": "**ML** is an AI subset enabling systems to learn from data without explicit programming, improving performance through experience rather than fixed rules.",
        },
    ]
