"""Writing expert – professional writing, editing, and content creation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from orchestrator.core.types import ExpertGuidance, Intent, ExpertType
from .base_expert import ExpertBase

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "expert_writing.txt"
_SYSTEM_PROMPT: str = _PROMPT_FILE.read_text(encoding="utf-8").strip()


class WritingExpert(ExpertBase):
    expert_type = "writing"

    _system_prompt = _SYSTEM_PROMPT

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Write a professional email declining a meeting request",
            "assistant": """Subject: Re: Meeting Request – [Your Topic]

Hi [Name],

Thank you for reaching out. Unfortunately, I'm unable to join on [date] due to a prior commitment.

I'd be happy to connect at another time. Would [alternative date/time] work for you, or feel free to share a few options that fit your schedule?

Best regards,
[Your Name]""",
        },
        {
            "user": "Improve this sentence: 'There are a lot of reasons why this is a good idea that we should think about.'",
            "assistant": "This idea offers several compelling reasons worth considering.\n\n*(Cuts word count by 60%, removes 'there are' construction, eliminates hedge 'a lot of')*",
        },
        {
            "user": "Write an executive summary for a product launch report",
            "assistant": """**Executive Summary**

[Product Name] launched on [date], achieving [X]% of projected revenue in Q1 with [N] active users.

**Key results:** [Metric 1] | [Metric 2] | [Metric 3]

**What worked:** [Top 2 success factors]

**What needs attention:** [Top risk or gap]

**Recommended next steps:** [Action 1] by [date], [Action 2] by [date].""",
        },
        {
            "user": "Write a cover letter for a senior software engineer role",
            "assistant": """Dear Hiring Manager,

I am writing to apply for the Senior Software Engineer position at [Company]. With [X] years building [relevant systems], I have delivered [specific achievement] that [measurable impact].

At [Previous Company], I [key contribution 1] and [key contribution 2], directly contributing to [outcome]. I thrive in [environment type] teams and bring expertise in [top 2-3 relevant skills].

I would welcome the opportunity to discuss how my background aligns with your team's goals.

Sincerely,
[Name]""",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        prompt = self._system_prompt
        q = query.lower()
        if "email" in q or "message" in q:
            prompt += "\n\nFocus: Business communication. Keep it brief and action-oriented."
        elif "creative" in q or "story" in q or "poem" in q:
            prompt += "\n\nFocus: Creative writing. Prioritize originality and vivid language."
        elif "proofread" in q or "edit" in q or "grammar" in q:
            prompt += "\n\nFocus: Editing. Track changes with before/after for key improvements."
        elif "report" in q or "article" in q or "essay" in q:
            prompt += "\n\nFocus: Long-form content. Use clear section headers and logical flow."
        return prompt
