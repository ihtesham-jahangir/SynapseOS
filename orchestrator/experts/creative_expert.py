"""Creative expert – brainstorming, storytelling, ideation, and world-building."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from orchestrator.core.types import Intent
from .base_expert import ExpertBase

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "expert_creative.txt"
_SYSTEM_PROMPT: str = _PROMPT_FILE.read_text(encoding="utf-8").strip()


class CreativeExpert(ExpertBase):
    expert_type = "creative"

    _system_prompt = _SYSTEM_PROMPT

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Brainstorm 5 unusual SaaS product ideas",
            "assistant": """1. **GriefOS** — AI-powered memorial service that lets families create interactive digital legacies of loved ones, answering questions in their voice using recorded audio/writing.

2. **ConflictCoach** — Real-time workplace mediation tool that joins video calls and surfaces communication patterns contributing to conflict, offering in-moment rephrasing suggestions.

3. **SkillDecay** — Spaced-repetition platform for professionals — sends micro-quizzes to prevent skill atrophy in domains you're not actively using (e.g., a backend engineer who rarely does SQL).

4. **EthicsAudit** — AI that scans company policies, product features, and marketing copy to flag potential ethical issues before public release, mapped to industry-specific frameworks.

5. **PairMatch** — Assigns two strangers from different industries a shared 30-day project; people solve cross-domain problems and build unexpected professional relationships.""",
        },
        {
            "user": "Write the opening paragraph of a sci-fi short story about a lighthouse keeper on a gas giant",
            "assistant": """The pressure gauges never lied, but Keita had learned to distrust what they measured. At three thousand kilometers above Jupiter's hydrogen mantle, the lighthouse wasn't built for ships — it was built for minds: the vast, slow intelligences that drifted through the ammonia bands like continents in slow motion, navigating by radio frequencies humans couldn't hear and time scales humans couldn't imagine. Keita's job was simple. Keep the beacon on. Don't ask what it was saying.""",
        },
        {
            "user": "Generate creative names for a productivity app that helps people say no",
            "assistant": """**Declination** — sophisticated, sounds like software, implies deliberate choice
**Boundry** — intentional misspelling makes it memorable and web-friendly
**The Art of No** — clear positioning as a philosophy, not just a tool
**Moat** — a moat protects what matters; strong visual metaphor
**Refusal** — provocative, stands out in a sea of "Yes" tools
**Less** — minimalist, works as both brand and product philosophy
**Guardrail** — implies protection without aggression

*Recommend: Moat — memorable, abstract enough for brand extension, strong visual identity potential.*""",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        prompt = self._system_prompt
        q = query.lower()
        if "brainstorm" in q or "idea" in q:
            prompt += "\n\nFocus: Ideation. Produce a numbered list of distinct, developed ideas — no one-liners."
        elif "story" in q or "fiction" in q or "narrative" in q:
            prompt += "\n\nFocus: Storytelling. Use concrete scenes, specific details, and a distinct voice."
        elif "name" in q or "brand" in q:
            prompt += "\n\nFocus: Naming. Generate 5-8 options across different naming strategies. Explain the concept behind each."
        elif "poem" in q or "lyric" in q:
            prompt += "\n\nFocus: Poetry. Prioritize imagery, rhythm, and emotional resonance over rhyme."
        return prompt
