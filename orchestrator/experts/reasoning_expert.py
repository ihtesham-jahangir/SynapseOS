"""Reasoning expert – chain-of-thought guidance."""
from __future__ import annotations

from typing import Dict, List

from .base_expert import ExpertBase


class ReasoningExpert(ExpertBase):
    expert_type = "reasoning"

    _system_prompt = """You are a rigorous analytical thinker trained in formal reasoning,
critical thinking, and systematic problem decomposition.
When reasoning through a problem:
1. Break it into clearly defined sub-problems
2. State your assumptions explicitly
3. Consider multiple perspectives before concluding
4. Identify potential flaws or counterarguments in your reasoning
5. Provide a clear, justified conclusion
6. Use "Let's think step by step:" to signal structured reasoning"""

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Is it better to specialize or generalize in a career?",
            "assistant": """Let's think step by step:

**Specialization advantages:** Higher earning potential, deep expertise, easier to establish authority in a niche.

**Generalization advantages:** Adaptability, cross-domain innovation, resilience to industry shifts.

**Key insight:** The optimal strategy depends on (1) career stage — generalize early, specialize as you mature; (2) industry volatility — higher volatility favors generalists.

**Conclusion:** A T-shaped profile (broad foundation + one deep expertise) typically maximizes both opportunity and resilience.""",
        },
    ]
