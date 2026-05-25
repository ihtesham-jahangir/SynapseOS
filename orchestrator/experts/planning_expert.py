"""Planning expert – project planning, task breakdown, roadmaps, and strategy."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from orchestrator.core.types import Intent
from .base_expert import ExpertBase

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "expert_planning.txt"
_SYSTEM_PROMPT: str = _PROMPT_FILE.read_text(encoding="utf-8").strip()


class PlanningExpert(ExpertBase):
    expert_type = "planning"

    _system_prompt = _SYSTEM_PROMPT

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Create a 3-month project plan to build and launch a mobile app",
            "assistant": """**Month 1 — Foundation**
- [ ] Week 1: Define core user stories (P0 features only) + technical architecture decision
- [ ] Week 1-2: Set up repo, CI/CD pipeline, dev/staging environments
- [ ] Week 2-4: Build auth flow + core data models + API skeleton
- [ ] **Milestone:** Working auth + API skeleton. Gate: team alignment on scope.

**Month 2 — Core Build**
- [ ] Week 5-6: Build primary feature set (P0 stories)
- [ ] Week 6-7: UI polish + error handling + offline support basics
- [ ] Week 7-8: Internal alpha testing + bug fix sprint
- [ ] **Milestone:** Feature-complete internal build. Gate: passes smoke tests.

**Month 3 — Launch**
- [ ] Week 9-10: Beta with 20-50 real users; collect and triage feedback
- [ ] Week 10-11: Critical bug fixes + App Store submission preparation
- [ ] Week 12: Soft launch → monitor crash rates, retention metrics
- [ ] **Milestone:** Public launch. Gate: <1% crash rate, core KPIs defined.

**Key risks:** Scope creep (mitigate: P0/P1 strict separation); App Store review delays (buffer 2 weeks).""",
        },
        {
            "user": "Help me prioritize these 10 backlog items",
            "assistant": """Use the **Impact × Effort matrix**:

| Priority | Item | Impact (1-5) | Effort (1-5) | Ratio | Action |
|---|---|---|---|---|---|
| P0 | [Your item] | 5 | 1 | 5.0 | Do first |
| P1 | [Your item] | 4 | 2 | 2.0 | Next sprint |
| P2 | [Your item] | 3 | 3 | 1.0 | Backlog |
| ❌ | [Your item] | 2 | 4 | 0.5 | Cut or defer |

**Share your actual items** and I'll score them. Also consider:
- **Dependency blockers** — unblock other high-value work first
- **User pain** — which causes the most complaints/churn?
- **Strategic alignment** — which advances your Q1 goal most?""",
        },
        {
            "user": "What steps do I need to take to start a SaaS business?",
            "assistant": """**Phase 1: Validate (0-4 weeks)**
1. Define the problem you're solving and the specific customer segment
2. Interview 10+ potential customers — confirm they experience the pain
3. Build a landing page with a waitlist to measure demand
4. Identify 1-3 direct competitors; find the gap they leave

**Phase 2: Build MVP (1-3 months)**
5. Define the single core workflow (resist scope creep)
6. Choose tech stack (default to boring/reliable: PostgreSQL, established framework)
7. Build the MVP; aim for something a pilot customer can use in week 8
8. Ship to 3-5 pilot users — watch them use it, don't assume

**Phase 3: Revenue & Growth (month 3+)**
9. Set pricing (anchor high, discount early; $X/month > free trials)
10. Establish a feedback loop: weekly call with top 3 users
11. Define your acquisition channel (SEO, outbound, partnerships — pick one)
12. Measure: MRR, churn rate, CAC, LTV — weekly dashboards from day 1

**Critical first milestone:** First dollar of revenue. Everything before that is hypothesis.""",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        prompt = self._system_prompt
        q = query.lower()
        if "sprint" in q or "agile" in q or "scrum" in q or "backlog" in q:
            prompt += "\n\nFocus: Agile planning. Use sprint-structured output with clear acceptance criteria."
        elif "roadmap" in q or "quarter" in q or "annual" in q:
            prompt += "\n\nFocus: Roadmap. Use Now/Next/Later framework. Distinguish themes from features."
        elif "steps" in q or "how to start" in q or "what do i need" in q:
            prompt += "\n\nFocus: Actionable step-by-step guide. Number every step. Include time estimates."
        elif "priorit" in q:
            prompt += "\n\nFocus: Prioritization. Apply Impact×Effort or RICE scoring. Give a ranked output."
        return prompt
