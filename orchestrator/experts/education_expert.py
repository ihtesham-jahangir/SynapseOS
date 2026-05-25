"""Education expert – step-by-step tutoring, concept explanation, and learning guidance."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from orchestrator.core.types import Intent
from .base_expert import ExpertBase

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "expert_education.txt"
_SYSTEM_PROMPT: str = _PROMPT_FILE.read_text(encoding="utf-8").strip()


class EducationExpert(ExpertBase):
    expert_type = "education"

    _system_prompt = _SYSTEM_PROMPT

    _few_shot_examples: List[Dict[str, str]] = [
        {
            "user": "Explain recursion like I'm 10",
            "assistant": """**Recursion is when something calls itself to solve a smaller version of the same problem.**

Imagine you're searching for a lost sock in a pile of clothes:
1. Pick up the top item
2. Is it the sock? → Done!
3. Not the sock? → Put it aside and search the *smaller* remaining pile (same problem, smaller pile)

That's recursion: solving a big problem by solving a smaller version of the exact same problem.

**In code:**
```python
def find_factorial(n):
    if n == 1:           # Base case: stop here
        return 1
    return n * find_factorial(n - 1)  # Call itself with smaller n
```
`factorial(5)` = `5 × factorial(4)` = `5 × 4 × factorial(3)` ... until we hit 1.

**The one thing to remember:** Every recursive function needs a *base case* — the point where it stops calling itself. Without it, it's like mirrors facing each other: infinite reflections, infinite calls, then crash.""",
        },
        {
            "user": "I don't understand how async/await works in Python",
            "assistant": """**The core idea:** `async`/`await` lets your program *wait for slow things* (network, disk) without freezing.

**Analogy:** Imagine a chef cooking 3 dishes. Without async: start dish 1, wait for water to boil (frozen, doing nothing), finish dish 1, start dish 2... With async: start dish 1, while water boils → start dish 2, while that simmers → start dish 3. Same chef, 3× throughput.

**The two keywords:**
- `async def` → marks a function as a coroutine (can be paused)
- `await` → pauses *this* function while something slow finishes, letting other functions run

```python
import asyncio

async def fetch_data(url):
    await asyncio.sleep(1)  # simulates network request
    return f"data from {url}"

async def main():
    # Run three fetches concurrently (not one-by-one)
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3"),
    )
    print(results)  # all 3 done in ~1 second, not 3 seconds

asyncio.run(main())
```

**Common confusion:** `await` doesn't mean "stop everything" — it means "pause *me* so others can run."

**Try this:** Run the code above with and without `gather` to see the timing difference.""",
        },
        {
            "user": "What's the difference between supervised and unsupervised learning?",
            "assistant": """**One-line answer:** Supervised = you give the model the answers; Unsupervised = you don't.

**Supervised learning:**
- Training data has labels (input → correct output)
- Model learns to predict the output for new inputs
- Examples: spam detection (email → spam/not spam), house price prediction (features → $)
- Analogy: a student learning with an answer key

**Unsupervised learning:**
- Training data has NO labels — just inputs
- Model finds hidden structure, patterns, or groupings on its own
- Examples: customer segmentation, topic modeling, anomaly detection
- Analogy: a scientist who discovers that rocks naturally group into 3 types without being told

**When to use which?**
| You have... | Use |
|---|---|
| Labeled data (input + correct output) | Supervised |
| Only inputs, want to discover patterns | Unsupervised |
| A few labels, lots of unlabeled data | Semi-supervised |
| An environment with rewards | Reinforcement learning |

**Most common mistake:** Treating unsupervised as "cheaper supervised" — it's a different tool for different questions.""",
        },
    ]

    def _build_system_prompt(self, query: str, intent: Intent) -> str:
        prompt = self._system_prompt
        q = query.lower()
        if "beginner" in q or "basic" in q or "from scratch" in q or "never" in q:
            prompt += "\n\nAudience: Complete beginner. Assume zero prior knowledge. Use everyday analogies."
        elif "advanced" in q or "deep dive" in q or "internals" in q:
            prompt += "\n\nAudience: Experienced practitioner. Skip basics; go deep on nuance and edge cases."
        elif "explain" in q and ("like" in q or "simple" in q):
            prompt += "\n\nUse the Feynman technique: simplest possible language, powerful analogy, then build up."
        elif "example" in q or "show me" in q:
            prompt += "\n\nLead with a runnable, concrete example. Explain the theory after showing it working."
        return prompt
