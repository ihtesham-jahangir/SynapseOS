"""
TaskPlanner — decomposes a complex query into a SubTask DAG using the LLM.

The planner calls the LLM with a structured prompt and parses JSON output
into a list of SubTask objects with dependency edges.  On parse failure it
falls back to a single GENERAL sub-task so the pipeline always produces
something.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from orchestrator.agents.task_types import AgentRole, SubTask, TaskGraph
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

# Multi-word or unambiguous phrases that indicate genuine multi-step queries.
# Single words like "first", "also", "additionally" are intentionally excluded
# because they appear in simple sentences ("first explain X") and cause false
# positives that route trivial queries through the expensive multi-agent pipeline.
_COMPLEXITY_KEYWORDS = frozenset({
    "and then", "step by step", "explain and", "write and", "create and",
    "compare", "versus", "analyze", "breakdown", "walk me through",
    "followed by", "after that", "in addition to", "as well as",
})

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "task_planner.txt"
_PLAN_SYSTEM: str = _PROMPT_FILE.read_text(encoding="utf-8").strip()


def _is_complex(query: str) -> bool:
    """Heuristic: long queries or multi-step keywords → complex."""
    if len(query) > 200:
        return True
    q = query.lower()
    return any(kw in q for kw in _COMPLEXITY_KEYWORDS)


class TaskPlanner:
    """Converts a user query into a TaskGraph of SubTasks."""

    def __init__(self, llama_client) -> None:
        self._llama = llama_client

    def is_complex(self, query: str) -> bool:
        return _is_complex(query)

    async def plan(
        self,
        query: str,
        session_id: str,
        max_subtasks: int = 5,
    ) -> TaskGraph:
        """Return a TaskGraph for the given query."""
        graph = TaskGraph(session_id=session_id, original_query=query)
        try:
            raw = await self._llama.chat(
                messages=[
                    {"role": "system", "content": _PLAN_SYSTEM},
                    {"role": "user", "content": f"Decompose this query: {query}"},
                ],
                max_tokens=512,
                temperature=0.1,
                cache_prompt=True,  # reuse system-prompt KV-cache across planning calls
            )
            sub_tasks = self._parse(raw, max_subtasks)
        except Exception as exc:
            log.warning(
                "Task planning failed — using single-task fallback",
                error=str(exc),
            )
            sub_tasks = [
                SubTask(id="t1", description=query, role=AgentRole.GENERAL, dependencies=[])
            ]

        for st in sub_tasks:
            graph.sub_tasks[st.id] = st

        log.info(
            "Task plan created",
            task_id=graph.id,
            session=session_id,
            n_subtasks=len(graph.sub_tasks),
        )
        return graph

    def _parse(self, raw: str, max_subtasks: int) -> List[SubTask]:
        """Parse LLM JSON output into SubTask list. Raises on bad output."""
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in planner response")

        data = json.loads(match.group())
        items = data.get("sub_tasks", [])[:max_subtasks]

        if not items:
            raise ValueError("Planner returned empty sub_tasks list")

        sub_tasks: List[SubTask] = []
        for item in items:
            try:
                role = AgentRole(item.get("role", "general"))
            except ValueError:
                role = AgentRole.GENERAL

            sub_tasks.append(SubTask(
                id=str(item.get("id", f"t{len(sub_tasks) + 1}")),
                description=str(item.get("description", "")),
                role=role,
                dependencies=[str(d) for d in item.get("dependencies", [])],
            ))

        # Validate dependency IDs — drop non-existent or self-referential deps
        valid_ids = {st.id for st in sub_tasks}
        for st in sub_tasks:
            # A task must not depend on itself or a non-existent task
            invalid = [d for d in st.dependencies if d not in valid_ids or d == st.id]
            if invalid:
                log.warning(
                    "Planner returned invalid dependency IDs — dropping",
                    task=st.id,
                    invalid=invalid,
                )
                st.dependencies = [d for d in st.dependencies if d in valid_ids and d != st.id]

        return sub_tasks
