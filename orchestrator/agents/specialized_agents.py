"""
Concrete agent implementations for v3.0.

Each agent uses a role-specific system prompt and generation temperature.
All call `self._llama.chat()` directly with plain dicts (no Pydantic overhead).
"""
from __future__ import annotations

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.agents.task_types import AgentRole, SubTask


class ResearchAgent(BaseAgent):
    """Gathers and synthesizes information from context."""

    role = AgentRole.RESEARCH

    async def _execute(self, sub_task: SubTask, context: str) -> str:
        prompt = (
            "You are a thorough research assistant. Based on the context below, "
            "answer the research question clearly and completely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {sub_task.description}"
        )
        return await self._llama.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )


class CoderAgent(BaseAgent):
    """Generates clean, correct code for programming sub-tasks."""

    role = AgentRole.CODER

    async def _execute(self, sub_task: SubTask, context: str) -> str:
        prompt = (
            "You are an expert programmer. Write clean, correct, well-commented code.\n\n"
            f"Context:\n{context}\n\n"
            f"Task: {sub_task.description}"
        )
        return await self._llama.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.2,
        )


class ReasonerAgent(BaseAgent):
    """Performs step-by-step logical analysis."""

    role = AgentRole.REASONER

    async def _execute(self, sub_task: SubTask, context: str) -> str:
        prompt = (
            "You are an analytical reasoning expert. Think step-by-step and provide "
            "a clear, well-reasoned analysis.\n\n"
            f"Context:\n{context}\n\n"
            f"Task: {sub_task.description}"
        )
        return await self._llama.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
        )


class SummarizerAgent(BaseAgent):
    """Synthesizes results from multiple sub-tasks into a coherent final answer."""

    role = AgentRole.SUMMARIZER

    async def _execute(self, sub_task: SubTask, context: str) -> str:
        prompt = (
            "You are a synthesis expert. Combine the partial results below into "
            "a single coherent, well-structured response.\n\n"
            f"Partial results:\n{context}\n\n"
            f"Synthesis task: {sub_task.description}"
        )
        return await self._llama.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.5,
        )


class GeneralAgent(BaseAgent):
    """Fallback agent for unclassified sub-tasks."""

    role = AgentRole.GENERAL

    async def _execute(self, sub_task: SubTask, context: str) -> str:
        prompt = f"Context:\n{context}\n\nTask: {sub_task.description}"
        return await self._llama.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )


# Role → class mapping (used by AgentPool)
ROLE_TO_AGENT_CLASS = {
    AgentRole.RESEARCH: ResearchAgent,
    AgentRole.CODER: CoderAgent,
    AgentRole.REASONER: ReasonerAgent,
    AgentRole.SUMMARIZER: SummarizerAgent,
    AgentRole.GENERAL: GeneralAgent,
}
