"""BaseAgent — abstract worker with llama.cpp client and memory bus access."""
from __future__ import annotations

import abc
import time

from orchestrator.agents.task_types import AgentRole, SubTask, TaskStatus
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class BaseAgent(abc.ABC):
    """
    Abstract base for all specialized agents.

    Subclasses implement `_execute(sub_task, context) -> str`.
    `run()` wraps execution with timing, status management, error capture,
    and result publishing to the shared memory bus.
    """

    role: AgentRole = AgentRole.GENERAL

    def __init__(self, llama_client, bus) -> None:
        self._llama = llama_client
        self._bus = bus

    async def run(self, task_id: str, sub_task: SubTask, context: str) -> SubTask:
        """Execute a sub-task: update status, time the call, capture errors."""
        sub_task.status = TaskStatus.RUNNING
        t0 = time.perf_counter()
        try:
            result = await self._execute(sub_task, context)
            sub_task.result = result
            sub_task.status = TaskStatus.DONE
            await self._bus.publish(
                topic=f"result.{sub_task.id}",
                sender_id=self.role.value,
                content=result,
            )
            log.debug(
                "Agent sub-task done",
                role=self.role.value,
                subtask=sub_task.id,
                task=task_id,
            )
        except Exception as exc:
            sub_task.status = TaskStatus.FAILED
            sub_task.error = str(exc)
            log.error(
                "Agent sub-task failed",
                role=self.role.value,
                subtask=sub_task.id,
                error=str(exc),
            )
        finally:
            sub_task.elapsed_ms = (time.perf_counter() - t0) * 1000
        return sub_task

    @abc.abstractmethod
    async def _execute(self, sub_task: SubTask, context: str) -> str:
        """Core agent logic — return the result string."""
