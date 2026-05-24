"""
Type definitions for the v3.0 multi-agent system.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class AgentRole(str, Enum):
    RESEARCH = "research"       # information gathering
    CODER = "coder"             # code generation
    REASONER = "reasoner"       # logical analysis
    SUMMARIZER = "summarizer"   # synthesis / final answer
    GENERAL = "general"         # fallback


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    role: AgentRole = AgentRole.GENERAL
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskGraph(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    original_query: str
    sub_tasks: Dict[str, SubTask] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    final_answer: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BusMessage(BaseModel):
    topic: str
    sender_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTaskRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    max_subtasks: int = 5
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: TaskStatus
    sub_tasks: List[SubTask]
    final_answer: Optional[str] = None
    total_elapsed_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
