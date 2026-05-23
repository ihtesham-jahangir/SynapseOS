"""
Core data types for SynapseOS.
All domain contracts flow through these Pydantic models.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    CODING = "coding"
    MATH = "math"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    CONVERSATION = "conversation"


class Intent(BaseModel):
    intent_type: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    requires_rag: bool = False
    requires_expert: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubTask(BaseModel):
    id: str
    description: str
    intent: Intent
    priority: int = 0
    dependencies: List[str] = Field(default_factory=list)


class ExecutionTarget(str, Enum):
    TINYLLAMA = "tinyllama"
    CODE_EXPERT = "code_expert"
    TRANSLATION_EXPERT = "translation_expert"
    MATH_EXPERT = "math_expert"
    SUMMARIZATION_EXPERT = "summarization_expert"
    TOOL = "tool"


class ExecutionResult(BaseModel):
    task_id: str
    target: ExecutionTarget
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    id: str
    content: str
    role: str  # "user" | "assistant" | "system"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    relevance_score: float = 0.0


class FusedOutput(BaseModel):
    session_id: str
    query: str
    intent: Intent
    context_items: List[MemoryItem] = Field(default_factory=list)
    execution_results: List[ExecutionResult] = Field(default_factory=list)
    fused_prompt: str = ""
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    session_id: str = "default"
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    session_id: str
    content: str
    intent: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
