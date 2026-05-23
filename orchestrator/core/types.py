"""
Core type definitions for SynapseOS.
All data contracts flow through these Pydantic models.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


# ─── Enumerations ─────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    CODING = "coding"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    MATH = "math"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    CONVERSATION = "conversation"
    VOICE = "voice"
    UNKNOWN = "unknown"


class MemoryLevel(str, Enum):
    L1 = "l1_hot"
    L2 = "l2_summary"
    L3 = "l3_vector"
    L4 = "l4_knowledge"


class ExpertType(str, Enum):
    CODE = "code"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    MATH = "math"
    REASONING = "reasoning"
    GENERAL = "general"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class StreamEventType(str, Enum):
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"
    METADATA = "metadata"


# ─── Request / Response models ────────────────────────────────────────────────

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def last_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return ""


class ChatResponse(BaseModel):
    session_id: str
    content: str
    intent: IntentType
    tokens_generated: int
    total_tokens: int
    time_to_first_token_ms: float
    total_time_ms: float
    memory_levels_used: List[MemoryLevel]
    rag_chunks_used: int
    expert_used: Optional[ExpertType]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    event_type: StreamEventType
    session_id: str
    content: str = ""
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Intent models ────────────────────────────────────────────────────────────

class Intent(BaseModel):
    intent_type: IntentType
    confidence: float
    subtype: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    requires_expert: bool = False
    requires_rag: bool = False


# ─── Memory models ────────────────────────────────────────────────────────────

class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    level: MemoryLevel
    session_id: Optional[str] = None
    relevance_score: float = 0.0
    importance_score: float = 0.5
    recency_score: float = 1.0
    composite_score: float = 0.0
    token_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryResult(BaseModel):
    items: List[MemoryItem]
    total_tokens: int
    levels_queried: List[MemoryLevel]
    query_time_ms: float


# ─── RAG models ───────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    chunks: List[DocumentChunk]
    scores: List[float]
    total_tokens: int
    query_time_ms: float


class IngestRequest(BaseModel):
    content: str
    source: str = "user_upload"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Expert models ────────────────────────────────────────────────────────────

class ExpertGuidance(BaseModel):
    expert_type: ExpertType
    system_prompt: str
    few_shot_examples: List[Dict[str, str]] = Field(default_factory=list)
    confidence: float = 1.0
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Fusion models ────────────────────────────────────────────────────────────

class ContextItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str
    composite_score: float
    token_count: int
    level: Optional[MemoryLevel] = None
    is_mandatory: bool = False


class FusedContext(BaseModel):
    system_prompt: str
    context_items: List[ContextItem]
    conversation_turns: List[Message]
    expert_guidance: Optional[ExpertGuidance] = None
    total_token_count: int
    token_budget_used: float  # fraction 0-1
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        """Flatten fused context into llama.cpp message format."""
        msgs: List[Dict[str, str]] = []

        # System prompt
        system_parts = [self.system_prompt]
        if self.context_items:
            context_text = "\n---\n".join(
                item.content.strip() for item in self.context_items
            )
            system_parts.append(f"\n\nBackground information:\n{context_text}")
        msgs.append({"role": "system", "content": "\n".join(system_parts)})

        # Conversation history
        for turn in self.conversation_turns:
            msgs.append({"role": turn.role.value, "content": turn.content})

        return msgs


# ─── Generation / Runtime models ──────────────────────────────────────────────

class GenerationParams(BaseModel):
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = Field(default_factory=lambda: [
        "</s>", "[/INST]", "<|user|>", "<|system|>",
        "\n### ", "\n[memory:", "\nuser:", "\nassistant:",
    ])


class InferenceRequest(BaseModel):
    fused_context: FusedContext
    params: GenerationParams
    stream: bool = False
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class InferenceResponse(BaseModel):
    content: str
    tokens_generated: int
    prompt_tokens: int
    time_to_first_token_ms: float
    total_time_ms: float
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Health models ────────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str = "0.1.0"
    components: List[ComponentHealth]
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Admin / Document models ──────────────────────────────────────────────────

class DocumentIngestionResponse(BaseModel):
    document_id: str
    chunks_created: int
    tokens_indexed: int
    time_ms: float


class SessionInfo(BaseModel):
    session_id: str
    turn_count: int
    total_tokens: int
    created_at: datetime
    last_active: datetime
