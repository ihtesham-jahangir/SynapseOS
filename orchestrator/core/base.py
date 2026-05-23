"""
Abstract base classes that define the contracts for every major subsystem.
All concrete implementations inherit from these interfaces, enabling
dependency injection and clean swapping at runtime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from .types import (
    ChatRequest,
    DocumentChunk,
    ExpertGuidance,
    FusedContext,
    GenerationParams,
    InferenceResponse,
    Intent,
    MemoryItem,
    MemoryResult,
    RAGResult,
)


class BaseEmbedder(ABC):
    """Contracts for any embedding backend (BGE, OpenAI, etc.)."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Return L2-normalized embedding vectors."""

    async def embed_single(self, text: str) -> List[float]:
        results = await self.embed([text])
        return results[0]


class BaseMemoryStore(ABC):
    """Contract for a single cache level."""

    @abstractmethod
    async def store(self, item: MemoryItem) -> None: ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        session_id: str,
        limit: int = 5,
    ) -> List[MemoryItem]: ...

    @abstractmethod
    async def clear_session(self, session_id: str) -> None: ...


class BaseIntentClassifier(ABC):
    """Contract for intent classification."""

    @abstractmethod
    async def classify(self, text: str, context: Optional[str] = None) -> Intent: ...


class BaseExpert(ABC):
    """Contract for a specialised expert module."""

    expert_type: str = "base"

    @abstractmethod
    async def get_guidance(
        self,
        query: str,
        intent: Intent,
    ) -> ExpertGuidance: ...

    @abstractmethod
    def is_available(self) -> bool: ...


class BaseRAGPipeline(ABC):
    """Contract for the RAG retrieval pipeline."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 6) -> RAGResult: ...

    @abstractmethod
    async def ingest(self, content: str, source: str, metadata: Dict[str, Any]) -> int:
        """Return number of chunks indexed."""


class BaseFusionEngine(ABC):
    """Contract for the hierarchical fusion layer."""

    @abstractmethod
    async def fuse(
        self,
        query: str,
        intent: Intent,
        memory_result: MemoryResult,
        rag_result: RAGResult,
        expert_guidance: Optional[ExpertGuidance],
        conversation: List[Any],
    ) -> FusedContext: ...


class BaseInferenceEngine(ABC):
    """Contract for LLM inference backends."""

    @abstractmethod
    async def generate(
        self,
        context: FusedContext,
        params: GenerationParams,
    ) -> InferenceResponse: ...

    @abstractmethod
    async def stream(
        self,
        context: FusedContext,
        params: GenerationParams,
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
