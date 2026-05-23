"""Custom exception hierarchy for SynapseOS."""
from __future__ import annotations


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""

    def __init__(self, message: str, code: str = "ORCHESTRATOR_ERROR") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LlamaServerError(OrchestratorError):
    """llama.cpp server is unavailable or returned an error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "LLAMA_SERVER_ERROR")


class LlamaTimeoutError(LlamaServerError):
    """Generation timed out."""

    def __init__(self) -> None:
        super().__init__("LLM inference timed out")
        self.code = "LLAMA_TIMEOUT"


class EmbeddingError(OrchestratorError):
    """Embedding model failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "EMBEDDING_ERROR")


class MemoryError(OrchestratorError):
    """Memory system failure."""

    def __init__(self, message: str, level: str = "unknown") -> None:
        super().__init__(message, f"MEMORY_ERROR_{level.upper()}")
        self.level = level


class RAGError(OrchestratorError):
    """RAG pipeline failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "RAG_ERROR")


class FusionError(OrchestratorError):
    """Context fusion failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "FUSION_ERROR")


class ExpertError(OrchestratorError):
    """Expert module failure."""

    def __init__(self, message: str, expert: str = "unknown") -> None:
        super().__init__(message, f"EXPERT_ERROR_{expert.upper()}")
        self.expert = expert


class TokenBudgetExceededError(OrchestratorError):
    """Context exceeds available token budget."""

    def __init__(self, budget: int, actual: int) -> None:
        super().__init__(
            f"Token budget exceeded: budget={budget}, actual={actual}",
            "TOKEN_BUDGET_EXCEEDED",
        )
        self.budget = budget
        self.actual = actual


class SessionNotFoundError(OrchestratorError):
    """Requested session does not exist."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' not found", "SESSION_NOT_FOUND")
        self.session_id = session_id


class DocumentNotFoundError(OrchestratorError):
    """Requested document does not exist in the index."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(f"Document '{doc_id}' not found", "DOCUMENT_NOT_FOUND")
        self.doc_id = doc_id


class ConfigurationError(OrchestratorError):
    """Invalid or missing configuration."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "CONFIGURATION_ERROR")
