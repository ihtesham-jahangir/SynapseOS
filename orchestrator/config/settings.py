"""
Central configuration for SynapseOS.
All tunables live here; pulled from environment or .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _auto_model_path() -> str:
    """Return the best available .gguf path, or a sensible default string."""
    try:
        from orchestrator.config.model_selector import select_model
        path, _ = select_model()
        if path is not None:
            return str(path)
    except Exception:
        pass
    return "models/Llama-3.2-3B-Instruct-Q5_K_M.gguf"


class LlamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLAMA_", env_file=".env", extra="ignore")

    server_url: str = "http://localhost:8080"
    model_path: str = Field(default_factory=_auto_model_path)
    context_size: int = 4096
    threads: int = 4
    gpu_layers: int = 0
    batch_size: int = 512
    timeout: int = 120
    draft_model_url: Optional[str] = None  # set to enable speculative decoding
    draft_k_tokens: int = 5
    n_parallel: int = 1          # env: LLAMA_N_PARALLEL (match --parallel N in start_llama.sh)
    batch_timeout_ms: float = 0.0  # env: LLAMA_BATCH_TIMEOUT_MS (>0 enables time-window batching)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages: "


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="L", env_file=".env", extra="ignore")

    # L1 – hot conversation buffer
    l1_max_turns: int = Field(default=20, alias="L1_MAX_TURNS")
    l1_max_tokens: int = Field(default=512, alias="L1_MAX_TOKENS")
    l1_ttl_seconds: int = Field(default=3600, alias="L1_TTL_SECONDS")

    # L2 – rolling summary store
    l2_max_tokens: int = Field(default=1024, alias="L2_MAX_TOKENS")
    l2_summary_every_n_turns: int = Field(default=10, alias="L2_SUMMARY_EVERY_N_TURNS")

    # L3 – vector memory
    l3_max_results: int = Field(default=8, alias="L3_MAX_RESULTS")
    l3_similarity_threshold: float = Field(default=0.72, alias="L3_SIMILARITY_THRESHOLD")

    # L4 – persistent knowledge
    l4_max_results: int = Field(default=5, alias="L4_MAX_RESULTS")

    # Async compression queue (v2.1)
    compression_queue_maxsize: int = Field(default=32, alias="COMPRESSION_QUEUE_MAXSIZE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    faiss_index_path: str = "data/faiss_index"
    documents_path: str = "data/documents"
    chunk_size: int = 512
    chunk_overlap: int = 64
    rag_top_k: int = 6
    rag_rerank_top_n: int = 3

    # Hybrid search (BM25 + vector)
    use_hybrid_search: bool = False
    hybrid_alpha: float = 0.6          # 0 = pure BM25, 1 = pure semantic

    # Cross-encoder reranking
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Semantic chunking
    use_semantic_chunking: bool = False
    semantic_chunk_threshold: float = 0.75


class FusionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    fusion_semantic_weight: float = 0.45
    fusion_recency_weight: float = 0.25
    fusion_priority_weight: float = 0.20
    fusion_importance_weight: float = 0.10
    fusion_dedup_threshold: float = 0.88
    context_token_budget: int = 1400
    system_token_reserve: int = 256
    conversation_token_reserve: int = 512

    @field_validator(
        "fusion_semantic_weight",
        "fusion_recency_weight",
        "fusion_priority_weight",
        "fusion_importance_weight",
    )
    @classmethod
    def weights_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0 and 1")
        return v


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEFAULT_", env_file=".env", extra="ignore")

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = ["</s>", "[/INST]", "[INST]"]


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    cors_origins: str = "*"
    key: str = "changeme-in-production"     # env: API_KEY
    auth_enabled: bool = False              # env: API_AUTH_ENABLED
    rate_limit_requests: int = 60
    rate_limit_window: int = 60
    shutdown_timeout_s: int = 30            # env: API_SHUTDOWN_TIMEOUT_S


class MultiAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    enabled: bool = True            # AGENT_ENABLED — set False to skip decomposition
    max_subtasks: int = 5           # AGENT_MAX_SUBTASKS
    max_parallel: int = 4           # AGENT_MAX_PARALLEL (concurrent agents per wave)
    task_timeout_s: float = 180.0   # AGENT_TASK_TIMEOUT_S (per sub-task timeout)
    max_retries: int = 1          # AGENT_MAX_RETRIES (per subtask)
    retry_delay_base_s: float = 2.0  # AGENT_RETRY_DELAY_BASE_S


class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sqlite_path: str = "data/synapseos.db"   # legacy: kept for backward compat
    memory_db_path: str = "data/memory.db"   # L2 summaries + L4 knowledge
    tasks_db_path: str = "data/tasks.db"     # TaskStore + AuditLog
    data_dir: str = "data"
    models_dir: str = "models"

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        for path in (self.sqlite_path, self.memory_db_path, self.tasks_db_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Unified settings object – singleton via get_settings()."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llama: LlamaSettings = Field(default_factory=LlamaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    agent: MultiAgentSettings = Field(default_factory=MultiAgentSettings)

    debug: bool = False
    mock_llm: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
