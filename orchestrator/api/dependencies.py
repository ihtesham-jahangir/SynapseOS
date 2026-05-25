"""
Dependency Injection container for FastAPI.

All major subsystems are singletons created once at startup.
FastAPI's Depends() mechanism distributes them to route handlers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from orchestrator.config.settings import get_settings
from orchestrator.rag.embedder import BGEEmbedder
from orchestrator.rag.indexer import FAISSIndex
from orchestrator.rag.pipeline import RAGPipeline
from orchestrator.memory.l1_cache import L1ConversationCache
from orchestrator.memory.l2_cache import L2SummaryCache
from orchestrator.memory.l3_cache import L3VectorMemory
from orchestrator.memory.l4_cache import L4KnowledgeBase
from orchestrator.memory.compressor import ContextCompressor
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.router.intent_classifier import HybridIntentClassifier
from orchestrator.router.routing_engine import RoutingEngine
from orchestrator.experts.expert_manager import ExpertManager
from orchestrator.fusion.fusion_engine import AdaptiveFusionEngine
from orchestrator.runtime.llama_client import LlamaClient
from orchestrator.runtime.inference_engine import InferenceEngine
from orchestrator.runtime.adaptive_compute import AdaptiveComputeController
from orchestrator.runtime.speculative import SpeculativeDecoder
from orchestrator.runtime.request_batcher import InferenceBatcher
from orchestrator.memory.compression_queue import CompressionQueue
from orchestrator.streaming.websocket_manager import WebSocketManager
from orchestrator.cache.kv_manager import KVCacheManager
from orchestrator.cache.response_cache import ResponseCache
from orchestrator.agents.memory_bus import SharedMemoryBus
from orchestrator.agents.task_planner import TaskPlanner
from orchestrator.agents.agent_pool import AgentPool
from orchestrator.agents.multi_agent_engine import MultiAgentEngine
from orchestrator.agents.task_store import TaskStore
from orchestrator.utils.audit_log import AuditLog


class Container:
    """Singleton DI container – initialized once at application startup."""

    _instance: Optional["Container"] = None

    def __init__(self) -> None:
        cfg = get_settings()
        cfg.storage.ensure_dirs()

        # ── Shared embedding model ─────────────────────────────────────────
        self.embedder = BGEEmbedder()

        # ── FAISS indices ──────────────────────────────────────────────────
        self.rag_index = FAISSIndex(
            dimension=384,
            index_path=cfg.rag.faiss_index_path,
        )
        self.memory_index = FAISSIndex(
            dimension=384,
            index_path=f"{cfg.rag.faiss_index_path}_memory",
        )

        # ── Optional RAG components (feature-flag controlled) ──────────────
        _rag_cfg = cfg.rag

        # Semantic chunker (replaces RecursiveTextChunker when enabled)
        if _rag_cfg.use_semantic_chunking:
            from orchestrator.rag.chunker import SemanticChunker
            _chunker = SemanticChunker(
                embedder=self.embedder,
                breakpoint_threshold=_rag_cfg.semantic_chunk_threshold,
                max_chunk_tokens=_rag_cfg.chunk_size,
            )
        else:
            _chunker = None  # pipeline uses RecursiveTextChunker default

        # BM25 index + hybrid retriever
        if _rag_cfg.use_hybrid_search:
            from orchestrator.rag.bm25_index import BM25Index
            from orchestrator.rag.retriever import HybridRetriever, SemanticRetriever
            self.bm25_index = BM25Index(
                persist_path=f"{_rag_cfg.faiss_index_path}_bm25.json"
            )
            self.bm25_index.load()
            _retriever = HybridRetriever(
                semantic=SemanticRetriever(self.embedder, self.rag_index),
                bm25=self.bm25_index,
                alpha=_rag_cfg.hybrid_alpha,
            )
        else:
            self.bm25_index = None
            _retriever = None  # pipeline uses SemanticRetriever default

        # Cross-encoder reranker
        if _rag_cfg.use_cross_encoder:
            from orchestrator.rag.reranker import CrossEncoderReranker
            _reranker = CrossEncoderReranker(model_name=_rag_cfg.cross_encoder_model)
        else:
            _reranker = None  # pipeline uses EmbeddingReranker default

        # ── RAG pipeline ───────────────────────────────────────────────────
        self.rag_pipeline = RAGPipeline(
            embedder=self.embedder,
            index=self.rag_index,
            chunker=_chunker,
            retriever=_retriever,
            reranker=_reranker,
            bm25_index=self.bm25_index,
        )

        # ── Memory tiers ───────────────────────────────────────────────────
        self.l1 = L1ConversationCache(
            max_turns=cfg.memory.l1_max_turns,
            max_tokens=cfg.memory.l1_max_tokens,
            ttl_seconds=cfg.memory.l1_ttl_seconds,
        )
        self.l2 = L2SummaryCache(db_path=cfg.storage.memory_db_path)
        self.l3 = L3VectorMemory(
            embedder=self.embedder,
            index=self.memory_index,
        )
        self.l4 = L4KnowledgeBase(db_path=cfg.storage.memory_db_path)

        # ── llama.cpp client ───────────────────────────────────────────────
        self.llama_client = LlamaClient(
            base_url=cfg.llama.server_url,
            timeout=cfg.llama.timeout,
        )

        # ── Compressor (needs llama client) ───────────────────────────────
        self.compressor = ContextCompressor(llama_client=self.llama_client)

        # ── Async compression queue (v2.1) ─────────────────────────────────
        self.compression_queue = CompressionQueue(
            compressor=self.compressor,
            l2=self.l2,
            maxsize=cfg.memory.compression_queue_maxsize,
        )

        # ── Memory manager ─────────────────────────────────────────────────
        self.memory_manager = MemoryManager(
            l1=self.l1,
            l2=self.l2,
            l3=self.l3,
            l4=self.l4,
            compressor=self.compressor,
            compression_queue=self.compression_queue,
        )

        # ── Router ─────────────────────────────────────────────────────────
        self.intent_classifier = HybridIntentClassifier(embedder=self.embedder)
        self.routing_engine = RoutingEngine()

        # ── Experts ────────────────────────────────────────────────────────
        self.expert_manager = ExpertManager(embedder=self.embedder)

        # ── Fusion ─────────────────────────────────────────────────────────
        self.fusion_engine = AdaptiveFusionEngine()

        # ── Compute controller ─────────────────────────────────────────────
        self.compute_controller = AdaptiveComputeController()

        # ── Response cache (with semantic similarity layer) ────────────────
        self.response_cache = ResponseCache(
            maxsize=512,
            ttl_seconds=3600,
            embedder=self.embedder.embed_query,
        )

        # ── Inference batcher (v2.1) — controls llama.cpp slot concurrency ──
        self.batcher = InferenceBatcher(
            n_parallel=cfg.llama.n_parallel,
            batch_timeout_ms=cfg.llama.batch_timeout_ms,
        )

        # ── Speculative decoder (optional — needs draft_model_url in .env) ─
        if cfg.llama.draft_model_url:
            self.draft_client = LlamaClient(
                base_url=cfg.llama.draft_model_url,
                timeout=cfg.llama.timeout,
            )
            self.speculative_decoder: Optional[SpeculativeDecoder] = SpeculativeDecoder(
                verifier_client=self.llama_client,
                draft_client=self.draft_client,
                k_tokens=cfg.llama.draft_k_tokens,
            )
        else:
            self.speculative_decoder = None

        # ── Inference engine ───────────────────────────────────────────────
        self.inference_engine = InferenceEngine(
            llama_client=self.llama_client,
            memory_manager=self.memory_manager,
            rag_pipeline=self.rag_pipeline,
            intent_classifier=self.intent_classifier,
            routing_engine=self.routing_engine,
            expert_manager=self.expert_manager,
            fusion_engine=self.fusion_engine,
            compute_controller=self.compute_controller,
            response_cache=self.response_cache,
            speculative_decoder=self.speculative_decoder,
            batcher=self.batcher,
        )

        # ── Streaming ──────────────────────────────────────────────────────
        self.ws_manager = WebSocketManager()

        # ── KV cache ───────────────────────────────────────────────────────
        self.kv_cache = KVCacheManager()

        # ── Multi-agent system (v3.0) ──────────────────────────────────────
        self.bus = SharedMemoryBus()
        self.task_store = TaskStore(db_path=cfg.storage.tasks_db_path)
        self.audit_log = AuditLog(db_path=cfg.storage.tasks_db_path)
        self.task_planner = TaskPlanner(llama_client=self.llama_client)
        # Cap agent concurrency to the llama server's actual slot count so we
        # never queue more requests than it can serve simultaneously.
        effective_parallel = min(cfg.agent.max_parallel, cfg.llama.n_parallel)
        self.agent_pool = AgentPool(
            llama_client=self.llama_client,
            bus=self.bus,
            max_parallel=effective_parallel,
            task_timeout_s=cfg.agent.task_timeout_s,
            max_retries=cfg.agent.max_retries,
            retry_delay_base_s=cfg.agent.retry_delay_base_s,
        )
        self.multi_agent_engine = MultiAgentEngine(
            planner=self.task_planner,
            pool=self.agent_pool,
            bus=self.bus,
            inference_engine=self.inference_engine,
            task_store=self.task_store,
            response_cache=self.response_cache,
        )

    @classmethod
    def get(cls) -> "Container":
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance


def get_container() -> Container:
    return Container.get()


# FastAPI dependency shortcuts
def get_inference_engine() -> InferenceEngine:
    return Container.get().inference_engine


def get_multi_agent_engine() -> MultiAgentEngine:
    return Container.get().multi_agent_engine


def get_rag_pipeline() -> RAGPipeline:
    return Container.get().rag_pipeline


def get_memory_manager() -> MemoryManager:
    return Container.get().memory_manager


def get_ws_manager() -> WebSocketManager:
    return Container.get().ws_manager


def get_l4() -> L4KnowledgeBase:
    return Container.get().l4


def get_kv_cache() -> KVCacheManager:
    return Container.get().kv_cache


def get_audit_log() -> AuditLog:
    return Container.get().audit_log
