"""
Main Inference Engine – the top-level orchestrator.

Executes the full SynapseOS pipeline per request:

  User Query
    │
    ├─► Response Cache check  (< 1 ms — returns instantly on hit)
    │
    ├─► Intent Classification
    │
    ├─► Parallel Fan-Out:
    │     ├─ Memory retrieval (all 4 tiers)
    │     ├─ RAG retrieval (if needed)
    │     └─ Expert guidance generation
    │
    ├─► Adaptive Fusion Engine
    │
    ├─► Adaptive Compute Optimization
    │
    ├─► Llama Generation (via llama.cpp)
    │
    ├─► Background tasks (fire-and-forget):
    │     ├─ Memory update (record_turn → L1 → L2 compression)
    │     └─ L3 fact extraction (heuristic, no extra LLM call)
    │
    ├─► Response Cache store (cacheable intents only)
    │
    └─► Response
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

from orchestrator.core.base import BaseInferenceEngine
from orchestrator.core.types import (
    ChatRequest,
    ChatResponse,
    ExpertType,
    FusedContext,
    GenerationParams,
    InferenceResponse,
    Intent,
    IntentType,
    MemoryLevel,
    Message,
    MessageRole,
)


@dataclass
class GenerationPlan:
    """Everything the streaming/generation paths need, built once."""
    fused: FusedContext
    params: GenerationParams
    intent: Intent
    memory_levels: List[MemoryLevel] = field(default_factory=list)
    rag_chunks: int = 0
    session_id: str = ""
    expert_used: Optional[ExpertType] = None
from orchestrator.core.exceptions import LlamaServerError, OrchestratorError
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger, bind_request_context
from orchestrator.utils.metrics import (
    LatencyTracker,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TTFT_HISTOGRAM,
    TOKEN_THROUGHPUT,
    LLAMA_ERRORS,
)
from orchestrator.utils.async_utils import gather_with_fallback
from orchestrator.utils.fact_extractor import extract_facts, should_extract

from orchestrator.router.intent_classifier import HybridIntentClassifier
from orchestrator.router.routing_engine import RoutingEngine
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.rag.pipeline import RAGPipeline
from orchestrator.experts.expert_manager import ExpertManager
from orchestrator.fusion.fusion_engine import AdaptiveFusionEngine
from orchestrator.runtime.adaptive_compute import AdaptiveComputeController
from orchestrator.runtime.llama_client import LlamaClient
from orchestrator.runtime.speculative import SpeculativeDecoder
from orchestrator.runtime.request_batcher import InferenceBatcher
from orchestrator.cache.response_cache import ResponseCache

log = get_logger(__name__)


class InferenceEngine(BaseInferenceEngine):
    """
    SynapseOS multi-path inference pipeline.

    All components are dependency-injected for easy testing and replacement.
    ``response_cache`` is optional — pass None to disable caching entirely.
    """

    def __init__(
        self,
        llama_client: LlamaClient,
        memory_manager: MemoryManager,
        rag_pipeline: RAGPipeline,
        intent_classifier: HybridIntentClassifier,
        routing_engine: RoutingEngine,
        expert_manager: ExpertManager,
        fusion_engine: AdaptiveFusionEngine,
        compute_controller: AdaptiveComputeController,
        response_cache: Optional[ResponseCache] = None,
        speculative_decoder: Optional[SpeculativeDecoder] = None,
        batcher: Optional[InferenceBatcher] = None,
    ) -> None:
        self._llama = llama_client
        self._memory = memory_manager
        self._rag = rag_pipeline
        self._classifier = intent_classifier
        self._router = routing_engine
        self._experts = expert_manager
        self._fusion = fusion_engine
        self._compute = compute_controller
        self._cache = response_cache
        self._speculative = speculative_decoder
        self._batcher = batcher

    # ── Shared pipeline stages (used by both process_request and stream routes) ─

    async def build_generation_plan(self, request: ChatRequest) -> GenerationPlan:
        """
        Run pipeline stages 1–6 (intent → routing → memory/RAG → fusion → compute).

        Returns a GenerationPlan that can be passed directly to generate() or stream().
        Both the blocking and streaming chat handlers use this so the pipeline
        logic lives in exactly one place.
        """
        session_id = request.session_id
        query = request.last_user_message

        intent = await self._classifier.classify(query)
        decision = self._router.route(intent)

        stored_conversation = await self._memory.get_conversation(session_id)
        request_history = [m for m in request.messages[:-1]]
        if request_history:
            seen_contents = {m.content for m in stored_conversation}
            for m in request_history:
                if m.content not in seen_contents:
                    stored_conversation.append(m)
        stored_conversation.append(Message(role=MessageRole.USER, content=query))
        conversation = stored_conversation

        memory_context = f"{query}\n" + " ".join(m.content for m in conversation[-5:-1])
        memory_result, rag_result, expert_guidance = await gather_with_fallback(
            self._memory.retrieve_all(memory_context, session_id),
            self._rag.retrieve(query) if decision.run_rag else _empty_rag(),
            self._experts.get_guidance(query, intent, decision.active_experts),
            fallbacks=[_empty_memory_result(), _empty_rag_result(), None],
        )

        fused = await self._fusion.fuse(
            query=query,
            intent=intent,
            memory_result=memory_result,
            rag_result=rag_result,
            expert_guidance=expert_guidance,
            conversation=conversation,
            system_prompt=decision.system_prompt_hint,
        )

        cfg = get_settings().generation
        request_base = GenerationParams(
            max_tokens=request.max_tokens or cfg.max_tokens,
            temperature=request.temperature or cfg.temperature,
            top_p=request.top_p or cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
        )
        params = self._compute.optimize(
            intent=intent,
            fused_context=fused,
            base_params=decision.generation_override or request_base,
        )

        return GenerationPlan(
            fused=fused,
            params=params,
            intent=intent,
            memory_levels=memory_result.levels_queried,
            rag_chunks=len(rag_result.chunks),
            session_id=session_id,
            expert_used=expert_guidance.expert_type if expert_guidance else None,
        )

    # ── Main pipeline ─────────────────────────────────────────────────────────

    async def process_request(self, request: ChatRequest) -> ChatResponse:
        """Full non-streaming inference pipeline."""
        tracker = LatencyTracker()
        session_id = request.session_id
        query = request.last_user_message

        bind_request_context(session_id)

        try:
            # ── Stage 1: Intent classification (needed for cache key) ─────────
            intent = await self._classifier.classify(query)
            tracker.checkpoint("intent")

            # ── Stage 2: Response cache check ─────────────────────────────────
            if self._cache and self._cache.should_cache(query, intent.intent_type.value):
                cached = await self._cache.get(query)
                if cached:
                    tracker.checkpoint("cache_hit")
                    REQUEST_COUNT.labels(
                        intent=intent.intent_type.value, status="cache_hit"
                    ).inc()
                    log.info(
                        "Returning cached response",
                        session=session_id,
                        intent=intent.intent_type.value,
                        cache_hits=cached.hit_count,
                    )
                    return ChatResponse(
                        session_id=session_id,
                        content=cached.content,
                        intent=intent.intent_type,
                        tokens_generated=cached.tokens_generated,
                        total_tokens=cached.tokens_generated,
                        time_to_first_token_ms=0.0,
                        total_time_ms=tracker.elapsed_ms(),
                        memory_levels_used=[],
                        rag_chunks_used=0,
                        expert_used=None,
                        metadata={
                            **tracker.report(),
                            "from_cache": True,
                            "cache_hits": cached.hit_count,
                        },
                    )

            # ── Stages 3–6: routing → memory/RAG → fusion → compute ───────────
            plan = await self.build_generation_plan(request)
            tracker.checkpoint("pipeline")

            # ── Stage 7: Llama inference ──────────────────────────────────────
            inference_resp = await self.generate(plan.fused, plan.params)
            tracker.checkpoint("inference")

            # Unpack aliases needed below
            fused = plan.fused
            gen_params = plan.params
            memory_result_levels = plan.memory_levels
            rag_chunks = plan.rag_chunks
            expert_used = plan.expert_used

            total_ms = tracker.elapsed_ms()

            # ── Stage 8: Background tasks ─────────────────────────────────────
            user_msg = Message(role=MessageRole.USER, content=query)
            asst_msg = Message(role=MessageRole.ASSISTANT, content=inference_resp.content)

            asyncio.create_task(
                self._memory.record_turn(session_id, user_msg, asst_msg)
            )
            asyncio.create_task(
                self._populate_l3(session_id, query, inference_resp.content)
            )

            # ── Stage 9: Populate response cache ──────────────────────────────
            if self._cache and self._cache.should_cache(query, intent.intent_type.value):
                asyncio.create_task(
                    self._cache.set(
                        query=query,
                        content=inference_resp.content,
                        intent_type=intent.intent_type.value,
                        tokens_generated=inference_resp.tokens_generated,
                    )
                )

            # ── Metrics ────────────────────────────────────────────────────────
            REQUEST_COUNT.labels(intent=intent.intent_type.value, status="success").inc()
            REQUEST_LATENCY.labels(intent=intent.intent_type.value).observe(total_ms / 1000)
            TTFT_HISTOGRAM.observe(inference_resp.time_to_first_token_ms / 1000)
            if total_ms > 0:
                tps = inference_resp.tokens_generated / (total_ms / 1000)
                TOKEN_THROUGHPUT.observe(tps)

            log.info(
                "Request complete",
                session=session_id,
                intent=intent.intent_type.value,
                tokens_gen=inference_resp.tokens_generated,
                total_ms=f"{total_ms:.0f}",
            )

            return ChatResponse(
                session_id=session_id,
                content=inference_resp.content,
                intent=intent.intent_type,
                tokens_generated=inference_resp.tokens_generated,
                total_tokens=fused.total_token_count + inference_resp.tokens_generated,
                time_to_first_token_ms=inference_resp.time_to_first_token_ms,
                total_time_ms=total_ms,
                memory_levels_used=memory_result_levels,
                rag_chunks_used=rag_chunks,
                expert_used=expert_used,
                metadata={
                    **tracker.report(),
                    "intent": intent.intent_type.value,
                    "subtype": intent.subtype,
                    "classification_stage": intent.classification_stage,
                    "secondary_intents": [
                        {"intent": s.intent_type.value, "confidence": s.confidence}
                        for s in intent.secondary_intents
                    ],
                    "multi_agent": False,
                },
            )

        except Exception as exc:
            REQUEST_COUNT.labels(intent="unknown", status="error").inc()
            LLAMA_ERRORS.labels(error_type=type(exc).__name__).inc()
            log.error("Request failed", session=session_id, error=str(exc), exc_info=True)
            raise

    # ── L3 background fact extraction ─────────────────────────────────────────

    async def _populate_l3(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """
        Extract personal facts from the user message and store them in L3.
        Runs in background — never blocks the response.
        """
        if not should_extract(user_message):
            return
        try:
            facts = extract_facts(user_message)
            if not facts:
                return
            for fact_text, importance in facts:
                await self._memory.store_fact(
                    session_id=session_id,
                    fact=fact_text,
                    importance=importance,
                )
            log.debug(
                "L3 facts stored",
                session=session_id,
                count=len(facts),
                facts=[f for f, _ in facts],
            )
        except Exception as exc:
            # Never let L3 extraction crash the pipeline
            log.warning("L3 fact extraction failed", error=str(exc))

    # ── Generation helpers ────────────────────────────────────────────────────

    async def generate(
        self,
        context: FusedContext,
        params: GenerationParams,
    ) -> InferenceResponse:
        """Run non-streaming generation (slot-controlled via InferenceBatcher)."""
        messages = context.to_messages()
        t0 = time.perf_counter()

        if self._batcher is not None:
            async with self._batcher.acquire():
                content = await self._run_generate(messages, params)
        else:
            content = await self._run_generate(messages, params)

        total_ms = (time.perf_counter() - t0) * 1000
        tokens_gen = len(content.split())

        return InferenceResponse(
            content=content,
            tokens_generated=tokens_gen,
            prompt_tokens=context.total_token_count,
            time_to_first_token_ms=total_ms * 0.3,
            total_time_ms=total_ms,
        )

    async def _run_generate(self, messages: list, params: GenerationParams) -> str:
        """Inner generation call — speculative or direct."""
        if self._speculative is not None:
            return await self._speculative.generate(
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repeat_penalty=params.repeat_penalty,
                stop=params.stop_sequences,
            )
        return await self._llama.chat(
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repeat_penalty,
            stop=params.stop_sequences,
        )

    async def stream(
        self,
        context: FusedContext,
        params: GenerationParams,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation — slot-controlled, yields token strings."""
        messages = context.to_messages()
        if self._batcher is not None:
            async with self._batcher.acquire():
                async for token in self._run_stream(messages, params):
                    yield token
        else:
            async for token in self._run_stream(messages, params):
                yield token

    async def _run_stream(self, messages: list, params: GenerationParams) -> AsyncGenerator[str, None]:
        """Inner streaming call."""
        async for token in self._llama.stream_chat(
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repeat_penalty,
            stop=params.stop_sequences,
        ):
            yield token

    async def health_check(self) -> bool:
        return await self._llama.health_check()


# ─── Empty result factories ───────────────────────────────────────────────────

def _empty_rag() -> object:
    """Awaitable yielding an empty RAGResult (used when RAG is skipped)."""
    from orchestrator.core.types import RAGResult
    async def _():
        return RAGResult(chunks=[], scores=[], total_tokens=0, query_time_ms=0)
    return _()

def _empty_rag_result():
    from orchestrator.core.types import RAGResult
    return RAGResult(chunks=[], scores=[], total_tokens=0, query_time_ms=0)

def _empty_memory_result():
    from orchestrator.core.types import MemoryResult
    return MemoryResult(items=[], total_tokens=0, levels_queried=[], query_time_ms=0)
