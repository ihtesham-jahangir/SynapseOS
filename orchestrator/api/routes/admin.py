"""
Admin API – session management, cache control, benchmark endpoints.

DELETE /v1/admin/sessions/{session_id}  – clear a session's memory
POST   /v1/admin/benchmark              – run a benchmark suite
GET    /v1/admin/cache/stats            – KV cache statistics
DELETE /v1/admin/cache                  – clear KV cache
POST   /v1/admin/classify               – test intent classifier
"""
from __future__ import annotations

import time
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from orchestrator.core.types import Intent
from orchestrator.router.intent_classifier import HybridIntentClassifier
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.cache.kv_manager import KVCacheManager
from orchestrator.api.dependencies import get_container, get_memory_manager, get_kv_cache
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/v1/admin", tags=["admin"])


class ClassifyRequest(BaseModel):
    text: str


class BenchmarkRequest(BaseModel):
    prompts: List[str]
    iterations: int = 3


class BenchmarkResult(BaseModel):
    prompt: str
    avg_latency_ms: float
    intent: str
    confidence: float


@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
) -> dict:
    """Clear all memory tiers for a specific session."""
    await memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.post("/classify", response_model=Dict)
async def classify_intent(
    request: ClassifyRequest,
) -> Dict:
    """Test the intent classifier against a text string."""
    container = get_container()
    intent: Intent = await container.intent_classifier.classify(request.text)
    return {
        "text": request.text[:100],
        "intent": intent.intent_type.value,
        "confidence": round(intent.confidence, 3),
        "requires_expert": intent.requires_expert,
        "requires_rag": intent.requires_rag,
    }


@router.post("/benchmark", response_model=List[BenchmarkResult])
async def run_benchmark(
    request: BenchmarkRequest,
) -> List[BenchmarkResult]:
    """
    Run lightweight benchmark: classify + route (without LLM inference).
    Useful for measuring orchestration overhead without GPU.
    """
    container = get_container()
    results: List[BenchmarkResult] = []

    for prompt in request.prompts:
        latencies = []
        last_intent = None

        for _ in range(request.iterations):
            t0 = time.perf_counter()
            intent = await container.intent_classifier.classify(prompt)
            _ = container.routing_engine.route(intent)
            latencies.append((time.perf_counter() - t0) * 1000)
            last_intent = intent

        results.append(
            BenchmarkResult(
                prompt=prompt[:80],
                avg_latency_ms=sum(latencies) / len(latencies),
                intent=last_intent.intent_type.value,
                confidence=round(last_intent.confidence, 3),
            )
        )

    return results


@router.get("/cache/stats")
async def get_cache_stats(
    kv_cache: KVCacheManager = Depends(get_kv_cache),
) -> dict:
    return kv_cache.get_stats()


@router.delete("/cache")
async def clear_kv_cache(
    kv_cache: KVCacheManager = Depends(get_kv_cache),
) -> dict:
    kv_cache.clear()
    return {"status": "cleared"}


@router.get("/experts")
async def list_experts() -> dict:
    container = get_container()
    return {"experts": container.expert_manager.available_experts()}
