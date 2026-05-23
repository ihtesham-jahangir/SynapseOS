"""
Health and metrics endpoints.

GET /health        – full system health check
GET /health/live   – simple liveness probe (k8s-compatible)
GET /health/ready  – readiness probe
GET /metrics       – Prometheus metrics
GET /v1/stats      – runtime statistics
"""
from __future__ import annotations

import asyncio
import time
from typing import List

from fastapi import APIRouter, Depends, Response
from fastapi.responses import PlainTextResponse

from orchestrator.core.types import ComponentHealth, HealthResponse
from orchestrator.runtime.llama_client import LlamaClient
from orchestrator.rag.pipeline import RAGPipeline
from orchestrator.streaming.websocket_manager import WebSocketManager
from orchestrator.api.dependencies import get_container, get_rag_pipeline, get_ws_manager
from orchestrator.utils.metrics import get_metrics_output, METRICS_CONTENT_TYPE
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["health"])

_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    rag: RAGPipeline = Depends(get_rag_pipeline),
    ws_mgr: WebSocketManager = Depends(get_ws_manager),
) -> HealthResponse:
    """Full system health check with per-component status."""
    container = get_container()
    components: List[ComponentHealth] = []

    # Check llama.cpp
    t0 = time.perf_counter()
    llama_ok = await container.llama_client.health_check()
    components.append(
        ComponentHealth(
            name="llama_server",
            healthy=llama_ok,
            latency_ms=(time.perf_counter() - t0) * 1000,
            details={"url": container.llama_client._base_url} if llama_ok else {"error": "unreachable"},
        )
    )

    # Check embedding model (just verify it's loaded)
    try:
        t0 = time.perf_counter()
        await container.embedder.embed(["health check"])
        emb_ok = True
        emb_latency = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        emb_ok = False
        emb_latency = 0
        log.warning("Embedding health check failed", error=str(exc))

    components.append(
        ComponentHealth(
            name="embedding_model",
            healthy=emb_ok,
            latency_ms=emb_latency,
            details={"model": container.embedder._settings.model},
        )
    )

    # Check FAISS
    components.append(
        ComponentHealth(
            name="faiss_index",
            healthy=True,
            details={"total_vectors": rag.index_size},
        )
    )

    # Check active WebSocket sessions
    components.append(
        ComponentHealth(
            name="websocket_manager",
            healthy=True,
            details={"active_sessions": ws_mgr.active_session_count()},
        )
    )

    all_healthy = all(c.healthy for c in components)
    critical_ok = all(c.healthy for c in components if c.name == "llama_server")

    status = "healthy" if all_healthy else ("degraded" if critical_ok else "unhealthy")

    return HealthResponse(
        status=status,
        components=components,
        uptime_seconds=time.time() - _START_TIME,
    )


@router.get("/health/live")
async def liveness() -> dict:
    """Kubernetes liveness probe – always 200 if process is alive."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness() -> dict:
    """Kubernetes readiness probe – 200 only when llama.cpp is reachable."""
    container = get_container()
    if await container.llama_client.health_check():
        return {"status": "ready"}
    return Response(
        content='{"status": "not_ready", "reason": "llama_server_unreachable"}',
        status_code=503,
        media_type="application/json",
    )


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics_output(),
        media_type=METRICS_CONTENT_TYPE,
    )


@router.get("/v1/stats")
async def runtime_stats(
    rag: RAGPipeline = Depends(get_rag_pipeline),
    ws_mgr: WebSocketManager = Depends(get_ws_manager),
) -> dict:
    """Human-readable runtime statistics."""
    container = get_container()
    return {
        "uptime_seconds": time.time() - _START_TIME,
        "rag_index_size": rag.index_size,
        "active_ws_sessions": ws_mgr.active_session_count(),
        "l1_active_sessions": len(container.l1.active_sessions()),
        "kv_cache_stats": container.kv_cache.get_stats(),
    }
