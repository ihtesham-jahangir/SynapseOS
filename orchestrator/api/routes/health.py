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
import json
import time
from typing import List

import aiosqlite
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

    # Check draft model server (optional — only present when speculative decoding is enabled)
    if container.speculative_decoder is not None and container.speculative_decoder.enabled:
        t0 = time.perf_counter()
        draft_ok = await container.draft_client.health_check()
        components.append(
            ComponentHealth(
                name="draft_server",
                healthy=draft_ok,
                latency_ms=(time.perf_counter() - t0) * 1000,
                details=(
                    {"url": container.draft_client._base_url, "mode": "speculative_decoding"}
                    if draft_ok
                    else {"error": "unreachable", "hint": "run ./start_draft.sh"}
                ),
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

    # Check compression queue worker (v2.1)
    cq = container.compression_queue
    cq_worker_alive = (
        cq._worker_task is not None and not cq._worker_task.done()
    ) if cq._started else True  # not started yet → not unhealthy
    components.append(
        ComponentHealth(
            name="compression_queue",
            healthy=cq_worker_alive,
            details={
                "started": cq._started,
                "queue_depth": cq.queue_depth,
                "worker_alive": cq_worker_alive,
            },
        )
    )

    # Check agent pool (v3.0)
    components.append(
        ComponentHealth(
            name="agent_pool",
            healthy=True,
            details={
                "active_agents": container.agent_pool.active_count,
                "bus_topics": container.bus.topic_count,
            },
        )
    )

    # Check llama.cpp circuit breaker state
    cb_state = container.llama_client._circuit.state
    components.append(
        ComponentHealth(
            name="llama_circuit_breaker",
            healthy=cb_state != "open",
            details=container.llama_client._circuit.status(),
        )
    )

    # Check embedding circuit breaker state
    emb_cb_state = container.embedder._circuit.state
    components.append(
        ComponentHealth(
            name="embedding_circuit_breaker",
            healthy=emb_cb_state != "open",
            details=container.embedder._circuit.status(),
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
    """Kubernetes readiness probe — 200 only when llama.cpp AND SQLite are reachable."""
    container = get_container()
    from orchestrator.config.settings import get_settings
    cfg = get_settings()

    llama_ok, sqlite_ok = await asyncio.gather(
        container.llama_client.health_check(),
        _check_sqlite(cfg.storage.sqlite_path),
        return_exceptions=False,
    )

    if llama_ok and sqlite_ok:
        return {"status": "ready"}

    reasons = []
    if not llama_ok:
        reasons.append("llama_server_unreachable")
    if not sqlite_ok:
        reasons.append("sqlite_unavailable")

    return Response(
        content=json.dumps({"status": "not_ready", "reason": reasons}),
        status_code=503,
        media_type="application/json",
    )


async def _check_sqlite(db_path: str) -> bool:
    try:
        async with aiosqlite.connect(db_path) as db:
            await asyncio.wait_for(db.execute("SELECT 1"), timeout=1.0)
        return True
    except Exception:
        return False


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
    spec = container.speculative_decoder
    return {
        "version": "3.0.0",
        "uptime_seconds": time.time() - _START_TIME,
        "rag_index_size": rag.index_size,
        "active_ws_sessions": ws_mgr.active_session_count(),
        "l1_active_sessions": len(container.l1.active_sessions()),
        "kv_cache_stats": container.kv_cache.get_stats(),
        "speculative_decoding": {
            "enabled": spec is not None and spec.enabled,
            "k_tokens": spec._k if spec is not None else None,
        },
        "inference_batcher": {
            "n_parallel": container.batcher.n_parallel,
            "in_flight": container.batcher.in_flight,
            "queue_depth": container.batcher.queue_depth,
        },
        "compression_queue": {
            "queue_depth": container.compression_queue.queue_depth,
        },
        "multi_agent": {
            "enabled": True,
            "active_agents": container.agent_pool.active_count,
            "bus_topics": container.bus.topic_count,
        },
    }
