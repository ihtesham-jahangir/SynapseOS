"""
FastAPI application factory.

Creates and configures the ASGI app with:
  - All route registrations
  - CORS middleware
  - Logging middleware
  - Rate limiting
  - Lifespan events (startup / shutdown)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from orchestrator.api.routes import (
    chat_router,
    documents_router,
    health_router,
    admin_router,
    agents_router,
)
from orchestrator.api.middleware import (
    APIKeyMiddleware,
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from orchestrator.api.dependencies import Container
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

_L1_CLEANUP_INTERVAL_S = 300  # run TTL sweep every 5 minutes


async def _l1_ttl_cleanup_loop(container: Container) -> None:
    """Background coroutine: periodically evict idle L1 sessions."""
    while True:
        await asyncio.sleep(_L1_CLEANUP_INTERVAL_S)
        try:
            evicted = await container.l1.evict_expired()
            if evicted:
                log.info("L1 TTL sweep complete", evicted=len(evicted))
        except Exception as exc:
            log.warning("L1 TTL sweep error", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Startup: initialize the DI container (loads all subsystems).
    Shutdown: close HTTP clients and flush pending data.
    """
    cfg = get_settings()
    configure_logging(level=cfg.api.log_level)

    log.info(
        "SynapseOS starting",
        llama_url=cfg.llama.server_url,
        embedding_model=cfg.embedding.model,
    )

    # Initialize singleton container (triggers lazy loads)
    container = Container.get()

    # Warm up embedding model (first call loads the model weights)
    log.info("Warming up embedding model...")
    try:
        await container.embedder.embed(["warmup"])
        log.info("Embedding model ready")
    except Exception as exc:
        log.warning("Embedding warmup failed (will retry on first request)", error=str(exc))

    # Verify llama.cpp verifier connectivity (non-blocking warning)
    llama_ok = await container.llama_client.health_check()
    if llama_ok:
        log.info("llama.cpp verifier server is reachable")
    else:
        log.warning(
            "llama.cpp verifier server is not reachable",
            url=cfg.llama.server_url,
            hint="Start the verifier with: ./start_llama.sh",
        )

    # Verify draft model server connectivity (optional — only when configured)
    if container.speculative_decoder is not None and container.speculative_decoder.enabled:
        draft_ok = await container.draft_client.health_check()
        if draft_ok:
            log.info(
                "Speculative decoding ACTIVE",
                draft_url=cfg.llama.draft_model_url,
                k_tokens=cfg.llama.draft_k_tokens,
            )
        else:
            log.warning(
                "Draft model server not reachable — speculative decoding will fall back to verifier",
                draft_url=cfg.llama.draft_model_url,
                hint="Start the draft server with: ./start_draft.sh",
            )
    else:
        log.info(
            "Speculative decoding disabled",
            hint="Set LLAMA_DRAFT_MODEL_URL=http://localhost:8081 in .env and run ./start_draft.sh",
        )

    # Start async compression worker (v2.1)
    await container.compression_queue.start()
    log.info(
        "Compression queue worker started",
        maxsize=cfg.memory.compression_queue_maxsize,
    )

    # Start L1 TTL background sweep
    cleanup_task = asyncio.create_task(_l1_ttl_cleanup_loop(container))

    log.info(
        "SynapseOS v3.4 ready to serve",
        n_parallel=cfg.llama.n_parallel,
        batch_timeout_ms=cfg.llama.batch_timeout_ms,
        agent_max_parallel=cfg.agent.max_parallel,
        agent_max_subtasks=cfg.agent.max_subtasks,
        l1_ttl_seconds=cfg.memory.l1_ttl_seconds,
    )
    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Shutdown — wait for in-flight agents, then drain queue and close connections
    log.info("SynapseOS shutting down...")
    drain_deadline = asyncio.get_event_loop().time() + cfg.api.shutdown_timeout_s
    while container.agent_pool.active_count > 0:
        remaining = drain_deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            log.warning(
                "Shutdown drain timeout — forcing exit with agents still active",
                active=container.agent_pool.active_count,
            )
            break
        log.info("Draining active agents...", active=container.agent_pool.active_count)
        await asyncio.sleep(1)
    await container.compression_queue.stop()
    await container.llama_client.close()
    # Close persistent SQLite connections
    await container.l2.close()
    await container.l4.close()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="SynapseOS – AI Operating System",
        description=(
            "An AI Operating System for orchestrating intelligent multi-model systems. "
            "Features multi-tier memory (L1-L4), RAG retrieval, expert routing, "
            "adaptive context fusion, and speculative decoding (v2.0)."
        ),
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (order matters: outermost applied last) ─────────────────
    origins = (
        cfg.api.cors_origins.split(",")
        if cfg.api.cors_origins != "*"
        else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Structured error handlers ────────────────────────────────────────────
    from fastapi.exceptions import RequestValidationError
    from fastapi import HTTPException as FastAPIHTTPException

    @app.exception_handler(FastAPIHTTPException)
    async def http_exception_handler(request: Request, exc: FastAPIHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": _status_to_code(exc.status_code),
                    "message": exc.detail,
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        log.error("Unhandled exception", error=str(exc), request_id=request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                }
            },
        )

    # ── Routes ──────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(admin_router)
    app.include_router(agents_router)

    @app.get("/")
    async def root():
        return {
            "name": "SynapseOS",
            "version": "3.4.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


def _status_to_code(status: int) -> str:
    _map = {
        400: "BAD_REQUEST", 401: "UNAUTHORIZED", 403: "FORBIDDEN",
        404: "NOT_FOUND", 409: "CONFLICT", 422: "VALIDATION_ERROR",
        429: "RATE_LIMITED", 500: "INTERNAL_ERROR", 502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE", 504: "GATEWAY_TIMEOUT",
    }
    return _map.get(status, f"HTTP_{status}")


app = create_app()
