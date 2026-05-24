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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orchestrator.api.routes import (
    chat_router,
    documents_router,
    health_router,
    admin_router,
    agents_router,
)
from orchestrator.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware, RateLimitMiddleware
from orchestrator.api.dependencies import Container
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)


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

    log.info(
        "SynapseOS v3.0 ready to serve",
        n_parallel=cfg.llama.n_parallel,
        batch_timeout_ms=cfg.llama.batch_timeout_ms,
        agent_max_parallel=cfg.agent.max_parallel,
        agent_max_subtasks=cfg.agent.max_subtasks,
    )
    yield

    # Shutdown — drain compression queue before closing connections
    log.info("SynapseOS shutting down...")
    await container.compression_queue.stop()
    await container.llama_client.close()
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
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
