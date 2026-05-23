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

    # Verify llama.cpp connectivity (non-blocking warning)
    llama_ok = await container.llama_client.health_check()
    if llama_ok:
        log.info("llama.cpp server is reachable")
    else:
        log.warning(
            "llama.cpp server is not reachable",
            url=cfg.llama.server_url,
            hint="Start the server with: llama-server -m <model.gguf> --host 0.0.0.0 --port 8080",
        )

    log.info("SynapseOS ready to serve")
    yield

    # Shutdown
    log.info("SynapseOS shutting down...")
    await container.llama_client.close()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="SynapseOS – AI Operating System",
        description=(
            "An AI Operating System for orchestrating intelligent multi-model systems. "
            "Features multi-tier memory (L1-L4), RAG retrieval, expert routing, "
            "and adaptive context fusion."
        ),
        version="1.0.0",
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
