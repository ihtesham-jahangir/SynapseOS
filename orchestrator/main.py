"""
SynapseOS – AI Operating System
Entry point for production and development servers.

Usage:
    # Development
    python -m orchestrator.main

    # Production
    uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000 --workers 1

    # With CLI
    python -m orchestrator.main serve --port 8000
"""
from __future__ import annotations

import sys

import click
import uvicorn

from orchestrator.api.app import app  # noqa: F401 – re-exported for uvicorn
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import configure_logging


@click.group()
def cli():
    """SynapseOS – AI Operating System"""


@cli.command()
@click.option("--host", default=None, help="Bind host")
@click.option("--port", default=None, type=int, help="Bind port")
@click.option("--workers", default=None, type=int, help="Uvicorn worker count")
@click.option("--reload", is_flag=True, default=False, help="Enable hot-reload (dev mode)")
@click.option("--log-level", default=None, help="Log level")
def serve(
    host: str | None,
    port: int | None,
    workers: int | None,
    reload: bool,
    log_level: str | None,
) -> None:
    """Start the inference server."""
    cfg = get_settings()
    _host = host or cfg.api.host
    _port = port or cfg.api.port
    _workers = workers or cfg.api.workers
    _log_level = log_level or cfg.api.log_level

    configure_logging(level=_log_level)

    click.echo(f"Starting SynapseOS on {_host}:{_port}")
    click.echo(f"LLM backend: {cfg.llama.server_url}")
    click.echo(f"Embedding model: {cfg.embedding.model}")
    click.echo(f"Docs: http://{_host}:{_port}/docs")

    uvicorn.run(
        "orchestrator.main:app",
        host=_host,
        port=_port,
        workers=_workers if not reload else 1,
        reload=reload,
        log_level=_log_level,
        access_log=False,  # We handle logging via middleware
    )


@cli.command()
@click.argument("text")
def classify(text: str) -> None:
    """Quick intent classification test."""
    import asyncio
    from orchestrator.api.dependencies import Container

    async def _run():
        container = Container.get()
        intent = await container.intent_classifier.classify(text)
        click.echo(f"Text:       {text[:80]}")
        click.echo(f"Intent:     {intent.intent_type.value}")
        click.echo(f"Confidence: {intent.confidence:.3f}")
        click.echo(f"Expert:     {intent.requires_expert}")
        click.echo(f"RAG:        {intent.requires_rag}")

    asyncio.run(_run())


@cli.command()
@click.argument("file_path")
def ingest(file_path: str) -> None:
    """Ingest a document file into the RAG index."""
    import asyncio
    from pathlib import Path
    from orchestrator.api.dependencies import Container

    async def _run():
        path = Path(file_path)
        if not path.exists():
            click.echo(f"File not found: {file_path}", err=True)
            sys.exit(1)

        content = path.read_text(encoding="utf-8", errors="replace")
        container = Container.get()
        chunks = await container.rag_pipeline.ingest(
            content=content,
            source=path.name,
            metadata={"filename": path.name},
        )
        click.echo(f"Ingested: {path.name} → {chunks} chunks")

    asyncio.run(_run())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: start the server
        cfg = get_settings()
        uvicorn.run(
            "orchestrator.main:app",
            host=cfg.api.host,
            port=cfg.api.port,
            workers=cfg.api.workers,
            reload=False,
            log_level=cfg.api.log_level,
            access_log=False,
        )
    else:
        cli()
