"""
Structured logging setup using structlog.
Every module gets a pre-configured logger via get_logger().
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_configured = False


def configure_logging(level: str = "info", json_output: bool = False) -> None:
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        shared_processors.append(structlog.processors.JSONRenderer())
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
        handler = RichHandler(
            console=_console,
            show_time=False,
            show_path=False,
            markup=True,
        )

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        format="%(message)s",
    )

    # Quiet noisy libs
    for noisy in ("httpx", "httpcore", "faiss", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> structlog.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)


def bind_request_context(session_id: str, request_id: Optional[str] = None) -> None:
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        **({"request_id": request_id} if request_id else {}),
    )


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
