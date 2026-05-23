"""Structured request/response logging middleware."""
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator.utils.logging_utils import get_logger, bind_request_context, clear_request_context

log = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        bind_request_context(
            session_id=request.headers.get("X-Session-ID", "none"),
            request_id=request_id,
        )

        t0 = time.perf_counter()
        log.info(
            "→ Request",
            method=request.method,
            path=request.url.path,
            request_id=request_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            log.error(
                "Unhandled exception",
                error=str(exc),
                request_id=request_id,
                exc_info=True,
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "← Response",
                method=request.method,
                path=request.url.path,
                status=response.status_code if "response" in dir() else 500,
                elapsed_ms=f"{elapsed_ms:.1f}",
                request_id=request_id,
            )
            clear_request_context()

        return response
