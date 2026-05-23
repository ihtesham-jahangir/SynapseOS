"""
API key authentication middleware.

Validates the ``X-API-Key`` header (or ``Authorization: Bearer <key>``)
against the configured key.  Skips public endpoints so health checks,
metrics, and docs remain accessible without credentials.

Enable in .env:
  API_AUTH_ENABLED=true
  API_KEY=your-secret-key-here
"""
from __future__ import annotations

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

# Paths that bypass authentication entirely
_PUBLIC_PATHS: frozenset = frozenset(
    [
        "/",
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    ]
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Request-level API key gate.

    Reads the key from ``X-API-Key`` header first, then falls back to
    ``Authorization: Bearer <key>``.  Returns 401 when no key is sent,
    403 when the key is wrong.
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        cfg = get_settings().api
        self._enabled: bool = cfg.auth_enabled
        self._key: str = cfg.key
        if self._enabled:
            log.info("API key authentication enabled")
        else:
            log.info("API key authentication disabled (set API_AUTH_ENABLED=true to enable)")

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            return await call_next(request)

        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Extract key from header
        provided = (
            request.headers.get("X-API-Key")
            or _bearer(request.headers.get("Authorization", ""))
        )

        if not provided:
            log.warning(
                "Unauthenticated request",
                path=request.url.path,
                client=_client_ip(request),
            )
            return JSONResponse(
                {"error": "missing_api_key", "hint": "Set X-API-Key header"},
                status_code=401,
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if provided != self._key:
            log.warning(
                "Invalid API key",
                path=request.url.path,
                client=_client_ip(request),
            )
            return JSONResponse(
                {"error": "invalid_api_key"},
                status_code=403,
            )

        return await call_next(request)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bearer(authorization: str) -> str:
    """Extract token from 'Bearer <token>' header, or return empty string."""
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
