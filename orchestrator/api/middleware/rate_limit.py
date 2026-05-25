"""
Token-bucket rate limiter middleware.
Keys on API key when auth is enabled, falls back to client IP.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


def _bearer(authorization: str) -> str:
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter.

    When API_AUTH_ENABLED=true the bucket key is the first 16 chars of the
    API key so each key gets its own independent quota.  Falls back to
    client IP when auth is disabled.
    """

    def __init__(self, app, requests_per_window: int = 60, window_seconds: int = 60) -> None:
        super().__init__(app)
        cfg = get_settings().api
        self._limit = requests_per_window or cfg.rate_limit_requests
        self._window = window_seconds or cfg.rate_limit_window
        self._auth_enabled = cfg.auth_enabled
        self._buckets: Dict[str, list] = defaultdict(list)

    def _bucket_key(self, request: Request) -> str:
        """Return the rate-limit bucket identifier for this request."""
        if self._auth_enabled:
            key = (
                request.headers.get("X-API-Key")
                or _bearer(request.headers.get("Authorization", ""))
            )
            if key:
                return f"key:{key[:16]}"
        return f"ip:{self._client_ip(request)}"

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in ("/health/live", "/health/ready", "/metrics"):
            return await call_next(request)

        bucket_key = self._bucket_key(request)
        now = time.time()
        window_start = now - self._window

        bucket = self._buckets[bucket_key]
        self._buckets[bucket_key] = [t for t in bucket if t > window_start]

        if len(self._buckets[bucket_key]) >= self._limit:
            log.warning("Rate limit exceeded", bucket=bucket_key)
            return Response(
                content='{"error": "rate_limit_exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(self._window)},
            )

        self._buckets[bucket_key].append(now)
        return await call_next(request)
