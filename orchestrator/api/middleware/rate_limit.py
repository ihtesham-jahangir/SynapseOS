"""
Token-bucket rate limiter middleware.
Limits requests per IP address per rolling time window.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter.
    Tracks (client_ip → [(timestamp,)]) and rejects if count exceeds limit.
    """

    def __init__(self, app, requests_per_window: int = 60, window_seconds: int = 60) -> None:
        super().__init__(app)
        cfg = get_settings().api
        self._limit = requests_per_window or cfg.rate_limit_requests
        self._window = window_seconds or cfg.rate_limit_window
        self._buckets: Dict[str, list] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health/live", "/health/ready", "/metrics"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()
        window_start = now - self._window

        # Prune old timestamps
        bucket = self._buckets[client_ip]
        self._buckets[client_ip] = [t for t in bucket if t > window_start]

        if len(self._buckets[client_ip]) >= self._limit:
            log.warning("Rate limit exceeded", client=client_ip)
            return Response(
                content='{"error": "rate_limit_exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(self._window)},
            )

        self._buckets[client_ip].append(now)
        return await call_next(request)
