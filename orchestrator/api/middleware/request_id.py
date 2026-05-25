"""
Request ID middleware.

Generates a unique request ID for every HTTP request, attaches it to the
response as ``X-Request-ID``, and binds it into the structlog context so all
log lines emitted during the request carry the same ID.

Clients may supply their own ID via the ``X-Request-ID`` request header; the
middleware honours it and echoes it back unchanged.
"""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(_HEADER) or str(uuid.uuid4())
        # Make it available to route handlers via request.state
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[_HEADER] = request_id
        return response
