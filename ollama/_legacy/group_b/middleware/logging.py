"""Request and response logging middleware."""

import time
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log: Any = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log completion.

        Args:
            request: The incoming request.
            call_next: Next middleware or route handler.

        Returns:
            The response.
        """
        start_time = time.perf_counter()

        # Log request
        log.info(
            "http_request_start",
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception as e:
            log.error(
                "http_request_error",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            # Re-raise to let error handlers handle it
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log response
        log.info(
            "http_request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response
