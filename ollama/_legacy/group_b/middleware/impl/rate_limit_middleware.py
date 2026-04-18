"""FastAPI middleware for applying rate limiting to requests."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from ollama.middleware.impl.rate_limiter import RateLimiter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting.

    Applies rate limits based on:
    - User ID (for authenticated requests)
    - IP address (for unauthenticated requests)
    - API endpoint
    """

    def __init__(
        self,
        app: Any,
        requests_per_minute: int = 60,
        burst_size: int | None = None,
        exclude_paths: list[str] | None = None,
    ) -> None:
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Global rate limit
            burst_size: Maximum burst size
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute, burst_size)
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Determine rate limit key
        # Priority: user_id > api_key > ip_address
        rate_limit_key = None

        # Check for authenticated user (set by auth middleware)
        if hasattr(request.state, "user"):
            rate_limit_key = f"user:{request.state.user.id}"
        else:
            # Fall back to IP address
            client_ip = request.client.host if request.client else "unknown"
            rate_limit_key = f"ip:{client_ip}"

        # Check rate limit
        allowed, limit_info = self.limiter.check_rate_limit(rate_limit_key)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {rate_limit_key} on {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(limit_info["reset"]),
                    "Retry-After": "60",
                },
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])

        return response
