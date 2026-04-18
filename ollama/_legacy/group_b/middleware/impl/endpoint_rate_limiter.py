"""Decorator for endpoint-specific rate limiting."""

import logging

from fastapi import HTTPException, Request, status
from ollama.middleware.impl.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class EndpointRateLimiter:
    """Decorator for endpoint-specific rate limiting.

    Usage:
        @router.get("/expensive-endpoint")
        @EndpointRateLimiter(requests_per_minute=10)
        async def expensive_endpoint():
            ...
    """

    def __init__(self, requests_per_minute: int, burst_size: int | None = None):
        """Initialize endpoint rate limiter.

        Args:
            requests_per_minute: Rate limit for this endpoint
            burst_size: Maximum burst size
        """
        self.limiter = RateLimiter(requests_per_minute, burst_size)

    async def __call__(self, request: Request) -> None:
        """Check rate limit for endpoint.

        Args:
            request: Incoming request

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Determine rate limit key
        if hasattr(request.state, "user"):
            key = f"endpoint:{request.url.path}:user:{request.state.user.id}"
        else:
            client_ip = request.client.host if request.client else "unknown"
            key = f"endpoint:{request.url.path}:ip:{client_ip}"

        allowed, limit_info = self.limiter.check_rate_limit(key)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for endpoint {request.url.path}",
                headers={
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(limit_info["reset"]),
                    "Retry-After": "60",
                },
            )
