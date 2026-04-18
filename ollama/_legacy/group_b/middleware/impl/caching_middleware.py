"""Middleware for caching GET requests and safe endpoints."""

import logging
from typing import Any, ClassVar, cast

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from ollama.services import CacheManager

logger = logging.getLogger(__name__)


class CachingMiddleware(BaseHTTPMiddleware):
    """Middleware for caching GET requests and safe endpoints."""

    CACHEABLE_ENDPOINTS: ClassVar[dict[str, int]] = {
        "/health": 3600,  # 1 hour
        "/api/v1/models": 3600,  # 1 hour
        "/api/v1/models/": 3600,  # Model details
        "/metrics": 60,  # 1 minute
    }

    def __init__(self, app: ASGIApp, cache_manager: CacheManager) -> None:
        super().__init__(app)
        self.cache_manager = cache_manager

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Cache GET requests to specific endpoints."""

        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)

        # Check if endpoint is cacheable
        path = request.url.path
        ttl = None
        for endpoint, endpoint_ttl in self.CACHEABLE_ENDPOINTS.items():
            if path.startswith(endpoint):
                ttl = endpoint_ttl
                break

        if ttl is None:
            return await call_next(request)

        # Generate cache key
        cache_key = f"http:{request.method}:{path}"
        if request.url.query:
            cache_key += f"?{request.url.query}"

        # Try to get from cache
        cached = await self.cache_manager.get(cache_key)
        if cached:
            logger.debug(f"Cache HIT: {cache_key}")
            return Response(
                content=cached, media_type="application/json", headers={"X-Cache": "HIT"}
            )

        # Call endpoint
        response = await call_next(request)

        # Cache if successful
        if response.status_code == 200:
            try:
                # Read response body
                body = b""
                # BaseHTTPMiddleware's response is often a StreamingResponse
                # which has body_iterator. We cast to Any to avoid type errors.
                response_any = cast(Any, response)
                async for chunk in response_any.body_iterator:
                    body += chunk

                # Store in cache
                await self.cache_manager.set(cache_key, body.decode(), ttl=ttl)
                logger.debug(f"Cache SET: {cache_key} (TTL: {ttl}s)")

                # Return response with cache header
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers={**dict(response.headers), "X-Cache": "MISS"},
                    media_type=response.media_type,
                )
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
                return response

        return response
