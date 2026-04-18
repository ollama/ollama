"""Decorator for caching endpoint responses."""

import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from ollama.services import CacheManager

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def cache_response(
    key: str, ttl: int = 3600, cache_manager: CacheManager | None = None
) -> Callable[[F], F]:
    """Decorator for caching endpoint responses.

    Usage:
        @app.get("/api/endpoint")
        @cache_response(key="endpoint_cache", ttl=1800)
        async def my_endpoint():
            return {"data": "value"}
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if cache_manager is None:
                return await func(*args, **kwargs)

            # Try cache
            cached = await cache_manager.get(key)
            if cached:
                logger.debug(f"Cache HIT: {key}")
                if isinstance(cached, str):
                    return json.loads(cached)
                return cached

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                await cache_manager.set(key, result, ttl=ttl)
                logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"Failed to cache: {e}")

            return result

        return cast(F, wrapper)

    return decorator
