"""Caching decorators and utilities for FastAPI routes.

Provides decorators for easy integration of ResponseCache into
inference endpoints.

Example:
    >>> from ollama.api.cache_decorators import cached_inference
    >>>
    >>> @router.post("/generate")
    >>> @cached_inference(ttl=3600)
    >>> async def generate(
    ...     request: GenerateRequest,
    ...     cache: Annotated[CacheManager, Depends(get_cache_manager)],
    ... ) -> GenerateResponse:
    ...     return await ollama_client.generate(request)
"""

import hashlib
import json
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import structlog

log: Any = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def cached_inference(ttl: int = 3600) -> Callable[[F], F]:  # noqa: C901
    """Decorator for caching inference endpoint responses.

    Caches responses by model name and prompt hash with configurable TTL.
    Improves performance for repeated requests while maintaining freshness.

    This decorator requires the cache dependency to be injected by FastAPI.
    The decorated function must have a 'cache' parameter of type CacheManager.

    Args:
        ttl: Time to live in seconds (default: 3600 = 1 hour).

    Returns:
        Decorator function for async route handlers.

    Example:
        >>> @router.post("/generate")
        >>> @cached_inference(ttl=3600)
        >>> async def generate(
        ...     request: GenerateRequest,
        ...     cache: Annotated[CacheManager, Depends(get_cache_manager)],
        ... ) -> GenerateResponse:
        ...     return await ollama_client.generate(request)
    """

    def decorator(func: F) -> F:  # noqa: C901
        """Decorator implementation."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: C901
            """Wrapper for async inference functions with caching."""
            # Extract request and cache from arguments
            request: Any = None
            cache: Any = None

            # Try to find request and cache in args/kwargs
            for arg in args:
                if (hasattr(arg, "model") and hasattr(arg, "prompt")) or hasattr(arg, "text"):
                    request = arg
                    break

            if "cache" in kwargs:
                cache = kwargs["cache"]
            elif len(args) > 1:
                # Try to find cache in positional args
                for arg in args[1:]:
                    if hasattr(arg, "get") and hasattr(arg, "set"):
                        cache = arg
                        break

            # If we can't find request or cache, just call the function
            if not request or not cache:
                log.debug("cached_inference_fallback", reason="missing_request_or_cache")
                return await func(*args, **kwargs)

            # Generate cache key
            try:
                cache_key = _generate_cache_key(request)
                log.debug("cached_inference_attempting", cache_key=cache_key)

                # Try to get from cache
                cached_response = await cache.get(cache_key)
                if cached_response:
                    log.info(
                        "cached_inference_hit",
                        model=getattr(request, "model", "unknown"),
                        cache_key=cache_key,
                    )
                    return cached_response

                # Not in cache, call function
                log.debug("cached_inference_miss", cache_key=cache_key)
                result = await func(*args, **kwargs)

                # Cache the result if it's not a streaming response
                if result and not hasattr(result, "body_iterator"):
                    try:
                        await cache.set(cache_key, result, ttl=ttl)
                        log.debug("cached_inference_stored", cache_key=cache_key, ttl=ttl)
                    except Exception as e:
                        log.warning("cached_inference_store_failed", error=str(e))

                return result
            except Exception as e:
                log.warning(
                    "cached_inference_decorator_error",
                    error=str(e),
                    ttl=ttl,
                )
                # On error, just call the function directly
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def cache_by_model(ttl: int = 1800) -> Callable[[F], F]:  # noqa: C901
    """Decorator for caching responses by model only.

    Similar to @cached_inference but caches based on model name only,
    ignoring the specific prompt. Useful for endpoints where response
    is independent of input details.

    Args:
        ttl: Time to live in seconds (default: 1800 = 30 minutes).

    Returns:
        Decorator function for async route handlers.

    Example:
        >>> @router.post("/models/{model_name}/info")
        >>> @cache_by_model(ttl=3600)
        >>> async def get_model_info(
        ...     model_name: str,
        ...     cache: Annotated[CacheManager, Depends(get_cache_manager)],
        ... ) -> ModelInfo:
        ...     return await ollama_client.get_model_info(model_name)
    """

    def decorator(func: F) -> F:  # noqa: C901
        """Decorator implementation."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: C901
            """Wrapper for async functions with model-based caching."""
            # Extract request and cache from arguments
            cache: Any = None
            model_name: str | None = None

            # Try to find request in args/kwargs
            for arg in args:
                if hasattr(arg, "model"):
                    model_name = arg.model
                    break

            # Check kwargs for model_name
            if "model_name" in kwargs:
                model_name = kwargs["model_name"]

            if "cache" in kwargs:
                cache = kwargs["cache"]
            elif len(args) > 1:
                for arg in args[1:]:
                    if hasattr(arg, "get") and hasattr(arg, "set"):
                        cache = arg
                        break

            # If we can't find model or cache, just call the function
            if not model_name or not cache:
                log.debug("cache_by_model_fallback", reason="missing_model_or_cache")
                return await func(*args, **kwargs)

            # Generate cache key based on model only
            try:
                cache_key = f"inference:model:{model_name}"
                log.debug("cache_by_model_attempting", cache_key=cache_key)

                # Try to get from cache
                cached_response = await cache.get(cache_key)
                if cached_response:
                    log.info(
                        "cache_by_model_hit",
                        model=model_name,
                        cache_key=cache_key,
                    )
                    return cached_response

                # Not in cache, call function
                log.debug("cache_by_model_miss", cache_key=cache_key)
                result = await func(*args, **kwargs)

                # Cache the result
                if result:
                    try:
                        await cache.set(cache_key, result, ttl=ttl)
                        log.debug("cache_by_model_stored", cache_key=cache_key, ttl=ttl)
                    except Exception as e:
                        log.warning("cache_by_model_store_failed", error=str(e))

                return result
            except Exception as e:
                log.warning(
                    "cache_by_model_decorator_error",
                    error=str(e),
                    ttl=ttl,
                )
                # On error, just call the function directly
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def _generate_cache_key(request: Any) -> str:
    """Generate SHA256-based cache key for request.

    Args:
        request: Request object with model and prompt attributes.

    Returns:
        SHA256-based cache key for the request.
    """
    # Build payload for hashing
    payload = {
        "model": getattr(request, "model", ""),
        "prompt": getattr(request, "prompt", getattr(request, "text", "")),
    }

    # Add optional parameters if present
    for attr in ["system", "temperature", "top_p", "top_k", "num_predict"]:
        if hasattr(request, attr):
            payload[attr] = getattr(request, attr)

    dump = json.dumps(payload, sort_keys=True)
    h = hashlib.sha256(dump.encode()).hexdigest()
    return f"inference:v1:gen:{payload['model']}:{h}"


__all__ = ["cache_by_model", "cached_inference"]
