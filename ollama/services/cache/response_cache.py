"""Response Caching Service for inference endpoints.

Provides caching of inference responses with TTL to reduce model inference
latency for repeated or similar requests. Uses Redis backend.

Example:
    >>> cache = ResponseCache(cache_manager, ttl=3600)
    >>> # Cache inference response
    >>> await cache.set_response(model, prompt, response, ttl=3600)
    >>> # Retrieve cached response
    >>> cached = await cache.get_response(model, prompt)
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

from ollama.services.cache.cache import CacheManager

log = structlog.get_logger(__name__)


class ResponseCache:
    """Response cache for inference endpoints with TTL support."""

    # Cache key prefix for inference responses
    RESPONSE_PREFIX = "inference:response:"
    # Cache key prefix for request metadata
    METADATA_PREFIX = "inference:metadata:"

    def __init__(self, cache_manager: CacheManager, default_ttl: int = 3600) -> None:
        """Initialize response cache.

        Args:
            cache_manager: Redis cache manager instance.
            default_ttl: Default time-to-live in seconds (default: 1 hour).
        """
        self.cache = cache_manager
        self.default_ttl = default_ttl

    def _get_cache_key(
        self,
        model: str,
        prompt: str,
    ) -> str:
        """Generate cache key for prompt and model.

        Uses SHA256 hash of model + prompt for consistent, short keys.

        Args:
            model: Model name/identifier.
            prompt: User prompt.

        Returns:
            Redis cache key.
        """
        key_data = f"{model}:{prompt}".encode()
        key_hash = hashlib.sha256(key_data).hexdigest()
        return f"{self.RESPONSE_PREFIX}{model}:{key_hash}"

    def _get_metadata_key(self, cache_key: str) -> str:
        """Generate metadata key for cache entry.

        Args:
            cache_key: Main cache key.

        Returns:
            Metadata cache key.
        """
        return f"{self.METADATA_PREFIX}{cache_key}"

    async def get_response(
        self,
        model: str,
        prompt: str,
    ) -> dict[str, Any] | None:
        """Retrieve cached inference response.

        Args:
            model: Model name that generated the response.
            prompt: Original user prompt.

        Returns:
            Cached response dict or None if not found/expired.
        """
        cache_key = self._get_cache_key(model, prompt)

        try:
            cached = await self.cache.get(cache_key)
            if cached:
                log.info(
                    "response_cache_hit",
                    model=model,
                    prompt_len=len(prompt),
                )
                # Type cast - cache.get returns Any, we trust it's dict[str, Any]
                return cached  # type: ignore[no-any-return]
            return None
        except Exception as exc:
            log.warning(
                "response_cache_get_error",
                model=model,
                error=str(exc),
            )
            return None

    async def set_response(
        self,
        model: str,
        prompt: str,
        response: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache inference response with TTL.

        Args:
            model: Model name that generated the response.
            prompt: Original user prompt.
            response: Response dict to cache.
            ttl: Time-to-live in seconds (default: default_ttl).

        Returns:
            True if caching succeeded, False otherwise.
        """
        cache_key = self._get_cache_key(model, prompt)
        ttl = ttl or self.default_ttl

        try:
            success = await self.cache.set(cache_key, response, ttl=ttl)
            if success:
                log.info(
                    "response_cache_set",
                    model=model,
                    prompt_len=len(prompt),
                    ttl=ttl,
                )
            return success
        except Exception as exc:
            log.warning(
                "response_cache_set_error",
                model=model,
                error=str(exc),
            )
            return False

    async def delete_response(
        self,
        model: str,
        prompt: str,
    ) -> bool:
        """Delete cached inference response.

        Args:
            model: Model name.
            prompt: Original user prompt.

        Returns:
            True if deletion succeeded.
        """
        cache_key = self._get_cache_key(model, prompt)

        try:
            return await self.cache.delete(cache_key)
        except Exception as exc:
            log.warning(
                "response_cache_delete_error",
                model=model,
                error=str(exc),
            )
            return False

    async def clear_model_cache(self, model: str) -> int:
        """Clear all cached responses for a specific model.

        Args:
            model: Model name to clear cache for.

        Returns:
            Number of cache entries cleared.
        """
        try:
            pattern = f"{self.RESPONSE_PREFIX}{model}:*"
            if not self.cache.client:
                return 0
            keys = await self.cache.client.keys(pattern)
            if keys:
                await self.cache.client.delete(*keys)
            log.info("response_cache_cleared", model=model, count=len(keys))
            return len(keys)
        except Exception as exc:
            log.warning("response_cache_clear_error", model=model, error=str(exc))
            return 0

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics.

        Returns:
            Dict with cache metrics and configuration.
        """
        return {
            "service": "response_cache",
            "default_ttl": self.default_ttl,
            "prefix": self.RESPONSE_PREFIX,
        }
