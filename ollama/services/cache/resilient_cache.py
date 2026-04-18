"""Resilient Cache Manager with circuit breaker pattern.

Adds fault tolerance to Redis operations by wrapping the cache manager
with circuit breaker logic. This prevents the application from hanging
or overloading a struggling Redis instance.

Example:
    >>> from ollama.services.cache.cache import CacheManager
    >>> from ollama.services.cache.resilient_cache import ResilientCacheManager
    >>> cache_manager = CacheManager()
    >>> resilient_cache = ResilientCacheManager(cache_manager)
    >>> value = await resilient_cache.get("my_key")
"""

from typing import Any, cast

import structlog

from ollama.exceptions.circuit_breaker import CircuitBreakerError
from ollama.services.cache.cache import CacheManager
from ollama.services.resilience.circuit_breaker import (
    CircuitBreaker,
    get_circuit_breaker_manager,
)

log = structlog.get_logger(__name__)


class ResilientCacheManager:
    """Resilient wrapper for CacheManager with circuit breaker pattern.

    Detects Redis connection failures and temporarily opens the circuit to
    avoid latent calls and allow Redis to recover.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        service_name: str = "redis-cache",
    ) -> None:
        """Initialize resilient cache manager.

        Args:
            cache_manager: The underlying CacheManager instance.
            failure_threshold: Failures before opening circuit (default: 5).
            recovery_timeout: Seconds before recovery attempt (default: 30).
            service_name: Name for circuit breaker tracking (default: redis-cache).
        """
        self.cache = cache_manager
        self._cb_manager = get_circuit_breaker_manager()
        self.service_name = service_name

        self.breaker: CircuitBreaker = self._cb_manager.get_or_create(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    async def get(self, key: str) -> Any | None:
        """Get value from cache with circuit breaker protection.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found, expired, or circuit is open.
        """
        try:
            # We wrap the internal method. If it returns None due to error,
            # we might want to record failure manually if we can distinguish it.
            # For now, we trust the breaker to catch exceptions from the client if they surface.
            return await self.breaker.call_async(self.cache.get, key)
        except CircuitBreakerError:
            log.warning("cache_breaker_open", operation="get", key=key)
            return None
        except Exception as e:
            # The breaker already recorded this failure if it surfaced.
            log.error("cache_get_error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with circuit breaker protection.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time to live in seconds.

        Returns:
            True if successful, False otherwise.
        """
        try:
            return cast(bool, await self.breaker.call_async(self.cache.set, key, value, ttl))
        except CircuitBreakerError:
            log.warning("cache_breaker_open", operation="set", key=key)
            return False
        except Exception as e:
            log.error("cache_set_error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache with circuit breaker protection.

        Args:
            key: Cache key.

        Returns:
            True if successful, False otherwise.
        """
        try:
            return cast(bool, await self.breaker.call_async(self.cache.delete, key))
        except CircuitBreakerError:
            log.warning("cache_breaker_open", operation="delete", key=key)
            return False
        except Exception as e:
            log.error("cache_delete_error", key=key, error=str(e))
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter with circuit breaker protection.

        Args:
            key: Cache key.
            amount: Amount to increment by.

        Returns:
            Updated counter value or 0 if failed.
        """
        try:
            return cast(int, await self.breaker.call_async(self.cache.increment, key, amount))
        except CircuitBreakerError:
            log.warning("cache_breaker_open", operation="increment", key=key)
            return 0
        except Exception as e:
            log.error("cache_increment_error", key=key, error=str(e))
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists with circuit breaker protection.

        Args:
            key: Cache key.

        Returns:
            True if key exists, False otherwise.
        """
        try:
            return cast(bool, await self.breaker.call_async(self.cache.exists, key))
        except CircuitBreakerError:
            log.warning("cache_breaker_open", operation="exists", key=key)
            return False
        except Exception as e:
            log.error("cache_exists_error", key=key, error=str(e))
            return False

    async def initialize(self) -> None:
        """Initialize underlying cache client."""
        await self.cache.initialize()

    async def close(self) -> None:
        """Close underlying cache client."""
        await self.cache.close()
