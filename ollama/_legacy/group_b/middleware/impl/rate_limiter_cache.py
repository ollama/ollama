"""Rate limiting using Redis counters for middleware."""

import logging

from ollama.services import CacheManager

logger = logging.getLogger(__name__)


class RateLimiterCache:
    """Rate limiting using Redis counters."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    async def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is within rate limit."""
        counter_key = f"rate:{key}"

        count = await self.cache_manager.increment(counter_key, 1)

        # Set expiration on first increment
        if count == 1:
            await self.cache_manager.expire(counter_key, window)

        return count <= limit

    async def get_remaining(self, key: str, limit: int) -> int:
        """Get remaining requests in current window."""
        counter_key = f"rate:{key}"
        # We use raw get if we had one, but get() handles numbers as well
        current = await self.cache_manager.get(counter_key)
        count = int(current) if current is not None else 0
        return max(0, limit - count)
