"""Track cache performance statistics."""

import logging
from typing import Any

from ollama.services import CacheManager

logger = logging.getLogger(__name__)


class CacheStats:
    """Track cache performance statistics."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    async def record_hit(self, key: str) -> None:
        """Record cache hit."""
        stats_key = f"stats:hits:{key}"
        await self.cache_manager.increment(stats_key, 1)
        await self.cache_manager.expire(stats_key, 86400)  # 24h

    async def record_miss(self, key: str) -> None:
        """Record cache miss."""
        stats_key = f"stats:misses:{key}"
        await self.cache_manager.increment(stats_key, 1)
        await self.cache_manager.expire(stats_key, 86400)  # 24h

    async def get_stats(self, key: str) -> dict[str, Any]:
        """Get cache statistics for a key."""
        hits_key = f"stats:hits:{key}"
        misses_key = f"stats:misses:{key}"

        hits = await self.cache_manager.get(hits_key)
        misses = await self.cache_manager.get(misses_key)

        hits_val = int(hits) if hits is not None else 0
        misses_val = int(misses) if misses is not None else 0
        total = hits_val + misses_val

        hit_rate = (hits_val / total * 100) if total > 0 else 0

        return {
            "hits": hits,
            "misses": misses,
            "total": total,
            "hit_rate": f"{hit_rate:.2f}%",
        }
