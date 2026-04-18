"""Cache manager dependency helpers."""

import logging

from ollama.services.cache.resilient_cache import ResilientCacheManager
from ollama.services.persistence.cache import CacheManager

log = logging.getLogger(__name__)

# Global cache manager instance
_cache_manager: CacheManager | ResilientCacheManager | None = None


async def get_cache_manager() -> CacheManager | ResilientCacheManager:
    """FastAPI dependency that yields the cache manager."""
    global _cache_manager
    if _cache_manager is None:
        # Fallback to creating a new instance if not initialized
        # (Though it should be initialized in main.py)
        log.warning("Cache manager accessed before lifecycle initialization. Initializing now...")
        from ollama.config import get_settings

        settings = get_settings()
        _cache_manager = CacheManager(redis_url=settings.redis.url)
        await _cache_manager.initialize()

    return _cache_manager


def set_global_cache_manager(manager: CacheManager | ResilientCacheManager) -> None:
    """Set the global cache manager instance (called from main.py startup)."""
    global _cache_manager
    _cache_manager = manager
    log.info("Global cache manager dependency initialized")
