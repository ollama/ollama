"""
Cache Service - Redis-based caching layer
Provides distributed caching for multi-instance deployments
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Global cache manager instance
_cache_manager: CacheManager | None = None


class CacheManager:
    """Redis-based cache manager for distributed caching"""

    def __init__(self, redis_url: str = "redis://redis:6379/0", db: int = 0) -> None:
        """Initialize cache manager

        Args:
            redis_url: Redis connection URL (use 'redis' Docker service name, NOT localhost)
            db: Database number to use
        """
        self.redis_url = redis_url
        self.db = db
        self.client: aioredis.Redis | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection"""
        try:
            # from_url is a sync call in redis-py 4.0+
            self.client = aioredis.from_url(self.redis_url, db=self.db)  # type: ignore[no-untyped-call]
            if self.client is not None:
                await cast(Any, self.client.ping())
            logger.info(f"Cache manager initialized: {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("Cache manager closed")

    async def get(self, key: str) -> Any | None:
        """Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.client:
            return None

        try:
            value = await self.client.get(key)
            if value:
                return json.loads(cast(str, value))
            return None
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            raise

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            serialized = json.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """Delete value from cache

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            raise

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            Updated counter value
        """
        if not self.client:
            return 0

        try:
            return int(await self.client.incrby(key, amount))
        except Exception as e:
            logger.warning(f"Cache increment failed for key {key}: {e}")
            raise

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key.

        Args:
            key: Cache key
            seconds: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            return bool(await self.client.expire(key, seconds))
        except Exception as e:
            logger.warning(f"Cache expire failed for key {key}: {e}")
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self.client:
            return False

        try:
            result = await self.client.exists(key)
            return bool(result)
        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            raise

    async def clear(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern

        Args:
            pattern: Key pattern to match (default: clear all)

        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0

        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern)
                if keys:
                    deleted += await self.client.delete(*keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as e:
            logger.warning(f"Cache clear failed for pattern {pattern}: {e}")
            raise


def init_cache(redis_url: str = "redis://redis:6379/0", db: int = 0) -> CacheManager:
    """Initialize cache manager

    Args:
        redis_url: Redis connection URL (use 'redis' Docker service name, NOT localhost)
        db: Database number to use

    Returns:
        Initialized CacheManager instance
    """
    global _cache_manager
    _cache_manager = CacheManager(redis_url=redis_url, db=db)
    return _cache_manager


def get_cache_manager() -> CacheManager | None:
    """Get global cache manager instance

    Returns:
        Global CacheManager or None if not initialized
    """
    return _cache_manager
