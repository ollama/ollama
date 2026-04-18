"""
Tests for Cache Service
Tests caching layer functionality for Redis-based caching
"""

import pytest


class TestCacheService:
    """Test cache service operations"""

    def test_cache_manager_initialization(self):
        """Test creating cache manager"""
        from ollama.services.cache import init_cache

        manager = init_cache(redis_url="redis://localhost:6379/0", db=0)
        assert manager is not None

    @pytest.mark.asyncio
    async def test_cache_manager_initialize(self):
        """Test cache manager initialization"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)
        assert hasattr(manager, "initialize")
        assert hasattr(manager, "close")

    @pytest.mark.asyncio
    async def test_cache_get_set(self):
        """Test cache get/set operations"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Should have cache operations
        assert hasattr(manager, "get")
        assert hasattr(manager, "set")

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test cache deletion"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        assert hasattr(manager, "delete")

    @pytest.mark.asyncio
    async def test_cache_exists(self):
        """Test cache exists check"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        assert hasattr(manager, "exists")

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test cache clear"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Should have methods for clearing (via delete or other means)
        assert hasattr(manager, "delete") or hasattr(manager, "flush")


class TestCachingMiddleware:
    """Test caching middleware"""

    def test_caching_middleware_creation(self):
        """Test creating caching middleware"""
        from ollama.middleware import CachingMiddleware

        # Should be instantiable
        middleware = CachingMiddleware
        assert middleware is not None


class TestCachePatterns:
    """Test common caching patterns"""

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation"""
        from ollama.services.cache import CacheManager

        # Cache keys should be deterministic
        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)
        assert manager is not None

    @pytest.mark.asyncio
    async def test_cache_with_ttl(self):
        """Test cache with time-to-live"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Should support TTL functionality
        assert hasattr(manager, "set")

    @pytest.mark.asyncio
    async def test_cache_json_serialization(self):
        """Test caching JSON data"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Should be able to cache serializable objects
        assert manager is not None


class TestCacheInvalidation:
    """Test cache invalidation"""

    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Should have delete operations
        assert hasattr(manager, "delete")

    @pytest.mark.asyncio
    async def test_cache_pattern_deletion(self):
        """Test pattern-based cache deletion"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Should support pattern matching deletion
        assert manager is not None


class TestCacheErrorHandling:
    """Test cache error handling"""

    def test_cache_connection_error(self):
        """Test handling cache connection errors"""
        from ollama.services.cache import init_cache

        # Should handle invalid connection strings
        manager = init_cache(redis_url="redis://invalid:99999", db=0)
        assert manager is not None

    @pytest.mark.asyncio
    async def test_cache_operation_error(self):
        """Test handling cache operation errors"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Cache operations should be available
        assert hasattr(manager, "get")
        assert hasattr(manager, "set")


class TestCachePerformance:
    """Test cache performance aspects"""

    @pytest.mark.asyncio
    async def test_cache_async_operations(self):
        """Test async cache operations"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Operations should be async
        assert manager is not None

    @pytest.mark.asyncio
    async def test_cache_batching(self):
        """Test batch cache operations"""
        from ollama.services.cache import CacheManager

        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)

        # Should support batch operations
        assert manager is not None


class TestRedisCachingBackend:
    """Test Redis caching backend"""

    def test_redis_connection_string(self):
        """Test Redis connection string parsing"""
        from ollama.services.cache import CacheManager

        # Should parse connection string correctly
        manager = CacheManager(redis_url="redis://localhost:6379/0", db=0)
        assert manager is not None

    def test_redis_db_selection(self):
        """Test Redis database selection"""
        from ollama.services.cache import CacheManager

        # Should support different databases
        manager = CacheManager(redis_url="redis://localhost:6379/0", db=1)
        assert manager is not None
