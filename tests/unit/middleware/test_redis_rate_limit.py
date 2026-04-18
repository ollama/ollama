"""Tests for Redis rate limiting."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ollama.middleware import RedisRateLimiter


class TestRedisRateLimiter:
    """Tests for RedisRateLimiter class."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Create mock Redis client."""
        return MagicMock()

    @pytest.fixture
    def rate_limiter(self, mock_redis: MagicMock) -> RedisRateLimiter:
        """Create RedisRateLimiter instance."""
        return RedisRateLimiter(mock_redis, requests_per_minute=60)

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter: RedisRateLimiter) -> None:
        """Test that request is allowed within rate limit."""
        # Mock pipeline that returns request count < limit
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [15, None, 45000]  # 15 requests, 45s remaining
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        allowed, limit_info = await rate_limiter.check_rate_limit("user:123")

        assert allowed is True
        assert limit_info["remaining"] == 45
        assert limit_info["limit"] == 60
        assert "reset" in limit_info

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter: RedisRateLimiter) -> None:
        """Test that request is denied when rate limit exceeded."""
        # Mock pipeline that returns request count > limit
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [61, None, 30000]  # 61 requests (exceeded)
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        allowed, limit_info = await rate_limiter.check_rate_limit("user:456")

        assert allowed is False
        assert limit_info["remaining"] == 0
        assert limit_info["limit"] == 60

    @pytest.mark.asyncio
    async def test_check_rate_limit_at_boundary(self, rate_limiter: RedisRateLimiter) -> None:
        """Test request exactly at rate limit boundary."""
        # Mock pipeline that returns request count == limit
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [60, None, 15000]
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        allowed, limit_info = await rate_limiter.check_rate_limit("user:789")

        assert allowed is True
        assert limit_info["remaining"] == 0

    @pytest.mark.asyncio
    async def test_check_rate_limit_different_keys(self, rate_limiter: RedisRateLimiter) -> None:
        """Test that different keys have independent rate limits."""
        mock_pipeline = MagicMock()

        # User 1: 30 requests
        mock_pipeline.execute.side_effect = [
            [30, None, 45000],  # First call
            [25, None, 50000],  # Second call
        ]
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        # Check user 1
        allowed1, info1 = await rate_limiter.check_rate_limit("user:1")
        assert allowed1 is True
        assert info1["remaining"] == 30

        # Check user 2 (should have independent count)
        allowed2, info2 = await rate_limiter.check_rate_limit("user:2")
        assert allowed2 is True
        assert info2["remaining"] == 35  # Different count

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_pttl(self, rate_limiter: RedisRateLimiter) -> None:
        """Test handling when reset_pttl is None."""
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [10, None, None]  # None pttl
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        allowed, limit_info = await rate_limiter.check_rate_limit("user:123")

        assert allowed is True
        assert "reset" in limit_info
        # Reset should be approximately 60 seconds from now
        assert limit_info["reset"] > time.time()

    @pytest.mark.asyncio
    async def test_check_rate_limit_redis_error(self, rate_limiter: RedisRateLimiter) -> None:
        """Test graceful handling of Redis errors (fail open)."""
        # Mock pipeline that raises exception
        mock_pipeline = MagicMock()
        mock_pipeline.execute.side_effect = Exception("Redis connection error")
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        allowed, limit_info = await rate_limiter.check_rate_limit("user:error")

        # Should fail open (allow request) when Redis unavailable
        assert allowed is True
        assert limit_info["limit"] == 60
        assert limit_info["remaining"] == 60

    @pytest.mark.asyncio
    async def test_check_rate_limit_pipeline_operations(
        self, rate_limiter: RedisRateLimiter
    ) -> None:
        """Test that pipeline operations are called correctly."""
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [5, None, 55000]
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        await rate_limiter.check_rate_limit("test:key")

        # Verify pipeline methods were called
        mock_pipeline.incr.assert_called_once_with("ratelimit:window:test:key")
        mock_pipeline.pexpire.assert_called_once_with("ratelimit:window:test:key", 60000)
        mock_pipeline.pttl.assert_called_once_with("ratelimit:reset:test:key")
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_rate_limit_custom_requests_per_minute(self, mock_redis: MagicMock) -> None:
        """Test rate limiter with custom requests per minute."""
        custom_limiter = RedisRateLimiter(mock_redis, requests_per_minute=100)

        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [50, None, 30000]
        mock_redis.pipeline.return_value = mock_pipeline

        allowed, limit_info = await custom_limiter.check_rate_limit("user:custom")

        assert allowed is True
        assert limit_info["limit"] == 100
        assert limit_info["remaining"] == 50

    @pytest.mark.asyncio
    async def test_check_rate_limit_reset_timestamp_accuracy(
        self, rate_limiter: RedisRateLimiter
    ) -> None:
        """Test that reset timestamp is calculated accurately."""
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [30, None, 25000]  # 25 seconds remaining
        rate_limiter.redis.pipeline.return_value = mock_pipeline

        before_time = time.time()
        allowed, limit_info = await rate_limiter.check_rate_limit("user:123")
        after_time = time.time()

        # Reset should be approximately current_time + 25 seconds
        assert limit_info["reset"] >= int(before_time) + 25
        assert limit_info["reset"] <= int(after_time) + 26


class TestRedisRateLimiterIntegration:
    """Integration tests for Redis rate limiting (requires Redis)."""

    @pytest.fixture
    def real_redis(self):
        """Get real Redis client (only if available)."""
        try:
            import redis

            client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            client.ping()  # Test connection
            return client
        except Exception:
            pytest.skip("Redis not available")

    @pytest.fixture(autouse=True)
    def cleanup_redis(self, real_redis):
        """Clean up Redis test keys after each test."""
        yield
        # Delete test keys
        for key in real_redis.keys("ratelimit:*:test:*"):
            real_redis.delete(key)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_redis_rate_limiting_end_to_end(self, real_redis) -> None:
        """Test end-to-end rate limiting with real Redis."""
        limiter = RedisRateLimiter(real_redis, requests_per_minute=5)

        # Make 5 requests (should all be allowed)
        for i in range(5):
            allowed, info = await limiter.check_rate_limit("test:e2e:123")
            assert allowed is True

        # 6th request should be denied
        allowed, info = await limiter.check_rate_limit("test:e2e:123")
        assert allowed is False
        assert info["remaining"] == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_redis_rate_limiting_isolation(self, real_redis) -> None:
        """Test that different keys have isolated rate limits."""
        limiter = RedisRateLimiter(real_redis, requests_per_minute=3)

        # User 1: 3 requests
        for _ in range(3):
            allowed, _ = await limiter.check_rate_limit("test:user:1")
            assert allowed is True

        # User 1: 4th request denied
        allowed, _ = await limiter.check_rate_limit("test:user:1")
        assert allowed is False

        # User 2: Should still allow requests (independent limit)
        for _ in range(3):
            allowed, _ = await limiter.check_rate_limit("test:user:2")
            assert allowed is True

        allowed, _ = await limiter.check_rate_limit("test:user:2")
        assert allowed is False
