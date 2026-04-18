"""
Tests for Rate Limiting Middleware
Tests token bucket algorithm and rate limit enforcement
"""

import time
from unittest.mock import Mock

import pytest

from ollama.middleware import RateLimiter, RateLimitMiddleware


class TestRateLimiter:
    """Test RateLimiter class"""

    @pytest.fixture
    def limiter(self):
        """Create RateLimiter instance"""
        return RateLimiter(requests_per_minute=60, burst_size=100)

    def test_initialization(self, limiter):
        """Test RateLimiter initialization"""
        assert limiter.requests_per_minute == 60
        assert limiter.burst_size == 100
        assert limiter.tokens_per_second == 1.0  # 60/60

    def test_initial_rate_limit_allowed(self, limiter):
        """Test initial requests are allowed"""
        key = "test-user"
        allowed, info = limiter.check_rate_limit(key)

        assert allowed is True
        assert info["limit"] == 60
        # Tokens shown are before decrement (100 initially, then -1 to allow this request)
        assert info["remaining"] == 100

    def test_rate_limit_respects_burst(self, limiter):
        """Test rate limit respects burst size"""
        key = "test-user"

        # Use up all burst tokens
        for _ in range(100):
            allowed, _ = limiter.check_rate_limit(key)
            assert allowed is True

        # Next request should be denied
        allowed, info = limiter.check_rate_limit(key)
        assert allowed is False
        assert info["remaining"] == 0

    def test_tokens_refill_over_time(self, limiter):
        """Test tokens refill over time"""
        key = "test-user"

        # Use up burst
        for _ in range(100):
            limiter.check_rate_limit(key)

        # Verify rate limit exceeded
        allowed, _ = limiter.check_rate_limit(key)
        assert allowed is False

        # Wait for tokens to refill
        time.sleep(2)  # Should refill ~2 tokens

        # Should allow more requests now
        allowed, _ = limiter.check_rate_limit(key)
        assert allowed is True

    def test_different_keys_independent(self, limiter):
        """Test different keys have independent limits"""
        key1 = "user1"
        key2 = "user2"

        # Use up burst for key1
        for _ in range(100):
            limiter.check_rate_limit(key1)

        # key1 should be limited
        allowed1, _ = limiter.check_rate_limit(key1)
        assert allowed1 is False

        # key2 should still work
        allowed2, _ = limiter.check_rate_limit(key2)
        assert allowed2 is True

    def test_reset_clears_limit(self, limiter):
        """Test reset clears rate limit"""
        key = "test-user"

        # Use up burst
        for _ in range(100):
            limiter.check_rate_limit(key)

        # Verify limited
        allowed, _ = limiter.check_rate_limit(key)
        assert allowed is False

        # Reset
        limiter.reset(key)

        # Should work again
        allowed, _ = limiter.check_rate_limit(key)
        assert allowed is True

    def test_limit_info_structure(self, limiter):
        """Test limit info has correct structure"""
        key = "test-user"
        _, info = limiter.check_rate_limit(key)

        assert "limit" in info
        assert "remaining" in info
        assert "reset" in info
        assert info["limit"] == 60
        assert isinstance(info["remaining"], int)
        assert isinstance(info["reset"], int)


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware"""

    @pytest.fixture
    def middleware(self):
        """Create RateLimitMiddleware instance"""
        app = Mock()
        return RateLimitMiddleware(
            app, requests_per_minute=10, burst_size=20, exclude_paths=["/health"]
        )

    def test_middleware_initialization(self, middleware):
        """Test RateLimitMiddleware initializes correctly"""
        assert middleware.limiter is not None
        # When exclude_paths is provided, it overrides defaults
        assert "/health" in middleware.exclude_paths
