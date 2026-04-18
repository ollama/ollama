"""Tests for Resilient Cache Manager."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ollama.services.cache.cache import CacheManager
from ollama.services.cache.resilient_cache import ResilientCacheManager


class TestResilientCacheManager:
    """Test resilient cache manager functionality."""

    @pytest.fixture
    def mock_cache(self) -> MagicMock:
        """Mock CacheManager instance."""
        cache = MagicMock(spec=CacheManager)
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        return cache

    @pytest.fixture
    def resilient_cache(self, mock_cache: MagicMock) -> ResilientCacheManager:
        """ResilientCacheManager instance with mock storage."""
        return ResilientCacheManager(
            cast(CacheManager, mock_cache),
            failure_threshold=2,
            recovery_timeout=30,
            service_name="test-redis-cache",
        )

    @pytest.mark.asyncio
    async def test_get_success(self, resilient_cache, mock_cache):
        """Test get success through circuit breaker."""
        mock_cache.get.return_value = {"foo": "bar"}

        result = await resilient_cache.get("test_key")

        assert result == {"foo": "bar"}
        mock_cache.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_failure_and_open_circuit(self, resilient_cache, mock_cache):
        """Test that repeated failures open the circuit."""
        mock_cache.get.side_effect = Exception("Redis connection failed")

        # First failure
        result = await resilient_cache.get("test_key")
        assert result is None

        # Second failure - should open circuit
        result = await resilient_cache.get("test_key")
        assert result is None

        # Third call - should fail immediately with CircuitBreakerError (caught and returning None)
        result = await resilient_cache.get("test_key")
        assert result is None

        # Verify mock was called 6 times (3 retries per call, with threshold of 2 calls)
        assert mock_cache.get.call_count == 6

    @pytest.mark.asyncio
    async def test_set_success(self, resilient_cache, mock_cache):
        """Test set success through circuit breaker."""
        mock_cache.set.return_value = True

        result = await resilient_cache.set("test_key", "value")

        assert result is True
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_breaker_open(self, resilient_cache, mock_cache):
        """Test set when breaker is already open."""
        # Fail twice to open circuit
        mock_cache.get.side_effect = Exception("Redis error")
        await resilient_cache.get("k1")
        await resilient_cache.get("k2")

        # Circuit is now open
        result = await resilient_cache.set("test_key", "value")

        assert result is False
        # mock_cache.set should not even be called
        mock_cache.set.assert_not_called()
