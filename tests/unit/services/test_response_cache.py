"""Unit tests for Response Cache implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ollama.services.cache.response_cache import ResponseCache


@pytest.fixture
def mock_cache_manager() -> MagicMock:
    """Create mock cache manager."""
    return MagicMock()


@pytest.fixture
def response_cache(mock_cache_manager: MagicMock) -> ResponseCache:
    """Create response cache with mock manager."""
    return ResponseCache(mock_cache_manager, default_ttl=3600)


@pytest.mark.asyncio
async def test_get_cache_key_is_deterministic(response_cache: ResponseCache) -> None:
    """Cache key is consistent for same model/prompt."""
    key1 = response_cache._get_cache_key("llama3.2", "Hello")
    key2 = response_cache._get_cache_key("llama3.2", "Hello")
    key3 = response_cache._get_cache_key("llama3.2", "World")

    assert key1 == key2
    assert key1 != key3


@pytest.mark.asyncio
async def test_get_response_returns_none_when_not_cached(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """get_response returns None when response not cached."""
    mock_cache_manager.get = AsyncMock(return_value=None)

    result = await response_cache.get_response("llama3.2", "Hello")

    assert result is None
    mock_cache_manager.get.assert_called_once()


@pytest.mark.asyncio
async def test_get_response_returns_cached_response(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """get_response returns cached response when found."""
    cached_response = {"text": "Response", "tokens": 10}
    mock_cache_manager.get = AsyncMock(return_value=cached_response)

    result = await response_cache.get_response("llama3.2", "Hello")

    assert result == cached_response


@pytest.mark.asyncio
async def test_get_response_handles_cache_errors(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """get_response handles cache errors gracefully."""
    mock_cache_manager.get = AsyncMock(side_effect=Exception("Cache error"))

    result = await response_cache.get_response("llama3.2", "Hello")

    assert result is None


@pytest.mark.asyncio
async def test_set_response_caches_with_default_ttl(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """set_response caches with default TTL."""
    mock_cache_manager.set = AsyncMock(return_value=True)
    response = {"text": "Response", "tokens": 10}

    success = await response_cache.set_response("llama3.2", "Hello", response)

    assert success is True
    mock_cache_manager.set.assert_called_once()
    call_args = mock_cache_manager.set.call_args
    assert call_args.kwargs["ttl"] == 3600


@pytest.mark.asyncio
async def test_set_response_caches_with_custom_ttl(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """set_response respects custom TTL."""
    mock_cache_manager.set = AsyncMock(return_value=True)
    response = {"text": "Response", "tokens": 10}

    success = await response_cache.set_response("llama3.2", "Hello", response, ttl=7200)

    assert success is True
    call_args = mock_cache_manager.set.call_args
    assert call_args.kwargs["ttl"] == 7200


@pytest.mark.asyncio
async def test_set_response_handles_cache_errors(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """set_response handles cache errors gracefully."""
    mock_cache_manager.set = AsyncMock(side_effect=Exception("Cache error"))
    response = {"text": "Response", "tokens": 10}

    success = await response_cache.set_response("llama3.2", "Hello", response)

    assert success is False


@pytest.mark.asyncio
async def test_delete_response_removes_cache_entry(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """delete_response removes cached response."""
    mock_cache_manager.delete = AsyncMock(return_value=True)

    success = await response_cache.delete_response("llama3.2", "Hello")

    assert success is True
    mock_cache_manager.delete.assert_called_once()


@pytest.mark.asyncio
async def test_delete_response_handles_errors(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """delete_response handles errors gracefully."""
    mock_cache_manager.delete = AsyncMock(side_effect=Exception("Cache error"))

    success = await response_cache.delete_response("llama3.2", "Hello")

    assert success is False


@pytest.mark.asyncio
async def test_clear_model_cache_clears_all_responses(
    response_cache: ResponseCache,
    mock_cache_manager: MagicMock,
) -> None:
    """clear_model_cache removes all responses for model."""
    mock_client = AsyncMock()
    mock_client.keys = AsyncMock(return_value=["key1", "key2", "key3"])
    mock_client.delete = AsyncMock()
    mock_cache_manager.client = mock_client

    count = await response_cache.clear_model_cache("llama3.2")

    assert count == 3
    mock_client.delete.assert_called_once_with("key1", "key2", "key3")


def test_get_metrics_returns_cache_info(response_cache: ResponseCache) -> None:
    """get_metrics returns cache configuration."""
    metrics = response_cache.get_metrics()

    assert metrics["service"] == "response_cache"
    assert metrics["default_ttl"] == 3600
    assert "prefix" in metrics
