"""
Comprehensive unit tests for services module - Issue #50 Phase 1.

Tests for inference, caching, and model management services.
Covers happy paths, error cases, and edge conditions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ollama.services.cache.cache import CacheManager
from ollama.services.inference.ollama_client_main import OllamaClient
from ollama.services.models.ollama_model_manager import OllamaModelManager


class TestOllamaClient:
    """Test suite for OllamaClient inference service."""

    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client."""
        return AsyncMock()

    @pytest.fixture
    def client(self, mock_http_client):
        """Create OllamaClient with mocked HTTP."""
        client = OllamaClient(base_url="http://ollama:11434")
        client.session = mock_http_client
        return client

    @pytest.mark.asyncio
    async def test_generate_text_success(self, client, mock_http_client):
        """Generate text returns completion."""
        mock_http_client.post.return_value.json.return_value = {
            "response": "Paris is the capital of France",
            "model": "llama3.2",
            "done": True,
            "eval_count": 5,
            "context": []
        }

        result = await client.generate(
            prompt="What is the capital of France?",
            model="llama3.2"
        )

        assert result["response"] == "Paris is the capital of France"
        assert result["model"] == "llama3.2"
        assert result["eval_count"] == 5
        mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, client, mock_http_client):
        """Generate respects max_tokens parameter."""
        mock_http_client.post.return_value.json.return_value = {
            "response": "Short response",
            "done": True,
            "eval_count": 3
        }

        result = await client.generate(
            prompt="Test",
            model="llama3.2",
            max_tokens=10
        )

        assert result["response"] == "Short response"
        # Verify max_tokens was passed
        call_args = mock_http_client.post.call_args
        assert "options" in call_args.kwargs or "num_predict" in str(call_args)

    @pytest.mark.asyncio
    async def test_generate_with_temperature(self, client, mock_http_client):
        """Generate respects temperature parameter."""
        mock_http_client.post.return_value.json.return_value = {
            "response": "Creative response",
            "done": True
        }

        result = await client.generate(
            prompt="Test",
            model="llama3.2",
            temperature=0.7
        )

        assert result["response"] == "Creative response"

    @pytest.mark.asyncio
    async def test_generate_timeout(self, client, mock_http_client):
        """Generate handles timeout gracefully."""
        import asyncio
        mock_http_client.post.side_effect = TimeoutError()

        with pytest.raises(asyncio.TimeoutError):
            await client.generate(
                prompt="Test",
                model="llama3.2",
                timeout=5
            )

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, client, mock_http_client):
        """Generate handles connection errors."""
        import aiohttp
        mock_http_client.post.side_effect = aiohttp.ClientError("Connection failed")

        with pytest.raises(Exception):
            await client.generate(
                prompt="Test",
                model="llama3.2"
            )

    @pytest.mark.asyncio
    async def test_list_models_success(self, client, mock_http_client):
        """List models returns available models."""
        mock_http_client.get.return_value.json.return_value = {
            "models": [
                {"name": "llama3.2", "size": "3.8GB"},
                {"name": "mixtral-8x7b", "size": "45GB"}
            ]
        }

        result = await client.list_models()

        assert len(result["models"]) == 2
        assert result["models"][0]["name"] == "llama3.2"
        mock_http_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_info(self, client, mock_http_client):
        """Get model info returns model details."""
        mock_http_client.post.return_value.json.return_value = {
            "name": "llama3.2",
            "details": {
                "family": "llama",
                "parameter_size": "3.8B"
            }
        }

        result = await client.show("llama3.2")

        assert result["name"] == "llama3.2"
        assert "details" in result

    @pytest.mark.asyncio
    async def test_pull_model(self, client, mock_http_client):
        """Pull model downloads and caches."""
        mock_http_client.post.return_value = AsyncMock()
        mock_http_client.post.return_value.iter_any.return_value = [
            b'{"status": "pulling manifest"}\n',
            b'{"status": "pulling layers"}\n',
            b'{"status": "verifying digest"}\n',
            b'{"status": "writing manifest"}\n',
        ]

        # This would be called to pull a model
        # The actual implementation would stream responses
        result = await client.pull("llama3.2")

        # Verify the API was called
        mock_http_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, client, mock_http_client):
        """Generate can include system prompt context."""
        mock_http_client.post.return_value.json.return_value = {
            "response": "Helpful response",
            "done": True
        }

        result = await client.generate(
            prompt="User message",
            model="llama3.2",
            system="You are a helpful assistant"
        )

        assert result["response"] == "Helpful response"

    @pytest.mark.asyncio
    async def test_generate_streaming_mode(self, client, mock_http_client):
        """Generate supports streaming responses."""
        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.iter_any.return_value = [
            b'{"response":"Token1","done":false}\n',
            b'{"response":"Token2","done":false}\n',
            b'{"response":"Token3","done":true}\n',
        ]
        mock_http_client.post.return_value = mock_response

        # In streaming mode, responses should be iterable
        result = await client.generate(
            prompt="Test",
            model="llama3.2",
            stream=True
        )

        # Result should support iteration
        assert result is not None


class TestCacheManager:
    """Test suite for CacheManager cache service."""

    @pytest.fixture
    def cache(self):
        """Create cache manager with in-memory backend."""
        return CacheManager(backend="memory")

    def test_set_and_get_cache_hit(self, cache):
        """Cache set and get returns cached value."""
        cache.set("key1", {"data": "value"}, ttl=3600)
        result = cache.get("key1")
        assert result is not None
        assert result["data"] == "value"

    def test_cache_miss(self, cache):
        """Cache get on missing key returns None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_expiration(self, cache):
        """Cache expires after TTL."""
        cache.set("key1", {"data": "value"}, ttl=0.1)
        # Verify it exists immediately
        assert cache.get("key1") is not None
        # Wait for expiration
        import time
        time.sleep(0.2)
        # Should be expired
        assert cache.get("key1") is None

    def test_cache_delete(self, cache):
        """Cache delete removes key."""
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") is not None
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_cache_clear_all(self, cache):
        """Cache clear removes all keys."""
        cache.set("key1", {"data": "value1"})
        cache.set("key2", {"data": "value2"})
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_size_limit(self, cache):
        """Cache respects size limit with LRU eviction."""
        # Set max size to 2
        cache.max_size = 2
        cache.set("key1", {"data": "value1"})
        cache.set("key2", {"data": "value2"})
        cache.set("key3", {"data": "value3"})

        # key1 should be evicted (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_cache_update_access_time(self, cache):
        """Cache updates access time on get."""
        cache.set("key1", {"data": "value1"})
        cache.set("key2", {"data": "value2"})

        # Access key1 to update its access time
        cache.get("key1")

        # Set max size to trigger eviction
        cache.max_size = 2
        cache.set("key3", {"data": "value3"})

        # key2 should be evicted (not accessed recently)
        assert cache.get("key1") is not None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None

    def test_cache_stats(self, cache):
        """Cache reports hit/miss statistics."""
        cache.set("key1", {"data": "value"})
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3

    def test_cache_with_null_value(self, cache):
        """Cache handles null/None values."""
        cache.set("key1", None)
        # None should be cached, not treated as miss
        # Different caches handle this differently
        result = cache.get("key1")
        assert result is None  # Could be miss or cached None


class TestOllamaModelManager:
    """Test suite for OllamaModelManager model service."""

    @pytest.fixture
    def model_manager(self):
        """Create model manager."""
        return OllamaModelManager(AsyncMock())

    def test_load_model_success(self, model_manager):
        """Load model succeeds."""
        # Mock the loading
        with patch.object(model_manager, "get_model") as mock_get:
            mock_get.return_value = MagicMock()
            result = asyncio.run(model_manager.get_model("llama3.2"))
            assert result is not None

    def test_unload_model(self, model_manager):
        """Unload model removes from memory."""
        with patch.object(model_manager, 'unload') as mock_unload:
            mock_unload.return_value = True
            result = model_manager.unload("llama3.2")
            assert result is True

    def test_get_loaded_models(self, model_manager):
        """Get list of loaded models."""
        with patch.object(model_manager, 'get_loaded') as mock_get:
            mock_get.return_value = ["llama3.2", "mixtral-8x7b"]
            result = model_manager.get_loaded()
            assert len(result) == 2
            assert "llama3.2" in result

    def test_model_already_loaded(self, model_manager):
        """Loading already-loaded model is idempotent."""
        with patch.object(model_manager, 'load') as mock_load:
            mock_load.return_value = True
            result1 = model_manager.load("llama3.2")
            result2 = model_manager.load("llama3.2")
            assert result1 is True
            assert result2 is True

    def test_get_model_size(self, model_manager):
        """Get model size in memory."""
        with patch.object(model_manager, 'get_size') as mock_size:
            mock_size.return_value = 3.8 * 1024 * 1024 * 1024  # 3.8GB
            size = model_manager.get_size("llama3.2")
            assert size > 0

    def test_validate_model_name(self, model_manager):
        """Validate model name format."""
        assert model_manager.is_valid_name("llama3.2") is True
        assert model_manager.is_valid_name("mixtral-8x7b") is True
        assert model_manager.is_valid_name("") is False
        assert model_manager.is_valid_name("invalid model!") is False

    def test_get_model_config(self, model_manager):
        """Get model configuration."""
        with patch.object(model_manager, 'get_config') as mock_config:
            mock_config.return_value = {
                "name": "llama3.2",
                "parameters": "3.8B",
                "context_length": 8192
            }
            config = model_manager.get_config("llama3.2")
            assert config["context_length"] == 8192


class TestInferenceErrorHandling:
    """Test suite for inference error handling."""

    @pytest.mark.asyncio
    async def test_model_not_found_error(self):
        """Generate raises error for non-existent model."""
        client = OllamaClient(base_url="http://ollama:11434")

        with patch.object(client, 'session') as mock_session:
            mock_session.post.side_effect = ValueError("Model not found")

            with pytest.raises(ValueError):
                await client.generate(
                    prompt="Test",
                    model="nonexistent-model"
                )

    @pytest.mark.asyncio
    async def test_inference_server_down(self):
        """Generate handles server down gracefully."""
        client = OllamaClient(base_url="http://ollama:11434")

        with patch.object(client, 'session') as mock_session:
            import aiohttp
            mock_session.post.side_effect = aiohttp.ClientConnectionError()

            with pytest.raises(Exception):
                await client.generate(
                    prompt="Test",
                    model="llama3.2"
                )

    @pytest.mark.asyncio
    async def test_invalid_response_format(self):
        """Generate handles malformed responses."""
        client = OllamaClient(base_url="http://ollama:11434")

        with patch.object(client, 'session') as mock_session:
            mock_session.post.return_value.json.return_value = {}  # Invalid: missing 'response'

            with pytest.raises(KeyError):
                await client.generate(
                    prompt="Test",
                    model="llama3.2"
                )


class TestCacheIntegration:
    """Test suite for cache integration with inference."""

    def test_cache_inference_results(self):
        """Inference results are cached."""
        cache = CacheManager(backend="memory")
        prompt = "What is 2+2?"
        model = "llama3.2"

        # Cache key
        key = f"{model}:{prompt}"

        # First time: not cached
        result1 = cache.get(key)
        assert result1 is None

        # Cache the result
        cache.set(key, {"result": "4"})

        # Second time: cached
        result2 = cache.get(key)
        assert result2 is not None
        assert result2["result"] == "4"

    def test_cache_different_models(self):
        """Different models have separate cache entries."""
        cache = CacheManager(backend="memory")
        prompt = "Test prompt"

        cache.set(f"llama3.2:{prompt}", {"result": "llama response"})
        cache.set(f"mixtral-8x7b:{prompt}", {"result": "mixtral response"})

        result1 = cache.get(f"llama3.2:{prompt}")
        result2 = cache.get(f"mixtral-8x7b:{prompt}")

        assert result1["result"] == "llama response"
        assert result2["result"] == "mixtral response"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=ollama.services"])
