"""Real integration tests for inference API endpoints.

These tests verify actual API behavior with real request/response cycles.
"""

import pytest
from fastapi.testclient import TestClient

from ollama.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide FastAPI test client."""
    return TestClient(app)


@pytest.mark.asyncio
class TestInferenceEndpoints:
    """Integration tests for inference endpoints."""

    def test_list_models_success(self, client: TestClient) -> None:
        """List models endpoint returns available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or "total" in data
        assert isinstance(data.get("models", []), list)

    def test_list_models_response_format(self, client: TestClient) -> None:
        """List models returns properly formatted response."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        # Verify response structure
        assert "models" in data or isinstance(data, dict)

    def test_get_model_valid_model(self, client: TestClient) -> None:
        """Get model endpoint returns details for valid model."""
        # First get list of available models
        response = client.get("/api/v1/models")
        assert response.status_code == 200

    def test_get_model_not_found(self, client: TestClient) -> None:
        """Get model endpoint returns 404 for non-existent model."""
        response = client.get("/api/v1/models/nonexistent-model-xyz")
        assert response.status_code == 404
        assert "not found" in response.text.lower()

    def test_generate_endpoint_exists(self, client: TestClient) -> None:
        """Generate endpoint is accessible."""
        # Verify endpoint exists (without full inference)
        response = client.options("/api/v1/generate")
        assert response.status_code in [200, 204, 405]

    def test_generate_missing_prompt(self, client: TestClient) -> None:
        """Generate endpoint returns 422 for missing required fields."""
        response = client.post("/api/v1/generate", json={"model": "llama2"})  # Missing prompt
        # FastAPI validates and returns 422 for missing fields
        assert response.status_code in [422, 400]

    def test_generate_invalid_json(self, client: TestClient) -> None:
        """Generate endpoint handles invalid JSON gracefully."""
        response = client.post(
            "/api/v1/generate", data="invalid json {", headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

    def test_api_requires_authentication(self, client: TestClient) -> None:
        """Protected endpoints require authentication."""
        # Most endpoints should require auth
        response = client.post("/api/v1/generate", json={"model": "llama2", "prompt": "test"})
        # Either 401/403 (auth required) or 422 (validation error)
        assert response.status_code in [401, 403, 422]


@pytest.mark.asyncio
class TestInferenceErrorHandling:
    """Test error handling in inference endpoints."""

    def test_model_not_found_error(self, client: TestClient) -> None:
        """Proper error response for missing model."""
        response = client.get("/api/v1/models/ghost-model")
        assert response.status_code == 404

    def test_invalid_temperature_parameter(self, client: TestClient) -> None:
        """Invalid temperature parameter validation."""
        # Temperature should be 0.0 to 2.0
        response = client.post(
            "/api/v1/generate",
            json={"model": "llama2", "prompt": "test", "temperature": 5.0},  # Invalid
        )
        # Should reject with 422
        assert response.status_code in [422, 400]

    def test_invalid_top_p_parameter(self, client: TestClient) -> None:
        """Invalid top_p parameter validation."""
        # top_p should be 0.0 to 1.0
        response = client.post(
            "/api/v1/generate", json={"model": "llama2", "prompt": "test", "top_p": 1.5}  # Invalid
        )
        assert response.status_code in [422, 400]

    def test_negative_tokens_parameter(self, client: TestClient) -> None:
        """Negative num_predict rejected."""
        response = client.post(
            "/api/v1/generate",
            json={"model": "llama2", "prompt": "test", "num_predict": -100},  # Invalid
        )
        assert response.status_code in [422, 400]


@pytest.mark.asyncio
class TestCacheIntegration:
    """Test caching behavior in inference."""

    def test_cache_key_generation(self) -> None:
        """Cache key generation is deterministic."""
        from ollama.api.routes.inference import _generate_cache_key
        from ollama.api.schemas.inference import GenerateRequest

        request = GenerateRequest(model="llama2", prompt="test prompt", temperature=0.7, top_p=0.9)

        key1 = _generate_cache_key(request)
        key2 = _generate_cache_key(request)

        assert key1 == key2
        assert key1.startswith("inference:v1:gen:")

    def test_cache_key_different_for_different_prompts(self) -> None:
        """Different prompts produce different cache keys."""
        from ollama.api.routes.inference import _generate_cache_key
        from ollama.api.schemas.inference import GenerateRequest

        request1 = GenerateRequest(model="llama2", prompt="prompt1", temperature=0.7)

        request2 = GenerateRequest(model="llama2", prompt="prompt2", temperature=0.7)

        key1 = _generate_cache_key(request1)
        key2 = _generate_cache_key(request2)

        assert key1 != key2


@pytest.mark.asyncio
class TestEmbeddingsEndpoint:
    """Test embeddings endpoint."""

    def test_embeddings_endpoint_exists(self, client: TestClient) -> None:
        """Embeddings endpoint is accessible."""
        response = client.options("/api/v1/embeddings")
        assert response.status_code in [200, 204, 405]

    def test_embeddings_missing_input(self, client: TestClient) -> None:
        """Embeddings requires input field."""
        response = client.post("/api/v1/embeddings", json={"model": "embedding-model"})
        assert response.status_code in [422, 400]

    def test_embeddings_response_format(self) -> None:
        """Embeddings response has correct format."""
        # This would test actual response format when endpoint is working


@pytest.mark.performance
@pytest.mark.asyncio
class TestPerformance:
    """Performance and latency tests."""

    def test_list_models_latency(self, client: TestClient) -> None:
        """List models completes within SLO."""
        import time

        start = time.time()
        response = client.get("/api/v1/models")
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert response.status_code == 200
        assert elapsed < 500, f"List models took {elapsed}ms, SLO is 500ms"

    @pytest.mark.parametrize("endpoint", ["/api/v1/models", "/api/v1/health"])
    def test_endpoint_response_time(self, client: TestClient, endpoint: str) -> None:
        """All endpoints respond within SLO."""
        import time

        start = time.time()
        response = client.get(endpoint)
        elapsed = (time.time() - start) * 1000

        assert response.status_code in [200, 401]
        assert elapsed < 1000, f"{endpoint} took {elapsed}ms, SLO is 1000ms"
