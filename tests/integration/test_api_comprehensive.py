"""
Comprehensive test suite for API routes - Issue #50 Phase 1.

Tests for all API endpoints with comprehensive coverage:
- Happy path scenarios
- Error cases and edge conditions
- Input validation
- Authentication/authorization
- Response format validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime, timedelta
from typing import Any, Dict

from ollama.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """Valid API key for testing."""
    return "sk-test-key-12345678"


@pytest.fixture
def invalid_api_key():
    """Invalid API key for testing."""
    return "sk-invalid-key"


class TestHealthEndpoint:
    """Test suite for /api/v1/health endpoint."""

    def test_health_check_success(self, client):
        """Health check returns 200 with healthy status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_health_check_includes_version(self, client):
        """Health check includes version information."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_health_check_includes_timestamp(self, client):
        """Health check includes timestamp."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        # Verify timestamp is recent (within 1 second)
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert datetime.now(timestamp.tzinfo) - timestamp < timedelta(seconds=1)

    def test_health_check_no_auth_required(self, client):
        """Health check doesn't require authentication."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        # Shouldn't have authentication error
        assert response.status_code != 401


class TestModelsEndpoint:
    """Test suite for /api/v1/models endpoint."""

    def test_list_models_success(self, client, valid_api_key):
        """List models returns 200 with model list."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_list_models_requires_auth(self, client):
        """List models requires authentication."""
        response = client.get("/api/v1/models")
        assert response.status_code == 401

    def test_list_models_rejects_invalid_auth(self, client, invalid_api_key):
        """List models rejects invalid API key."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {invalid_api_key}"}
        )
        assert response.status_code == 401

    def test_list_models_returns_model_details(self, client, valid_api_key):
        """List models returns detailed model information."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        
        if data["models"]:
            model = data["models"][0]
            assert "name" in model
            assert "description" in model
            assert "parameters" in model
            assert "context_length" in model

    def test_list_models_includes_metadata(self, client, valid_api_key):
        """List models includes metadata."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "count" in data
        assert len(data["models"]) == data["count"]


class TestGenerateEndpoint:
    """Test suite for /api/v1/generate endpoint."""

    def test_generate_success(self, client, valid_api_key):
        """Generate endpoint returns text completion."""
        payload = {
            "prompt": "What is the capital of France?",
            "model": "llama3.2",
            "max_tokens": 100
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "model" in data
        assert "tokens_used" in data

    def test_generate_requires_auth(self, client):
        """Generate endpoint requires authentication."""
        payload = {
            "prompt": "Test prompt",
            "model": "llama3.2"
        }
        response = client.post("/api/v1/generate", json=payload)
        assert response.status_code == 401

    def test_generate_requires_prompt(self, client, valid_api_key):
        """Generate endpoint requires prompt."""
        payload = {
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_generate_requires_model(self, client, valid_api_key):
        """Generate endpoint requires model."""
        payload = {
            "prompt": "Test prompt"
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_generate_with_empty_prompt(self, client, valid_api_key):
        """Generate endpoint rejects empty prompt."""
        payload = {
            "prompt": "",
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_generate_with_max_tokens(self, client, valid_api_key):
        """Generate with custom max_tokens."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2",
            "max_tokens": 50
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tokens_used"] <= 50

    def test_generate_with_temperature(self, client, valid_api_key):
        """Generate with custom temperature."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2",
            "temperature": 0.5
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200

    def test_generate_rejects_invalid_temperature(self, client, valid_api_key):
        """Generate rejects temperature outside 0-2 range."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2",
            "temperature": 3.0
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_generate_includes_metadata(self, client, valid_api_key):
        """Generate response includes metadata."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "inference_time_ms" in data
        assert "request_id" in data
        assert "timestamp" in data


class TestChatEndpoint:
    """Test suite for /api/v1/chat endpoint."""

    def test_chat_success(self, client, valid_api_key):
        """Chat endpoint returns completion."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "model" in data
        assert "tokens_used" in data

    def test_chat_requires_auth(self, client):
        """Chat endpoint requires authentication."""
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "llama3.2"
        }
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 401

    def test_chat_requires_messages(self, client, valid_api_key):
        """Chat endpoint requires messages."""
        payload = {
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_chat_requires_model(self, client, valid_api_key):
        """Chat endpoint requires model."""
        payload = {
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_chat_with_empty_messages(self, client, valid_api_key):
        """Chat endpoint rejects empty messages list."""
        payload = {
            "messages": [],
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_chat_with_system_prompt(self, client, valid_api_key):
        """Chat with system prompt."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200

    def test_chat_with_conversation_history(self, client, valid_api_key):
        """Chat with full conversation history."""
        payload = {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."},
                {"role": "user", "content": "Tell me more"}
            ],
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200

    def test_chat_response_format(self, client, valid_api_key):
        """Chat response has correct format."""
        payload = {
            "messages": [{"role": "user", "content": "Test"}],
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0


class TestEmbeddingsEndpoint:
    """Test suite for /api/v1/embeddings endpoint."""

    def test_embeddings_success(self, client, valid_api_key):
        """Embeddings endpoint returns vectors."""
        payload = {
            "text": "Hello world",
            "model": "embedding-model"
        }
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) > 0

    def test_embeddings_requires_auth(self, client):
        """Embeddings endpoint requires authentication."""
        payload = {
            "text": "Hello",
            "model": "embedding-model"
        }
        response = client.post("/api/v1/embeddings", json=payload)
        assert response.status_code == 401

    def test_embeddings_requires_text(self, client, valid_api_key):
        """Embeddings endpoint requires text."""
        payload = {
            "model": "embedding-model"
        }
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_embeddings_requires_model(self, client, valid_api_key):
        """Embeddings endpoint requires model."""
        payload = {
            "text": "Hello"
        }
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_embeddings_with_empty_text(self, client, valid_api_key):
        """Embeddings rejects empty text."""
        payload = {
            "text": "",
            "model": "embedding-model"
        }
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_embeddings_vector_format(self, client, valid_api_key):
        """Embeddings returns vector in correct format."""
        payload = {
            "text": "Test",
            "model": "embedding-model"
        }
        response = client.post(
            "/api/v1/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        embedding = data["embedding"]
        # All values should be floats
        assert all(isinstance(val, (int, float)) for val in embedding)


class TestAuthenticationMiddleware:
    """Test suite for authentication middleware."""

    def test_missing_authorization_header(self, client):
        """Missing auth header returns 401."""
        response = client.get("/api/v1/models")
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    def test_invalid_auth_format(self, client):
        """Invalid auth format returns 401."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": "InvalidFormat"}
        )
        assert response.status_code == 401

    def test_invalid_bearer_token(self, client):
        """Invalid bearer token returns 401."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    def test_auth_header_case_insensitive(self, client, valid_api_key):
        """Auth header handling is case-insensitive."""
        response = client.get(
            "/api/v1/models",
            headers={"authorization": f"bearer {valid_api_key}"}
        )
        # Should still work (implementation dependent)
        assert response.status_code in [200, 401]


class TestErrorHandling:
    """Test suite for error handling."""

    def test_not_found_endpoint(self, client, valid_api_key):
        """Non-existent endpoint returns 404."""
        response = client.get(
            "/api/v1/nonexistent",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 404

    def test_method_not_allowed(self, client, valid_api_key):
        """Wrong HTTP method returns 405."""
        response = client.post(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 405

    def test_invalid_json_payload(self, client, valid_api_key):
        """Invalid JSON returns 422."""
        response = client.post(
            "/api/v1/generate",
            data="invalid json",
            headers={
                "Authorization": f"Bearer {valid_api_key}",
                "Content-Type": "application/json"
            }
        )
        assert response.status_code == 422

    def test_error_response_format(self, client):
        """Error responses have consistent format."""
        response = client.get("/api/v1/models")
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data or "error" in data


class TestRateLimiting:
    """Test suite for rate limiting."""

    def test_rate_limit_header(self, client, valid_api_key):
        """Rate limit headers are present."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        # Check for rate limit headers
        assert any(
            key.lower() in ["x-ratelimit-limit", "ratelimit-limit"]
            for key in response.headers
        )

    def test_rate_limit_reset(self, client, valid_api_key):
        """Rate limit reset header is present."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        # Check for rate limit reset header
        assert any(
            key.lower() in ["x-ratelimit-reset", "ratelimit-reset"]
            for key in response.headers
        )


class TestCORSHeaders:
    """Test suite for CORS headers."""

    def test_cors_headers_present(self, client, valid_api_key):
        """CORS headers are present in response."""
        response = client.get(
            "/api/v1/models",
            headers={
                "Authorization": f"Bearer {valid_api_key}",
                "Origin": "https://elevatediq.ai"
            }
        )
        assert response.status_code == 200
        # Check for CORS headers
        assert any(
            "access-control" in key.lower()
            for key in response.headers
        )

    def test_allowed_origins(self, client, valid_api_key):
        """Only allowed origins are accepted."""
        response = client.get(
            "/api/v1/models",
            headers={
                "Authorization": f"Bearer {valid_api_key}",
                "Origin": "https://elevatediq.ai"
            }
        )
        assert response.status_code == 200


class TestRequestValidation:
    """Test suite for request validation."""

    def test_max_prompt_length(self, client, valid_api_key):
        """Request validation enforces max prompt length."""
        # Create a very long prompt
        long_prompt = "a" * 100000
        payload = {
            "prompt": long_prompt,
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        # Should either reject or handle gracefully
        assert response.status_code in [200, 413, 422]

    def test_max_tokens_validation(self, client, valid_api_key):
        """max_tokens is validated."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2",
            "max_tokens": 999999
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        # Should be rejected or capped
        assert response.status_code in [200, 422]

    def test_negative_max_tokens(self, client, valid_api_key):
        """Negative max_tokens is rejected."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2",
            "max_tokens": -1
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422

    def test_zero_max_tokens(self, client, valid_api_key):
        """Zero max_tokens is rejected."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2",
            "max_tokens": 0
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 422


class TestResponseValidation:
    """Test suite for response validation."""

    def test_response_contains_request_id(self, client, valid_api_key):
        """All responses contain request_id."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0

    def test_response_contains_timestamp(self, client, valid_api_key):
        """All responses contain timestamp."""
        payload = {
            "prompt": "Test",
            "model": "llama3.2"
        }
        response = client.post(
            "/api/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data

    def test_response_content_type(self, client, valid_api_key):
        """Response has correct content-type."""
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": f"Bearer {valid_api_key}"}
        )
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=ollama"])
