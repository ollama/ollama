"""
Comprehensive tests for middleware and utilities - Issue #50 Phase 1.

Tests for authentication, request logging, and validation utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
from typing import Dict, Any

from ollama.middleware.auth import AuthenticationMiddleware
from ollama.utils.validators import validate_prompt, validate_model_name
from ollama.utils.formatters import format_response


class TestAuthenticationMiddleware:
    """Test suite for authentication middleware."""

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        return MagicMock()

    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next."""
        async def next_middleware(req):
            return MagicMock(status_code=200)
        return next_middleware

    def test_valid_api_key(self, mock_request, mock_call_next):
        """Valid API key passes authentication."""
        middleware = AuthenticationMiddleware()
        mock_request.headers = {"Authorization": "Bearer sk-valid-key"}
        
        # Mock validation
        with patch.object(middleware, 'validate_key') as mock_validate:
            mock_validate.return_value = True
            result = middleware.validate_auth(mock_request)
            assert result is True

    def test_missing_auth_header(self, mock_request):
        """Missing auth header fails."""
        middleware = AuthenticationMiddleware()
        mock_request.headers = {}
        
        with pytest.raises(ValueError):
            middleware.validate_auth(mock_request)

    def test_invalid_bearer_format(self, mock_request):
        """Invalid bearer format fails."""
        middleware = AuthenticationMiddleware()
        mock_request.headers = {"Authorization": "InvalidFormat"}
        
        with pytest.raises(ValueError):
            middleware.validate_auth(mock_request)

    def test_empty_api_key(self, mock_request):
        """Empty API key fails."""
        middleware = AuthenticationMiddleware()
        mock_request.headers = {"Authorization": "Bearer "}
        
        with pytest.raises(ValueError):
            middleware.validate_auth(mock_request)

    def test_case_insensitive_bearer(self, mock_request):
        """Bearer token is case-insensitive."""
        middleware = AuthenticationMiddleware()
        mock_request.headers = {"Authorization": "bearer sk-valid-key"}
        
        with patch.object(middleware, 'validate_key') as mock_validate:
            mock_validate.return_value = True
            result = middleware.validate_auth(mock_request)
            assert result is True

    def test_key_revocation(self, mock_request):
        """Revoked API key is rejected."""
        middleware = AuthenticationMiddleware()
        mock_request.headers = {"Authorization": "Bearer sk-revoked-key"}
        
        with patch.object(middleware, 'is_key_revoked') as mock_revoked:
            mock_revoked.return_value = True
            with pytest.raises(ValueError):
                middleware.validate_auth(mock_request)


class TestRequestValidation:
    """Test suite for request validation utilities."""

    def test_validate_prompt_valid(self):
        """Valid prompt passes validation."""
        prompt = "What is the capital of France?"
        assert validate_prompt(prompt) is True

    def test_validate_prompt_empty(self):
        """Empty prompt fails validation."""
        assert validate_prompt("") is False

    def test_validate_prompt_whitespace_only(self):
        """Whitespace-only prompt fails validation."""
        assert validate_prompt("   ") is False

    def test_validate_prompt_max_length(self):
        """Prompt respects max length."""
        long_prompt = "a" * 10000
        result = validate_prompt(long_prompt, max_length=5000)
        assert result is False

    def test_validate_prompt_valid_length(self):
        """Prompt under max length passes."""
        prompt = "a" * 100
        result = validate_prompt(prompt, max_length=5000)
        assert result is True

    def test_validate_model_name_valid(self):
        """Valid model name passes validation."""
        assert validate_model_name("llama3.2") is True
        assert validate_model_name("mixtral-8x7b") is True
        assert validate_model_name("neural-chat") is True

    def test_validate_model_name_empty(self):
        """Empty model name fails validation."""
        assert validate_model_name("") is False

    def test_validate_model_name_invalid_chars(self):
        """Model name with invalid characters fails."""
        assert validate_model_name("invalid model!") is False
        assert validate_model_name("model@name") is False

    def test_validate_model_name_too_long(self):
        """Model name respects max length."""
        long_name = "a" * 500
        result = validate_model_name(long_name, max_length=100)
        assert result is False

    def test_validate_temperature_valid(self):
        """Valid temperature values pass."""
        from ollama.utils.validators import validate_temperature
        assert validate_temperature(0.0) is True
        assert validate_temperature(0.5) is True
        assert validate_temperature(1.0) is True
        assert validate_temperature(2.0) is True

    def test_validate_temperature_invalid(self):
        """Invalid temperature values fail."""
        from ollama.utils.validators import validate_temperature
        assert validate_temperature(-0.1) is False
        assert validate_temperature(2.1) is False
        assert validate_temperature(10.0) is False

    def test_validate_max_tokens_valid(self):
        """Valid max_tokens values pass."""
        from ollama.utils.validators import validate_max_tokens
        assert validate_max_tokens(1) is True
        assert validate_max_tokens(100) is True
        assert validate_max_tokens(8192) is True

    def test_validate_max_tokens_invalid(self):
        """Invalid max_tokens values fail."""
        from ollama.utils.validators import validate_max_tokens
        assert validate_max_tokens(0) is False
        assert validate_max_tokens(-1) is False
        assert validate_max_tokens(999999) is False


class TestResponseFormatting:
    """Test suite for response formatting utilities."""

    def test_format_generate_response(self):
        """Format generate response."""
        response = {
            "text": "Paris",
            "model": "llama3.2",
            "tokens_used": 5,
            "inference_time_ms": 250
        }
        
        formatted = format_response(response)
        
        assert "text" in formatted
        assert "timestamp" in formatted
        assert "request_id" in formatted
        assert formatted["text"] == "Paris"

    def test_format_chat_response(self):
        """Format chat response."""
        response = {
            "response": "Hello! How can I help?",
            "model": "llama3.2",
            "tokens_used": 8
        }
        
        formatted = format_response(response, response_type="chat")
        
        assert "response" in formatted
        assert "timestamp" in formatted
        assert formatted["response"] == "Hello! How can I help?"

    def test_format_error_response(self):
        """Format error response."""
        error_response = {
            "error": "Model not found",
            "code": "MODEL_NOT_FOUND"
        }
        
        formatted = format_response(error_response, is_error=True)
        
        assert "error" in formatted
        assert "code" in formatted
        assert "timestamp" in formatted

    def test_format_adds_timestamp(self):
        """Format response adds timestamp."""
        response = {"text": "Test"}
        formatted = format_response(response)
        
        assert "timestamp" in formatted
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(formatted["timestamp"].replace("Z", "+00:00"))

    def test_format_adds_request_id(self):
        """Format response adds request ID."""
        response = {"text": "Test"}
        formatted = format_response(response)
        
        assert "request_id" in formatted
        assert formatted["request_id"].startswith("req_")

    def test_format_preserves_data(self):
        """Format response preserves original data."""
        original = {
            "text": "Test response",
            "model": "llama3.2",
            "tokens_used": 10,
            "custom_field": "custom_value"
        }
        formatted = format_response(original)
        
        assert formatted["text"] == original["text"]
        assert formatted["model"] == original["model"]
        assert formatted["tokens_used"] == original["tokens_used"]
        assert formatted.get("custom_field") == "custom_value"


class TestRequestLogging:
    """Test suite for request logging utilities."""

    def test_log_request_details(self):
        """Log request includes necessary details."""
        from ollama.middleware.logging import log_request
        
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/generate"
        mock_request.headers = {"content-length": "100"}
        
        with patch('ollama.middleware.logging.logger') as mock_logger:
            log_request(mock_request)
            mock_logger.info.assert_called()

    def test_log_response_details(self):
        """Log response includes necessary details."""
        from ollama.middleware.logging import log_response
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "50"}
        
        with patch('ollama.middleware.logging.logger') as mock_logger:
            log_response(mock_response, elapsed_ms=250)
            mock_logger.info.assert_called()

    def test_log_hides_sensitive_data(self):
        """Request logging hides sensitive data."""
        from ollama.middleware.logging import log_request
        
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/generate"
        mock_request.headers = {"Authorization": "Bearer sk-secret-key"}
        
        with patch('ollama.middleware.logging.logger') as mock_logger:
            log_request(mock_request)
            # Verify secret key is not logged
            call_args = str(mock_logger.info.call_args)
            assert "sk-secret-key" not in call_args


class TestErrorHandling:
    """Test suite for error handling utilities."""

    def test_handle_validation_error(self):
        """Validation error handler returns proper format."""
        from ollama.exceptions import ValidationError
        from ollama.api.schemas.error import ErrorResponse
        
        error = ValidationError("Invalid prompt")
        error_response = ErrorResponse.from_exception(error)
        
        assert error_response.code == "VALIDATION_ERROR"
        assert error_response.message == "Invalid prompt"

    def test_handle_model_not_found_error(self):
        """Model not found error handler returns proper format."""
        from ollama.exceptions import ModelNotFoundError
        from ollama.api.schemas.error import ErrorResponse
        
        error = ModelNotFoundError("llama3.2")
        error_response = ErrorResponse.from_exception(error)
        
        assert error_response.code == "MODEL_NOT_FOUND"
        assert "llama3.2" in error_response.message

    def test_handle_server_error(self):
        """Server error handler returns proper format."""
        from ollama.exceptions import InferenceError
        from ollama.api.schemas.error import ErrorResponse
        
        error = InferenceError("Inference failed")
        error_response = ErrorResponse.from_exception(error)
        
        assert error_response.code == "INFERENCE_ERROR"
        assert error_response.message == "Inference failed"


class TestRateLimiting:
    """Test suite for rate limiting utilities."""

    def test_rate_limit_key_generation(self):
        """Rate limit key is generated correctly."""
        from ollama.middleware.rate_limit import get_rate_limit_key
        
        api_key = "sk-test-key"
        key = get_rate_limit_key(api_key)
        
        assert key is not None
        assert isinstance(key, str)
        assert "sk-test-key" in key or key.startswith("rl_")

    def test_rate_limit_window(self):
        """Rate limit window is enforced."""
        from ollama.middleware.rate_limit import RateLimiter
        
        limiter = RateLimiter(requests_per_minute=10)
        api_key = "sk-test-key"
        
        # First request should pass
        assert limiter.is_allowed(api_key) is True
        
        # Requests up to limit should pass
        for _ in range(9):
            assert limiter.is_allowed(api_key) is True
        
        # Request exceeding limit should fail
        assert limiter.is_allowed(api_key) is False

    def test_rate_limit_window_reset(self):
        """Rate limit window resets after timeout."""
        from ollama.middleware.rate_limit import RateLimiter
        import time
        
        limiter = RateLimiter(requests_per_minute=2, window_size=0.1)
        api_key = "sk-test-key"
        
        # Use up requests
        assert limiter.is_allowed(api_key) is True
        assert limiter.is_allowed(api_key) is True
        assert limiter.is_allowed(api_key) is False
        
        # Wait for window reset
        time.sleep(0.15)
        
        # Should be allowed again
        assert limiter.is_allowed(api_key) is True


class TestSecurityHeaders:
    """Test suite for security header utilities."""

    def test_add_security_headers(self):
        """Security headers are added to response."""
        from ollama.middleware.security import add_security_headers
        
        response = {}
        add_security_headers(response)
        
        assert "X-Content-Type-Options" in response
        assert response["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response
        assert response["X-Frame-Options"] == "DENY"

    def test_csp_header(self):
        """CSP header is set correctly."""
        from ollama.middleware.security import add_security_headers
        
        response = {}
        add_security_headers(response)
        
        assert "Content-Security-Policy" in response

    def test_strict_transport_security(self):
        """HSTS header is set for HTTPS."""
        from ollama.middleware.security import add_security_headers
        
        response = {}
        add_security_headers(response, is_https=True)
        
        assert "Strict-Transport-Security" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=ollama"])
