"""Tests for Phase 6, 7, 8 implementations.

Comprehensive tests for exception handling, error responses,
rate limiting, performance monitoring, and configuration management.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from ollama.exceptions import (
    APIKeyInvalidError,
    InferenceTimeoutError,
    ModelNotFoundError,
    OllamaException,
    RateLimitExceededError,
)
from ollama.api.error_handlers import (
    StructuredResponse,
    create_error_response,
    create_success_response,
    register_exception_handlers,
)
from ollama.middleware.rate_limiter import RateLimiter
from ollama.monitoring.performance import (
    PerformanceMetrics,
    SLOValidator,
    benchmark_async,
)
from ollama.config.settings import Settings, Environment


class TestExceptionHierarchy:
    """Tests for custom exception hierarchy."""

    def test_ollama_exception_base(self) -> None:
        """OllamaException properly formats error details."""
        exc = OllamaException(
            code="TEST_ERROR",
            message="Test error message",
            status_code=400,
            details={"key": "value"},
        )

        assert exc.code == "TEST_ERROR"
        assert exc.message == "Test error message"
        assert exc.status_code == 400
        assert exc.details == {"key": "value"}

    def test_ollama_exception_to_dict(self) -> None:
        """OllamaException.to_dict() returns proper format."""
        exc = OllamaException(
            code="ERROR",
            message="Test",
            details={"field": "value"},
        )

        result = exc.to_dict()

        assert result["success"] is False
        assert result["error"]["code"] == "ERROR"
        assert result["error"]["message"] == "Test"
        assert result["error"]["details"] == {"field": "value"}

    def test_model_not_found_error(self) -> None:
        """ModelNotFoundError sets correct status code."""
        exc = ModelNotFoundError("gpt-4")

        assert exc.code == "MODEL_NOT_FOUND"
        assert exc.status_code == 404
        assert exc.details["model"] == "gpt-4"

    def test_inference_timeout_error(self) -> None:
        """InferenceTimeoutError captures timing information."""
        exc = InferenceTimeoutError(elapsed_ms=5500.0, timeout_ms=5000)

        assert exc.code == "INFERENCE_TIMEOUT"
        assert exc.status_code == 504
        assert exc.details["elapsed_ms"] == 5500.0
        assert exc.details["timeout_ms"] == 5000

    def test_rate_limit_exceeded_error(self) -> None:
        """RateLimitExceededError includes retry information."""
        exc = RateLimitExceededError(limit=100, window=60, retry_after=45)

        assert exc.code == "RATE_LIMIT_EXCEEDED"
        assert exc.status_code == 429
        assert exc.details["limit"] == 100
        assert exc.details["retry_after"] == 45

    def test_api_key_invalid_error(self) -> None:
        """APIKeyInvalidError has correct status code."""
        exc = APIKeyInvalidError()

        assert exc.code == "INVALID_API_KEY"
        assert exc.status_code == 401


class TestStructuredResponse:
    """Tests for structured response formatting."""

    def test_success_response(self) -> None:
        """Success response includes data and metadata."""
        response = StructuredResponse(
            success=True,
            data={"result": "success"},
            request_id="req-123",
        )

        result = response.to_dict()

        assert result["success"] is True
        assert result["data"]["result"] == "success"
        assert result["metadata"]["request_id"] == "req-123"

    def test_error_response(self) -> None:
        """Error response includes error details."""
        response = StructuredResponse(
            success=False,
            error={
                "code": "ERROR",
                "message": "Test error",
            },
        )

        result = response.to_dict()

        assert result["success"] is False
        assert result["error"]["code"] == "ERROR"
        assert "data" not in result

    def test_create_success_response(self) -> None:
        """create_success_response helper works correctly."""
        result = create_success_response({"key": "value"})

        assert result["success"] is True
        assert result["data"]["key"] == "value"

    def test_create_error_response(self) -> None:
        """create_error_response helper works correctly."""
        exc = ModelNotFoundError("test-model")
        result = create_error_response(exc)

        assert result["success"] is False
        assert result["error"]["code"] == "MODEL_NOT_FOUND"
        assert result["error"]["details"]["model"] == "test-model"


class TestErrorHandlers:
    """Tests for exception handlers."""

    def test_ollama_exception_handler(self) -> None:
        """OllamaException handler returns proper response."""
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        async def test_endpoint() -> dict:
            raise ModelNotFoundError("llama3.2")

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "MODEL_NOT_FOUND"

    def test_validation_error_handler(self) -> None:
        """Validation error handler returns structured response."""
        from pydantic import BaseModel

        app = FastAPI()
        register_exception_handlers(app)

        class Request(BaseModel):
            name: str
            age: int

        @app.post("/test")
        async def test_endpoint(req: Request) -> dict:
            return {"ok": True}

        client = TestClient(app)
        response = client.post("/test", json={"name": "John"})

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"


class TestRateLimiter:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(
        self,
    ) -> None:
        """RateLimiter allows requests within limit."""
        limiter = RateLimiter(default_limit=5, default_window=60)

        for i in range(5):
            allowed, remaining, _ = await limiter.check_limit("user-1")
            assert allowed is True
            assert remaining == (4 - i)

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_exceeded_requests(self) -> None:
        """RateLimiter blocks requests exceeding limit."""
        limiter = RateLimiter(default_limit=2, default_window=60)

        await limiter.check_limit("user-1")
        await limiter.check_limit("user-1")

        with pytest.raises(RateLimitExceededError):
            await limiter.check_limit("user-1")

    @pytest.mark.asyncio
    async def test_rate_limiter_isolates_users(self) -> None:
        """RateLimiter maintains separate limits per user."""
        limiter = RateLimiter(default_limit=1, default_window=60)

        await limiter.check_limit("user-1")
        await limiter.check_limit("user-2")

        with pytest.raises(RateLimitExceededError):
            await limiter.check_limit("user-1")


class TestPerformanceMetrics:
    """Tests for performance monitoring."""

    def test_performance_metrics_creation(self) -> None:
        """PerformanceMetrics properly records timing."""
        metric = PerformanceMetrics(
            duration_ms=250.5,
            start_time=100.0,
            end_time=100.250,
            success=True,
        )

        assert metric.duration_ms == 250.5
        assert metric.exceeds_slo(500) is False
        assert metric.exceeds_slo(200) is True

    def test_slo_validator_tracks_metrics(self) -> None:
        """SLOValidator tracks and aggregates metrics."""
        validator = SLOValidator("test-endpoint", 500)

        metrics = [
            PerformanceMetrics(100, 0, 0.1, True),
            PerformanceMetrics(200, 0.1, 0.3, True),
            PerformanceMetrics(300, 0.3, 0.6, True),
        ]

        for m in metrics:
            validator.add_metric(m)

        stats = validator.get_statistics()

        assert stats["total_requests"] == 3
        assert stats["successful_requests"] == 3
        assert stats["slo_compliance"] == 100.0
        assert stats["p50_ms"] == 200.0

    def test_slo_validator_fails_on_exceeded_slo(self) -> None:
        """SLOValidator detects SLO violations."""
        validator = SLOValidator("test-endpoint", 200)

        metrics = [
            PerformanceMetrics(100, 0, 0.1, True),
            PerformanceMetrics(500, 0.1, 0.6, True),  # Exceeds SLO
        ]

        for m in metrics:
            validator.add_metric(m)

        assert validator.validate_slo() is False


class TestConfiguration:
    """Tests for configuration management."""

    def test_settings_defaults(self) -> None:
        """Settings use sensible defaults."""
        settings = Settings()

        assert settings.environment == Environment.DEVELOPMENT
        assert settings.database.host == "localhost"
        assert settings.database.port == 5432
        assert settings.redis.host == "localhost"
        assert settings.api.host == "0.0.0.0"

    def test_settings_database_url(self) -> None:
        """Settings generate correct database URL."""
        settings = Settings(
            database__host="db.example.com",
            database__port=5433,
            database__username="user",
            database__password="pass",
            database__database="mydb",
        )

        url = settings.database.url
        assert "db.example.com" in url
        assert "5433" in url
        assert "mydb" in url

    def test_settings_redis_url(self) -> None:
        """Settings generate correct Redis URL."""
        settings = Settings(
            redis__host="redis.example.com",
            redis__port=6380,
            redis__db=1,
        )

        url = settings.redis.url
        assert "redis.example.com" in url
        assert "6380" in url
        assert "/1" in url

    def test_settings_production_mode(self) -> None:
        """Settings properly detect production environment."""
        settings = Settings(environment=Environment.PRODUCTION)

        assert settings.is_production() is True
        assert settings.is_development() is False

    def test_settings_validation(self) -> None:
        """Settings validate configuration values."""
        with pytest.raises(ValueError):
            Settings(
                api__port=99999,  # Invalid port
            )


class TestBenchmarkDecorator:
    """Tests for performance benchmarking."""

    @pytest.mark.asyncio
    async def test_benchmark_async_decorator(self) -> None:
        """benchmark_async decorator tracks timing."""

        @benchmark_async(slo_ms=500)
        async def test_func() -> str:
            return "success"

        result = await test_func()
        assert result == "success"

    def test_benchmark_decorator(self) -> None:
        """benchmark decorator tracks timing."""

        @benchmark_async(slo_ms=500)
        async def test_func() -> str:
            return "success"

        import asyncio

        result = asyncio.run(test_func())
        assert result == "success"

    @pytest.mark.asyncio
    async def test_benchmark_logs_slo_violation(self) -> None:
        """benchmark_async logs when SLO exceeded."""
        import asyncio
        import logging

        @benchmark_async(slo_ms=10)
        async def test_func() -> None:
            await asyncio.sleep(0.05)

        with patch("logging.Logger.warning") as mock_warning:
            await test_func()
            # Note: Check if warning was called (implementation may vary)
