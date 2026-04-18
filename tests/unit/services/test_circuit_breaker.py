"""Unit tests for Circuit Breaker implementation."""

from __future__ import annotations

import pytest

from ollama.exceptions.circuit_breaker import CircuitBreakerError
from ollama.services.resilience import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerState,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self) -> None:
        """Circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker("test-service")
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_successful_call_increments_success_count(self) -> None:
        """Successful calls in HALF_OPEN increment success counter."""
        breaker = CircuitBreaker("test-service", success_threshold=2)
        breaker.state = CircuitBreakerState.HALF_OPEN

        def mock_func() -> str:
            return "success"

        result = breaker.call(mock_func)
        assert result == "success"
        assert breaker.success_count == 1

    def test_circuit_opens_after_threshold_failures(self) -> None:
        """Circuit opens after failure_threshold failures."""
        breaker = CircuitBreaker("test-service", failure_threshold=3)

        def failing_func() -> None:
            raise Exception("Connection failed")

        # First failure
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.failure_count == 1
        assert breaker.state == CircuitBreakerState.CLOSED

        # Second failure
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.failure_count == 2

        # Third failure - circuit opens
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.failure_count == 3
        assert breaker.state == CircuitBreakerState.OPEN

    def test_circuit_rejects_calls_when_open(self) -> None:
        """Circuit breaker rejects calls when OPEN and recovery timeout not elapsed."""
        from datetime import datetime, timedelta, UTC

        breaker = CircuitBreaker("test-service")
        breaker.state = CircuitBreakerState.OPEN
        # Set failure time to recent past (within recovery timeout of 60s)
        breaker.last_failure_time = datetime.now(UTC) - timedelta(seconds=10)

        def func() -> None:
            pass

        with pytest.raises(CircuitBreakerError):
            breaker.call(func)

    def test_circuit_transitions_to_half_open_after_timeout(self) -> None:
        """Circuit transitions from OPEN to HALF_OPEN after timeout."""
        from datetime import UTC, datetime, timedelta

        breaker = CircuitBreaker("test-service", recovery_timeout=1)
        breaker.state = CircuitBreakerState.OPEN
        breaker.last_failure_time = datetime.now(UTC) - timedelta(seconds=2)

        def func() -> str:
            return "success"

        result = breaker.call(func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_half_open_closes_after_threshold_successes(self) -> None:
        """Circuit closes after success_threshold successes in HALF_OPEN."""
        breaker = CircuitBreaker("test-service", success_threshold=2)
        breaker.state = CircuitBreakerState.HALF_OPEN

        def func() -> str:
            return "success"

        # First success
        breaker.call(func)
        assert breaker.success_count == 1
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Second success - circuit closes
        breaker.call(func)
        assert breaker.success_count == 2
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_get_state_returns_current_state(self) -> None:
        """get_state returns current circuit breaker state."""
        breaker = CircuitBreaker("test-service")
        state = breaker.get_state()

        assert state["service"] == "test-service"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["success_count"] == 0


class TestCircuitBreakerManager:
    """Tests for CircuitBreakerManager class."""

    def test_get_or_create_creates_new_breaker(self) -> None:
        """get_or_create creates new circuit breaker for new service."""
        manager = CircuitBreakerManager()
        breaker = manager.get_or_create("test-service")

        assert breaker.name == "test-service"
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_get_or_create_returns_existing_breaker(self) -> None:
        """get_or_create returns existing breaker for service."""
        manager = CircuitBreakerManager()
        breaker1 = manager.get_or_create("test-service")
        breaker2 = manager.get_or_create("test-service")

        assert breaker1 is breaker2

    def test_get_state_returns_all_breakers(self) -> None:
        """get_state returns state of all circuit breakers."""
        manager = CircuitBreakerManager()
        manager.get_or_create("service1")
        manager.get_or_create("service2")

        state = manager.get_state()
        assert "service1" in state
        assert "service2" in state
        assert state["service1"]["state"] == "closed"
        assert state["service2"]["state"] == "closed"
