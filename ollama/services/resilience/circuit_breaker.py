"""Circuit Breaker implementation using tenacity.

Provides automatic failure detection and recovery for external service calls.
Implements the Circuit Breaker pattern with three states:
- CLOSED: Normal operation
- OPEN: Failures detected, requests fail fast
- HALF_OPEN: Testing recovery with limited requests
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ollama.monitoring.impl.metrics import (
    CIRCUIT_BREAKER_FAILURES,
    CIRCUIT_BREAKER_STATE,
    CIRCUIT_BREAKER_TRANSITIONS,
)

log = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitBreakerState(str, Enum):
    """Circuit breaker state enum."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for external service calls.

    Prevents cascading failures by failing fast when services are unavailable.

    Attributes:
        failure_threshold: Number of failures before opening circuit (default: 5)
        recovery_timeout: Seconds to wait before attempting recovery (default: 60)
        success_threshold: Successful calls in HALF_OPEN to close circuit (default: 2)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Service name (ollama, redis, postgres, etc).
            failure_threshold: Failures before opening (default: 5).
            recovery_timeout: Timeout before recovery attempt in seconds (default: 60).
            success_threshold: Successes in HALF_OPEN to close (default: 2).
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None

        # Initialize metric
        CIRCUIT_BREAKER_STATE.labels(service=self.name).set(0)

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to a new state and update metrics."""
        old_state = self.state
        self.state = new_state

        # Update Gauge
        state_value = {
            CircuitBreakerState.CLOSED: 0,
            CircuitBreakerState.OPEN: 1,
            CircuitBreakerState.HALF_OPEN: 2,
        }[new_state]
        CIRCUIT_BREAKER_STATE.labels(service=self.name).set(state_value)

        # Record Transition
        CIRCUIT_BREAKER_TRANSITIONS.labels(
            service=self.name,
            from_state=old_state.value,
            to_state=new_state.value,
        ).inc()

        log.info(
            "circuit_breaker_transition",
            service=self.name,
            from_state=old_state.value,
            to_state=new_state.value,
        )

    def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result from func.

        Raises:
            CircuitBreakerError: If circuit is OPEN.
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitBreakerState.HALF_OPEN)
                self.success_count = 0
            else:
                from ollama.exceptions.circuit_breaker import CircuitBreakerError

                raise CircuitBreakerError(
                    self.name,
                    f"Circuit breaker OPEN for {self.name}. "
                    f"Retry after {self.recovery_timeout}s.",
                )

        # Execute with exponential backoff retry
        try:
            result = self._execute_with_retry(func, *args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with exponential backoff retry.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result from func.

        Raises:
            RetryError: If all retries exhausted.
        """

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _retry_func() -> T:
            return func(*args, **kwargs)

        return _retry_func()

    async def call_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result from func.

        Raises:
            CircuitBreakerError: If circuit is OPEN.
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitBreakerState.HALF_OPEN)
                self.success_count = 0
            else:
                from ollama.exceptions.circuit_breaker import CircuitBreakerError

                raise CircuitBreakerError(
                    self.name,
                    f"Circuit breaker OPEN for {self.name}. "
                    f"Retry after {self.recovery_timeout}s.",
                )

        # Execute with exponential backoff retry
        try:
            result = await self._execute_with_retry_async(func, *args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    async def _execute_with_retry_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with exponential backoff retry.

        Args:
            func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result from func.

        Raises:
            RetryError: If all retries exhausted.
        """

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        async def _retry_func_async() -> Any:
            return await func(*args, **kwargs)

        return await _retry_func_async()

    def record_success(self) -> None:
        """Record a successful call outside circuit breaker context.

        Useful for streaming operations where the circuit breaker
        can't directly wrap the function.
        """
        self._on_success()

    def record_failure(self) -> None:
        """Record a failed call outside circuit breaker context.

        Useful for streaming operations where the circuit breaker
        can't directly wrap the function.
        """
        self._on_failure()

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        # Increment failure counter metric
        CIRCUIT_BREAKER_FAILURES.labels(service=self.name, state=self.state.value).inc()

        log.warning(
            "circuit_breaker_failure",
            service=self.name,
            failure_count=self.failure_count,
            threshold=self.failure_threshold,
        )

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count = 0
            self._transition_to(CircuitBreakerState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery.

        Returns:
            True if recovery_timeout has elapsed since last failure.
        """
        if not self.last_failure_time:
            return True

        elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state for monitoring.

        Returns:
            Dict with current state information.
        """
        return {
            "service": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
        }


class CircuitBreakerManager:
    """Manages circuit breakers for multiple external services."""

    def __init__(self) -> None:
        """Initialize circuit breaker manager."""
        self.breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
    ) -> CircuitBreaker:
        """Get or create circuit breaker for service.

        Args:
            service_name: Name of the service (ollama, redis, postgres).
            failure_threshold: Failures before opening.
            recovery_timeout: Timeout before recovery attempt.
            success_threshold: Successes in HALF_OPEN to close.

        Returns:
            CircuitBreaker instance for the service.
        """
        if service_name not in self.breakers:
            self.breakers[service_name] = CircuitBreaker(
                name=service_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
            )
        return self.breakers[service_name]

    def get_state(self) -> dict[str, Any]:
        """Get state of all circuit breakers.

        Returns:
            Dict mapping service names to their states.
        """
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers (primarily for testing)."""
        self.breakers.clear()


# Global circuit breaker manager
_circuit_breaker_manager: CircuitBreakerManager | None = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager (singleton).

    Returns:
        Global CircuitBreakerManager instance.
    """
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager
