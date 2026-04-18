"""Circuit Breaker exceptions for external service resilience.

Handles failures in external service calls (Ollama, Redis, PostgreSQL)
with automatic recovery and cascading failure prevention.
"""

from __future__ import annotations


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is in OPEN state (service unavailable)."""

    def __init__(self, service: str, message: str = "") -> None:
        """Initialize circuit breaker error.

        Args:
            service: Name of the service that failed (ollama, redis, postgres).
            message: Optional additional context.
        """
        self.service = service
        self.message = message or f"Circuit breaker OPEN for {service}"
        super().__init__(self.message)


class ServiceUnavailableError(Exception):
    """Raised when external service is temporarily unavailable."""

    def __init__(self, service: str, reason: str = "") -> None:
        """Initialize service unavailable error.

        Args:
            service: Name of the service that is unavailable.
            reason: Optional reason for unavailability.
        """
        self.service = service
        self.reason = reason
        super().__init__(f"{service} is unavailable: {reason}")


class ServiceTimeoutError(Exception):
    """Raised when external service request times out."""

    def __init__(self, service: str, timeout_seconds: float = 0.0) -> None:
        """Initialize service timeout error.

        Args:
            service: Name of the service that timed out.
            timeout_seconds: Timeout threshold in seconds.
        """
        self.service = service
        self.timeout_seconds = timeout_seconds
        super().__init__(f"{service} request timed out after {timeout_seconds}s")
