"""Resilience patterns for external service integration.

Provides circuit breaker, retry logic, and fault tolerance for calling
external services like Ollama, Redis, and PostgreSQL.
"""

from ollama.services.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerState,
    get_circuit_breaker_manager,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitBreakerState",
    "get_circuit_breaker_manager",
]
