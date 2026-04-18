"""Exception hierarchy.

This module exposes the canonical exception types and provides a small set
of backward-compatible aliases expected by older code and tests.
"""

from typing import Any

from .impl.base import AuthenticationError, OllamaError
from .impl.model import ModelNotFoundError

# Backwards-compatible aliases
# Older callers expect `OllamaException` as the base name; keep alias.
OllamaException = OllamaError

# Authentication-related aliases
APIKeyInvalidError = AuthenticationError


class ValidationError(OllamaError):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class InferenceError(OllamaError):
    """Raised when model inference fails."""


# Rate limiting error - not present in older impls, provide a simple class
class RateLimitExceededError(OllamaError):
    """Raised when a client exceeds allowed rate limits.

    Keeps structured metadata for callers to implement retry logic.
    """

    def __init__(
        self,
        limit: int,
        window: int,
        retry_after: int,
        message: str | None = None,
    ) -> None:
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        msg = message or f"Rate limit exceeded: {limit} req per {window}s"
        super().__init__(msg)


# Inference timeout - provided for backward compatibility with tests
class InferenceTimeoutError(OllamaError):
    """Raised when an inference request times out."""

    def __init__(self, elapsed_ms: float, timeout_ms: int) -> None:
        self.elapsed_ms = elapsed_ms
        self.timeout_ms = timeout_ms
        super().__init__(
            code="INFERENCE_TIMEOUT",
            message=f"Inference timed out after {elapsed_ms:.1f}ms (limit: {timeout_ms}ms)",
            status_code=504,
            details={"elapsed_ms": elapsed_ms, "timeout_ms": timeout_ms},
        )


__all__ = [
    "APIKeyInvalidError",
    "AuthenticationError",
    "InferenceTimeoutError",
    "ModelNotFoundError",
    "OllamaError",
    "OllamaException",
    "RateLimitExceededError",
]
