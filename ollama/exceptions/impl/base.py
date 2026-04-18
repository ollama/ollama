"""Base exception hierarchy for Ollama."""

from typing import Any


class OllamaError(Exception):
    """Base exception for all Ollama-specific errors.

    Provides structured fields expected by API handlers: `code`,
    `message`, `details`, and `status_code`, plus a `log()` helper.
    """

    def __init__(
        self,
        code: str = "OLLAMA_ERROR",
        message: str | None = None,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = message or code
        super().__init__(msg)
        self.code = code
        self.message = msg
        self.details = details or {}
        self.status_code = status_code

    def log(self) -> None:
        """Emit a structured log entry for this exception."""
        import logging

        logger = logging.getLogger("ollama.exceptions")
        logger.error(
            "exception_occurred",
            extra={
                "code": getattr(self, "code", "OLLAMA_ERROR"),
                "message": getattr(self, "message", ""),
                "details": getattr(self, "details", {}),
                "status_code": getattr(self, "status_code", 500),
            },
        )


class AuthenticationError(OllamaError):
    """Base class for authentication errors."""
