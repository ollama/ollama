"""Custom exception hierarchy.

Defines all application exceptions organized by domain. Exceptions provide
structured error information for logging, API responses, and error handling.

This module enforces explicit error handling across the application.
"""

from .base import AuthenticationError, OllamaError
from .model import ModelError, ModelLoadError, ModelNotFoundError

__all__ = [
    "AuthenticationError",
    "OllamaError",
    "ModelError",
    "ModelLoadError",
    "ModelNotFoundError",
]
