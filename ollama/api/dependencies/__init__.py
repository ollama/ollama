"""FastAPI dependency injection module.

Provides dependency functions for request handling, authentication, validation,
and service injection. Dependencies are injected into route handlers via FastAPI's
dependency system.

This module follows FastAPI best practices for dependency management.
"""

from ollama.api.dependencies.model_manager import (
    close_model_manager,
    get_model_manager,
    initialize_model_manager,
)

__all__: list[str] = [
    "initialize_model_manager",
    "get_model_manager",
    "close_model_manager",
]
