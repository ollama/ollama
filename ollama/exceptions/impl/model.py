"""Model-related exceptions."""

from .base import OllamaError


class ModelError(OllamaError):
    """Base class for model-related errors."""

    def __init__(self, message: str, model_name: str | None = None) -> None:
        super().__init__(message)
        self.model_name = model_name


class ModelNotFoundError(ModelError):
    """Raised when a requested model cannot be found."""


class ModelLoadError(ModelError):
    """Raised when a model fails to load into memory."""
