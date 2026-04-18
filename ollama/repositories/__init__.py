"""Compatibility package for `ollama.repositories` re-exporting the
implementation now located under `ollama.services.repositories`.

This module provides a stable import surface so code importing
`ollama.repositories` continues to work after the package reorganization.
"""

# Use absolute import to ensure the services package is resolved correctly
from ollama.services.repositories import *  # noqa: F401,F403

__all__ = getattr(__import__("ollama.services.repositories", fromlist=["*"]), "__all__", [])
