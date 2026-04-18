#!/usr/bin/env python
"""Ollama package initialization.

Lightweight package metadata and lazy attribute access to avoid importing
heavy dependencies (like `httpx`) at package import time. This allows test
discovery and other tooling to import `ollama` without requiring runtime
dependencies until the `Client` class is actually used.
"""

from typing import Any

__version__ = "1.0.0"
__author__ = "kushin77"
__description__ = "Elite local AI development platform for LLM inference"


def __getattr__(name: str) -> Any:
    """Lazy-import attributes from submodules on demand.

    Supports: `Client`.
    """
    if name == "Client":
        from .client import Client  # local import

        return Client
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["Client"]
