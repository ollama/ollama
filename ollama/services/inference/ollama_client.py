"""Ollama Client Service (backward compatibility).

This module re-exports classes split into individual modules.
For new code, import directly from specific modules.

Legacy imports (deprecated):
    >>> from ollama.services.inference.ollama_client import OllamaClient, ChatMessage, ChatRequest

Preferred imports:
    >>> from ollama.services.inference.ollama_client_main import OllamaClient
    >>> from ollama.services.persistence.chat_message import ChatMessage
    >>> from ollama.services.persistence.chat_request import ChatRequest
"""

from ollama.services.inference.generate_request import GenerateRequest
from ollama.services.inference.ollama_client_main import OllamaClient
from ollama.services.inference.resilient_ollama_client import ResilientOllamaClient
from ollama.services.persistence.chat_message import ChatMessage
from ollama.services.persistence.chat_request import ChatRequest

# Singleton management for backward compatibility
_ollama_client: OllamaClient | ResilientOllamaClient | None = None


def init_ollama_client(
    base_url: str, timeout: float = 60.0, use_resilience: bool = True
) -> OllamaClient | ResilientOllamaClient:
    """Initialize and store a global OllamaClient instance.

    Args:
        base_url: Base URL for the Ollama backend service.
        timeout: Request timeout in seconds.
        use_resilience: Whether to wrap client in a circuit breaker.

    Returns:
        The initialized OllamaClient or ResilientOllamaClient instance.
    """
    global _ollama_client
    if use_resilience:
        _ollama_client = ResilientOllamaClient(base_url=base_url, timeout=timeout)
    else:
        _ollama_client = OllamaClient(base_url=base_url, timeout=timeout)
    return _ollama_client


def get_ollama_client() -> OllamaClient | ResilientOllamaClient:
    """Get the global OllamaClient instance.

    Raises:
        RuntimeError: If the client has not been initialized.
    """
    if _ollama_client is None:
        raise RuntimeError("Ollama client not initialized")
    return _ollama_client


def clear_ollama_client() -> None:
    """Clear the global OllamaClient instance."""

    global _ollama_client
    _ollama_client = None


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "GenerateRequest",
    "OllamaClient",
    "clear_ollama_client",
    "get_ollama_client",
    "init_ollama_client",
]
