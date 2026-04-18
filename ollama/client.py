"""Main Ollama client interface."""

import os
from typing import Any, cast

httpx: Any = None
try:
    import httpx
except Exception:  # pragma: no cover - allow package import without httpx installed
    httpx = None


class Client:
    """
    Ollama client for interacting with local or remote inference server.

    Supports both local (http://localhost:8000) and public endpoints
    (https://elevatediq.ai/ollama) with automatic endpoint detection.

    Example:
        >>> # Local development
        >>> client = Client()
        >>>
        >>> # Public production endpoint (elevatediq.ai)
        >>> client = Client(base_url="https://elevatediq.ai/ollama")
        >>> response = client.generate(
        ...     model="llama2",
        ...     prompt="What is AI?",
        ...     stream=False
        ... )
        >>> print(response.text)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize Ollama client.

        Args:
            base_url: URL of Ollama server. Defaults to:
                     - OLLAMA_PUBLIC_URL env var if set
                     - https://elevatediq.ai/ollama if OLLAMA_ENV=production
                     - http://localhost:8000 otherwise
            api_key: API key for authentication. Defaults to OLLAMA_API_KEY env var
        """
        # Determine base URL with intelligent defaults
        if base_url is None:
            # Check for public URL first (production)
            base_url = os.getenv("OLLAMA_PUBLIC_URL")
            if not base_url:
                # Check for public flag
                if os.getenv("OLLAMA_ENV") == "production":
                    base_url = "https://elevatediq.ai/ollama"
                else:
                    base_url = os.getenv("OLLAMA_HOST", "http://localhost:8000")

        self.base_url = (base_url or "http://localhost:8000").rstrip("/")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")

        # Setup headers
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Add user agent
        headers["User-Agent"] = "ollama-client/1.0.0"

        if httpx is None:
            raise RuntimeError(
                "httpx is required for `Client` but is not installed. Install with `pip install httpx`."
            )

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=300.0,
        )

    def health(self) -> dict[str, Any]:
        """Check server health."""
        response = self.client.get("/health")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate text using specified model.

        Args:
            model: Model identifier (e.g., "llama2")
            prompt: Input prompt for generation
            stream: Whether to stream output
            **kwargs: Additional parameters (temperature, top_p, etc.)

        Returns:
            Generation response with text field

        Raises:
            httpx.HTTPError: If request fails

        Example:
            >>> response = client.generate(
            ...     model="llama2",
            ...     prompt="Explain local AI",
            ...     temperature=0.7
            ... )
            >>> print(response.text)
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs,
        }
        response = self.client.post("/api/generate", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Chat interface (OpenAI-compatible).

        Args:
            model: Model identifier
            messages: Chat messages
            **kwargs: Additional parameters

        Returns:
            Chat response
        """
        payload = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def embeddings(
        self,
        model: str,
        input_text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate embeddings.

        Args:
            model: Embedding model identifier
            input_text: Text to embed
            **kwargs: Additional parameters

        Returns:
            Embeddings response
        """
        payload = {
            "model": model,
            "input": input_text,
            **kwargs,
        }
        response = self.client.post("/v1/embeddings", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def list_models(self) -> dict[str, Any]:
        """List available models."""
        response = self.client.get("/api/models")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def __del__(self) -> None:
        """Cleanup client connection."""
        try:
            if getattr(self, "client", None) is not None:
                self.client.close()
        except Exception:
            pass
