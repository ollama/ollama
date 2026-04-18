"""Resilient Ollama client with circuit breaker pattern for fault tolerance.

This module wraps the OllamaClient with circuit breaker logic to handle transient
failures gracefully and prevent cascading failures when the Ollama service is
temporarily unavailable.

Example:
    >>> client = ResilientOllamaClient(
    ...     base_url="http://ollama:11434",
    ...     failure_threshold=5,
    ...     recovery_timeout=60
    ... )
    >>> await client.initialize()
    >>> request = GenerateRequest(
    ...     model="llama3.2",
    ...     prompt="Hello, world!"
    ... )
    >>> response = await client.generate(request)
    >>> print(response.response)
"""

from collections.abc import AsyncGenerator
from typing import Any, cast

import structlog

from ollama.exceptions.circuit_breaker import CircuitBreakerError
from ollama.services.inference.generate_request import GenerateRequest
from ollama.services.inference.generate_response import GenerateResponse
from ollama.services.inference.ollama_client_main import OllamaClient
from ollama.services.models.model import Model
from ollama.services.persistence.chat_message import ChatMessage
from ollama.services.persistence.chat_request import ChatRequest
from ollama.services.resilience.circuit_breaker import (
    CircuitBreakerState,
    get_circuit_breaker_manager,
)

log: Any = structlog.get_logger(__name__)


class ResilientOllamaClient:
    """Resilient wrapper around OllamaClient with circuit breaker pattern.

    Adds fault tolerance by detecting failures and temporarily rejecting
    requests when the Ollama service is unavailable. This prevents
    cascading failures and allows the service time to recover.

    Attributes:
        client: Underlying OllamaClient instance.
        breaker_manager: Circuit breaker manager for Ollama service.
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds before attempting recovery.
    """

    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        timeout: float = 60.0,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
    ) -> None:
        """Initialize resilient Ollama client.

        Args:
            base_url: Base URL for Ollama API (default: ollama:11434).
            timeout: Request timeout in seconds (default: 60).
            failure_threshold: Failures before opening circuit (default: 5).
            recovery_timeout: Seconds before recovery attempt (default: 60).
        """
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        self.breaker_manager = get_circuit_breaker_manager()

        # Get or create circuit breaker for Ollama service
        self.breaker = self.breaker_manager.get_or_create(
            service_name="ollama-inference",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    async def initialize(self) -> None:
        """Initialize client with health check.

        Performs a health check to verify Ollama service connectivity.
        Falls back to default initialization if circuit is open.

        Raises:
            CircuitBreakerError: If circuit is open and recovery timeout not elapsed.
            httpx.HTTPError: If health check fails.
        """
        try:
            # Wrap initialization with circuit breaker
            await self.breaker.call_async(self.client.initialize)
            log.info("resilient_ollama_client_initialized", status="healthy")
        except CircuitBreakerError as e:
            log.warning(
                "resilient_ollama_client_init_circuit_open",
                service="ollama-inference",
                message=str(e),
            )
            raise
        except Exception as e:
            log.error(
                "resilient_ollama_client_init_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def list_models(self) -> list[Model]:
        """List all available models with circuit breaker protection.

        Returns:
            List of available models with metadata.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            httpx.HTTPError: If API request fails.
        """
        try:
            models: list[Model] = cast(
                list[Model], await self.breaker.call_async(self.client.list_models)
            )
            log.info("resilient_ollama_list_models", count=len(models))
            return models
        except CircuitBreakerError:
            log.warning(
                "resilient_ollama_circuit_open",
                operation="list_models",
                service="ollama-inference",
            )
            raise
        except Exception as e:
            log.error(
                "resilient_ollama_list_models_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def show_model(self, name: str) -> Model:
        """Get model details with circuit breaker protection.

        Args:
            name: Model name/identifier.

        Returns:
            Model information and metadata.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            ValueError: If model not found.
        """
        try:
            model: Model = cast(Model, await self.breaker.call_async(self.client.show_model, name))
            log.info("resilient_ollama_show_model", model=name)
            return model
        except CircuitBreakerError:
            log.warning(
                "resilient_ollama_circuit_open",
                operation="show_model",
                model=name,
                service="ollama-inference",
            )
            raise
        except ValueError as e:
            log.warning("resilient_ollama_model_not_found", model=name, error=str(e))
            raise

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[GenerateResponse, None]:
        """Generate text completion with streaming and circuit breaker protection.

        Args:
            request: Generation request with prompt and parameters.

        Yields:
            Generated response chunks.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            httpx.HTTPError: If API request fails.
        """
        try:
            # For streaming, we need to handle the generator differently
            log.info(
                "resilient_ollama_generate_stream",
                model=request.model,
                prompt_len=len(request.prompt),
            )

            # Check circuit state before streaming
            if self.breaker.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker open for {self.breaker.name}. "
                    f"Service temporarily unavailable."
                )

            # Stream responses through the circuit breaker
            async for response in self.client.generate_stream(request):
                yield response

            # Record success
            self.breaker.record_success()

        except CircuitBreakerError:
            log.warning(
                "resilient_ollama_circuit_open",
                operation="generate_stream",
                model=request.model,
                service="ollama-inference",
            )
            raise
        except Exception as e:
            # Record failure and re-raise
            self.breaker.record_failure()
            log.error(
                "resilient_ollama_generate_stream_failed",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text completion with circuit breaker protection.

        Args:
            request: Generation request with prompt and parameters.

        Returns:
            Generated response with text and timing information.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            httpx.HTTPError: If API request fails.
        """
        try:
            log.info(
                "resilient_ollama_generate",
                model=request.model,
                prompt_len=len(request.prompt),
            )

            response: GenerateResponse = cast(
                GenerateResponse,
                await self.breaker.call_async(self.client.generate, request),
            )
            log.info(
                "resilient_ollama_generate_success",
                model=request.model,
                tokens=response.eval_count,
            )
            return response

        except CircuitBreakerError:
            log.warning(
                "resilient_ollama_circuit_open",
                operation="generate",
                model=request.model,
                service="ollama-inference",
            )
            raise
        except Exception as e:
            log.error(
                "resilient_ollama_generate_failed",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def generate_embeddings(self, model: str, prompt: str) -> list[float]:
        """Generate embeddings with circuit breaker protection.

        Args:
            model: Model to use for embeddings.
            prompt: Text to generate embeddings for.

        Returns:
            List of embedding values.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            httpx.HTTPError: If API request fails.
        """
        try:
            log.info(
                "resilient_ollama_generate_embeddings",
                model=model,
                prompt_len=len(prompt),
            )

            embeddings: list[float] = cast(
                list[float],
                await self.breaker.call_async(self.client.generate_embeddings, model, prompt),
            )
            log.info(
                "resilient_ollama_embeddings_success",
                model=model,
                embedding_dim=len(embeddings),
            )
            return embeddings

        except CircuitBreakerError:
            log.warning(
                "resilient_ollama_circuit_open",
                operation="generate_embeddings",
                model=model,
                service="ollama-inference",
            )
            raise
        except Exception as e:
            log.error(
                "resilient_ollama_embeddings_failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def chat(self, request: ChatRequest) -> ChatMessage:
        """Chat with circuit breaker protection."""
        try:
            return cast(ChatMessage, await self.breaker.call_async(self.client.chat, request))
        except Exception as e:
            log.error("resilient_ollama_chat_failed", model=request.model, error=str(e))
            raise

    async def pull_model(self, name: str) -> dict[str, Any]:
        """Pull model with circuit breaker protection."""
        try:
            return cast(dict[str, Any], await self.breaker.call_async(self.client.pull_model, name))
        except Exception as e:
            log.error("resilient_ollama_pull_model_failed", model=name, error=str(e))
            raise

    async def delete_model(self, name: str) -> None:
        """Delete model with circuit breaker protection."""
        try:
            await self.breaker.call_async(self.client.delete_model, name)
        except Exception as e:
            log.error("resilient_ollama_delete_model_failed", model=name, error=str(e))
            raise

    def get_breaker_state(self) -> dict[str, Any]:
        """Get current circuit breaker state.

        Returns:
            Dict with circuit breaker state information.
        """
        return self.breaker.get_state()

    async def close(self) -> None:
        """Close the client and clean up resources."""
        await self.client.close()


__all__ = ["ResilientOllamaClient"]
