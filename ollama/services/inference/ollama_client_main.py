"""Ollama API client for model inference."""

from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Any, cast

import httpx
import structlog

from ollama.services.inference.generate_request import GenerateRequest
from ollama.services.inference.generate_response import GenerateResponse
from ollama.services.models.model import Model
from ollama.services.models.model_type import ModelType
from ollama.services.persistence.chat_message import ChatMessage
from ollama.services.persistence.chat_request import ChatRequest

log: Any = structlog.get_logger(__name__)


class OllamaClient:
    """HTTP client for Ollama API.

    Communicates with local Ollama backend service for model inference,
    model management, and embedding generation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        """Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama API (default: localhost:11434).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def initialize(self) -> None:
        """Initialize client by performing a lightweight health check.

        This ensures connectivity to the Ollama backend. It's safe to call even if
        the backend is unavailable; exceptions should be handled by callers.
        """
        # Perform a simple request to validate the base URL; ignore response content
        resp = await self.client.get("/api/tags")
        resp.raise_for_status()

    async def list_models(self) -> list[Model]:
        """List all available models on Ollama server.

        Returns:
            List of available models with metadata.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        log.info("ollama_list_models", endpoint=self.base_url)

        response = await self.client.get("/api/tags")
        response.raise_for_status()

        data = response.json()
        models: list[Model] = [
            Model(
                name=m.get("name", "unknown"),
                size=str(m.get("size", "")),
                model_type=ModelType.TEXT_GENERATION,
                description=m.get("details", {}).get("description", ""),
                parameters=int(m.get("parameters", 0) or 0),
                context_length=int(m.get("context_length", 0) or 0),
                quantization=m.get("quantization", ""),
                loaded=bool(m.get("loaded", False)),
            )
            for m in data.get("models", [])
        ]

        log.info("ollama_models_listed", count=len(models))
        return models

    async def show_model(self, name: str) -> Model:
        """Get details for a specific model.

        Args:
            name: Model name/identifier.

        Returns:
            Model information and metadata.
        """
        models = await self.list_models()
        for model in models:
            if model.name == name:
                return model
        raise ValueError(f"Model {name} not found")

    async def embeddings(self, model: str, prompt: str) -> list[float]:
        """Generate embeddings for text (alias for generate_embeddings).

        Args:
            model: Model to use for embeddings.
            prompt: Text to generate embeddings for.

        Returns:
            List of embedding values.
        """
        return await self.generate_embeddings(model, prompt)

    async def generate_stream(
        self,
        request: GenerateRequest | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerateResponse, None]:
        """Generate text completion with streaming.

        Args:
            request: Generation request with prompt and parameters.
            **kwargs: Keyword arguments (prompt, model, etc.) for backward compatibility.

        Yields:
            Generated response chunks.
        """
        if request is None:
            # Reconstruct request from kwargs for backward compatibility
            request = GenerateRequest(
                prompt=kwargs.get("prompt", ""),
                model=kwargs.get("model", "llama3.2"),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                num_predict=kwargs.get("num_predict", 128),
                stop=kwargs.get("stop"),
            )

        log.info("ollama_generate_stream", model=request.model, prompt_len=len(request.prompt))

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "num_predict": request.num_predict,
            "stop": request.stop,
            "stream": True,
        }

        async with self.client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json

                data = json.loads(line)
                yield GenerateResponse(
                    model=data["model"],
                    prompt=request.prompt,
                    response=data["response"],
                    done=data.get("done", False),
                    context=data.get("context", []),
                    total_duration=data.get("total_duration", 0),
                    load_duration=data.get("load_duration", 0),
                    prompt_eval_count=data.get("prompt_eval_count", 0),
                    prompt_eval_duration=data.get("prompt_eval_duration", 0),
                    eval_count=data.get("eval_count", 0),
                    eval_duration=data.get("eval_duration", 0),
                )

    async def generate(
        self,
        request: GenerateRequest | None = None,
        **kwargs: Any,
    ) -> GenerateResponse:
        """Generate text completion.

        Args:
            request: Generation request with prompt and parameters.
            **kwargs: Keyword arguments (prompt, model, etc.) for backward compatibility.

        Returns:
            Generated response with text and timing information.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        if request is None:
            # Reconstruct request from kwargs for backward compatibility
            request = GenerateRequest(
                prompt=kwargs.get("prompt", ""),
                model=kwargs.get("model", "llama3.2"),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                num_predict=kwargs.get("num_predict", 128),
                stop=kwargs.get("stop"),
            )

        log.info("ollama_generate", model=request.model, prompt_len=len(request.prompt))

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "num_predict": request.num_predict,
            "stop": request.stop,
            "stream": False,
        }

        response = await self.client.post("/api/generate", json=payload)
        response.raise_for_status()

        data = response.json()

        return GenerateResponse(
            model=data["model"],
            prompt=request.prompt,
            response=data["response"],
            done=data.get("done", True),
            context=data.get("context", []),
            total_duration=data.get("total_duration", 0),
            load_duration=data.get("load_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            prompt_eval_duration=data.get("prompt_eval_duration", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration=data.get("eval_duration", 0),
        )

    async def chat(self, request: ChatRequest) -> ChatMessage:
        """Generate chat response.

        Args:
            request: Chat request with messages and parameters.

        Returns:
            Chat response message from assistant.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        log.info("ollama_chat", model=request.model, messages=len(request.messages))

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "num_predict": request.num_predict,
            "stream": False,
        }

        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        message = data["message"]

        return ChatMessage(role=message["role"], content=message["content"])

    async def pull_model(self, name: str) -> dict[str, Any]:
        """Pull a model from Ollama library.

        Args:
            name: Model name to pull.

        Returns:
            Success response.
        """
        log.info("ollama_pull_model", model=name)
        response = await self.client.post("/api/pull", json={"name": name, "stream": False})
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def delete_model(self, name: str) -> None:
        """Delete a model from Ollama.

        Args:
            name: Model name to delete.
        """
        log.info("ollama_delete_model", model=name)
        response = await self.client.request("DELETE", "/api/delete", json={"name": name})
        response.raise_for_status()

    async def generate_embeddings(self, model: str, prompt: str) -> list[float]:
        """Generate embeddings for a prompt.

        Args:
            model: Model name.
            prompt: Text to embed.

        Returns:
            List of floats representing the embedding.
        """
        log.info("ollama_embeddings", model=model, prompt_len=len(prompt))
        response = await self.client.post(
            "/api/embeddings", json={"model": model, "prompt": prompt}
        )
        response.raise_for_status()
        data = response.json()
        return cast(list[float], data.get("embedding", []))

    async def close(self) -> None:
        """Close HTTP client connection."""
        await self.client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
