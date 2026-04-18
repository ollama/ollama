"""Unit tests for ResilientOllamaClient integration."""

from unittest.mock import AsyncMock, patch

import pytest

from ollama.exceptions.circuit_breaker import CircuitBreakerError
from ollama.services.inference.generate_request import GenerateRequest
from ollama.services.inference.generate_response import GenerateResponse
from ollama.services.inference.resilient_ollama_client import ResilientOllamaClient
from ollama.services.models.model import Model
from ollama.services.models.model_type import ModelType


@pytest.mark.asyncio
async def test_resilient_client_wraps_with_circuit_breaker() -> None:
    """ResilientOllamaClient wraps OllamaClient with circuit breaker."""
    client = ResilientOllamaClient(base_url="http://ollama:11434")

    # Verify circuit breaker exists
    assert client.breaker is not None
    assert client.breaker.name == "ollama-inference"
    assert client.breaker_manager is not None


@pytest.mark.asyncio
async def test_resilient_client_list_models_success() -> None:
    """list_models returns models when circuit is closed."""
    client = ResilientOllamaClient(base_url="http://ollama:11434")

    # Mock the underlying client
    mock_models = [
        Model(
            name="llama3.2",
            size="4.7GB",
            model_type=ModelType.TEXT_GENERATION,
            description="Fast model",
            parameters=7000000000,
            context_length=8192,
            quantization="4-bit",
            loaded=True,
        )
    ]

    with patch.object(client.client, "list_models", new_callable=AsyncMock) as mock:
        mock.return_value = mock_models

        result = await client.list_models()
        assert len(result) == 1
        assert result[0].name == "llama3.2"


@pytest.mark.asyncio
async def test_resilient_client_generate_with_circuit_breaker() -> None:
    """generate method wraps with circuit breaker protection."""
    client = ResilientOllamaClient(
        base_url="http://ollama:11434",
        failure_threshold=3,
        recovery_timeout=60,
    )

    mock_response = GenerateResponse(
        model="llama3.2",
        prompt="Hello, world!",
        response="Hello!",
        done=True,
        context=[],
        total_duration=1000,
        load_duration=500,
        prompt_eval_count=5,
        prompt_eval_duration=100,
        eval_count=10,
        eval_duration=200,
    )

    request = GenerateRequest(
        model="llama3.2",
        prompt="Hello, world!",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        num_predict=100,
        stop=[],
    )

    with patch.object(client.client, "generate", new_callable=AsyncMock) as mock:
        mock.return_value = mock_response

        result = await client.generate(request)
        assert result.response == "Hello!"
        assert result.eval_count == 10


@pytest.mark.asyncio
async def test_resilient_client_circuit_opens_after_failures() -> None:
    """Circuit opens after configured failure threshold."""
    client = ResilientOllamaClient(
        base_url="http://ollama:11434",
        failure_threshold=3,
        recovery_timeout=60,
    )

    request = GenerateRequest(
        model="llama3.2",
        prompt="test",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        num_predict=100,
        stop=[],
    )

    # Mock generate to raise exception
    with patch.object(client.client, "generate", new_callable=AsyncMock) as mock:
        mock.side_effect = Exception("Ollama unavailable")

        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await client.generate(request)

        # Circuit should now be OPEN
        assert client.breaker.state.value == "open"

        # Fourth attempt should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await client.generate(request)


@pytest.mark.asyncio
async def test_resilient_client_breaker_state_query() -> None:
    """get_breaker_state returns circuit breaker state info."""
    client = ResilientOllamaClient(base_url="http://ollama:11434")

    state = client.get_breaker_state()
    assert state["service"] == "ollama-inference"
    assert state["state"] == "closed"
    assert state["failure_count"] == 0
    assert state["success_count"] == 0


@pytest.mark.asyncio
async def test_resilient_client_embeddings_with_circuit_breaker() -> None:
    """generate_embeddings is protected by circuit breaker."""
    client = ResilientOllamaClient(base_url="http://ollama:11434")

    mock_embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]

    with patch.object(client.client, "generate_embeddings", new_callable=AsyncMock) as mock:
        mock.return_value = mock_embeddings

        result = await client.generate_embeddings("all-minilm", "hello")
        assert len(result) == 5
        assert result == mock_embeddings
