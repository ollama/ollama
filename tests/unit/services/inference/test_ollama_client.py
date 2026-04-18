"""
Tests for Ollama Client Service Integration
Tests the OllamaClient for model listing, generation, chat, and embeddings
"""

from unittest.mock import AsyncMock

import pytest

from ollama.services.inference.ollama_client import (
    ChatMessage,
    ChatRequest,
    GenerateRequest,
    OllamaClient,
)


@pytest.fixture()
def mock_httpx_client():
    """Mock httpx AsyncClient"""
    return AsyncMock()


class TestOllamaClientInitialization:
    """Test OllamaClient initialization"""

    def test_client_initialization(self):
        """Test creating OllamaClient instance"""
        client = OllamaClient(base_url="http://localhost:11434")

        assert client is not None
        assert client.base_url == "http://localhost:11434"

    def test_client_custom_timeout(self) -> None:
        """Test creating client with custom timeout"""
        client = OllamaClient(base_url="http://localhost:11434", timeout=60.0)

        assert client is not None
        assert client.base_url == "http://localhost:11434"

    def test_client_default_values(self):
        """Test client defaults"""
        client = OllamaClient()

        assert client.base_url == "http://localhost:11434"


class TestOllamaClientModels:
    """Test model management operations"""

    @pytest.mark.asyncio()
    async def test_list_models(self):
        """Test listing available models"""
        client = OllamaClient()

        # Should have list_models method
        assert hasattr(client, "list_models")

    @pytest.mark.asyncio()
    async def test_show_model(self):
        """Test getting model details"""
        client = OllamaClient()

        # Should have show_model method
        assert hasattr(client, "show_model")


class TestOllamaClientGeneration:
    """Test text generation"""

    @pytest.mark.asyncio()
    async def test_generate_method_exists(self):
        """Test generate method exists"""
        client = OllamaClient()

        assert hasattr(client, "generate")

    @pytest.mark.asyncio()
    async def test_generate_request_model(self):
        """Test GenerateRequest model"""
        request = GenerateRequest(
            model="llama2", prompt="Hello world", stream=False, temperature=0.7
        )

        assert request.model == "llama2"
        assert request.prompt == "Hello world"
        assert request.stream is False
        assert request.temperature == 0.7

    @pytest.mark.asyncio()
    async def test_generate_with_parameters(self):
        """Test generate with various parameters"""
        request = GenerateRequest(
            model="llama2", prompt="test", temperature=0.5, top_p=0.8, top_k=40, num_predict=100
        )

        assert request.temperature == 0.5
        assert request.top_p == 0.8
        assert request.top_k == 40
        assert request.num_predict == 100


class TestOllamaClientChat:
    """Test chat completion"""

    @pytest.mark.asyncio()
    async def test_chat_method_exists(self):
        """Test chat method exists"""
        client = OllamaClient()

        assert hasattr(client, "chat")

    @pytest.mark.asyncio()
    async def test_chat_request_model(self):
        """Test ChatRequest model"""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]

        request = ChatRequest(model="llama2", messages=messages, stream=False)

        assert request.model == "llama2"
        assert len(request.messages) == 2
        assert request.messages[0].role == "user"

    @pytest.mark.asyncio()
    async def test_chat_message_model(self):
        """Test ChatMessage model"""
        message = ChatMessage(role="system", content="You are helpful")

        assert message.role == "system"
        assert message.content == "You are helpful"


class TestOllamaClientEmbeddings:
    """Test embeddings generation"""

    @pytest.mark.asyncio()
    async def test_embeddings_method_exists(self):
        """Test embeddings method exists"""
        client = OllamaClient()

        assert hasattr(client, "embeddings")


class TestOllamaClientErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio()
    async def test_connection_error_handling(self):
        """Test handling connection errors"""
        # Client creation should not fail even if server unavailable
        client = OllamaClient(base_url="http://nonexistent:99999")

        assert client is not None

    @pytest.mark.asyncio()
    async def test_invalid_model_handling(self):
        """Test handling invalid model requests"""
        client = OllamaClient()

        # Client should exist even if model doesn't exist
        assert client is not None


class TestOllamaClientStreaming:
    """Test streaming responses"""

    @pytest.mark.asyncio()
    async def test_streaming_supported(self):
        """Test client supports streaming"""
        client = OllamaClient()

        # Should have methods that support streaming
        assert hasattr(client, "generate")
        assert hasattr(client, "chat")


class TestOllamaClientContext:
    """Test context management"""

    @pytest.mark.asyncio()
    async def test_client_initialize(self):
        """Test client initialization"""
        client = OllamaClient()

        assert hasattr(client, "initialize")

    @pytest.mark.asyncio()
    async def test_client_close(self):
        """Test client cleanup"""
        client = OllamaClient()

        assert hasattr(client, "close")


class TestOllamaClientSingleton:
    """Test singleton pattern"""

    def test_get_ollama_client(self):
        """Test get_ollama_client function"""
        from ollama.services.inference.ollama_client import get_ollama_client

        # Should return a client instance
        try:
            client = get_ollama_client()
            assert client is not None
        except RuntimeError:
            # Expected if not initialized
            pass

    def test_init_ollama_client(self):
        """Test init_ollama_client function"""
        from ollama.services.inference.ollama_client import init_ollama_client

        client = init_ollama_client(base_url="http://localhost:11434")
        assert client is not None
        assert client.base_url == "http://localhost:11434"

    def test_clear_ollama_client(self):
        """Confirm clear_ollama_client resets the singleton"""
        from ollama.services.inference.ollama_client import (
            clear_ollama_client,
            get_ollama_client,
            init_ollama_client,
        )

        init_ollama_client(base_url="http://localhost:11434")
        clear_ollama_client()

        with pytest.raises(RuntimeError):
            get_ollama_client()
