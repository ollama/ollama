"""Test suite for Ollama client library."""

from uuid import uuid4

from ollama.client import Client


class TestClientInitialization:
    """Test client initialization with various endpoints."""

    def test_client_default_localhost(self):
        """Test client defaults to localhost for development."""
        client = Client()
        assert "localhost" in client.base_url or "127.0.0.1" in client.base_url

    def test_client_custom_url(self):
        """Test client with custom URL."""
        client = Client(base_url="http://example.com:9000")
        assert client.base_url == "http://example.com:9000"

    def test_client_public_endpoint(self):
        """Test client with public endpoint (elevatediq.ai)."""
        client = Client(base_url="https://elevatediq.ai/ollama")
        assert client.base_url == "https://elevatediq.ai/ollama"

    def test_client_with_api_key(self):
        """Test client initialization with API key."""
        api_key = f"key-{uuid4().hex}"
        client = Client(base_url="https://elevatediq.ai/ollama", api_key=api_key)
        assert client.api_key == api_key
        assert "X-API-Key" in client.client.headers

    def test_client_url_normalization(self):
        """Test client URL normalization (removes trailing slash)."""
        client = Client(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_client_from_env_public_url(self, monkeypatch):
        """Test client respects OLLAMA_PUBLIC_URL environment variable."""
        monkeypatch.setenv("OLLAMA_PUBLIC_URL", "https://elevatediq.ai/ollama")
        client = Client()
        assert client.base_url == "https://elevatediq.ai/ollama"

    def test_client_from_env_api_key(self, monkeypatch):
        """Test client respects OLLAMA_API_KEY environment variable."""
        api_key = f"env-api-key-{uuid4().hex}"
        monkeypatch.setenv("OLLAMA_API_KEY", api_key)
        client = Client()
        assert client.api_key == api_key

    def test_client_prioritizes_explicit_api_key(self, monkeypatch):
        """Test explicit API key takes priority over environment."""
        monkeypatch.setenv("OLLAMA_API_KEY", f"env-key-{uuid4().hex}")
        explicit_key = f"explicit-key-{uuid4().hex}"
        client = Client(api_key=explicit_key)
        assert client.api_key == explicit_key


class TestClientConfiguration:
    """Test client configuration options."""

    def test_client_supports_bearer_token(self):
        """Test client supports Bearer token authentication."""
        bearer_token = f"bearer-{uuid4().hex}"
        client = Client(base_url="https://elevatediq.ai/ollama", api_key=bearer_token)
        # Both X-API-Key and Authorization headers should be set
        assert "X-API-Key" in client.client.headers
        assert "Authorization" in client.client.headers

    def test_client_has_user_agent(self):
        """Test client includes User-Agent header."""
        client = Client()
        assert "User-Agent" in client.client.headers
        assert "ollama-client" in client.client.headers["User-Agent"]

    def test_client_timeout_configured(self):
        """Test client has appropriate timeout."""
        client = Client()
        assert client.client.timeout is not None
