"""Pytest configuration and fixtures for Ollama test suite.

This module provides shared fixtures used across all test modules.
It is automatically discovered and loaded by pytest.
"""

import random
import string
from unittest.mock import MagicMock, Mock

import pytest
from ollama.exceptions import AuthenticationError


@pytest.fixture()
def auth_manager() -> Mock:
    """Provide a mocked AuthManager for testing.

    Returns:
        Mocked AuthManager instance with password hashing support
    """
    manager = Mock()

    # Mock hash_password to return a realistic hash with random salt
    def mock_hash(pwd: str) -> str:
        """Hash a password with random salt for uniqueness."""
        salt = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        return f"$2b$12${salt}_hash_of_{pwd}"

    def mock_verify(pwd: str, hashed: str) -> bool:
        """Verify password matches hash."""
        # Extract original password from hash format if it contains the same pwd
        return pwd in hashed or hashed.endswith(pwd)

    # Mock JWT token operations
    def mock_create_access_token(
        user_id: any, username: any = None, expires_delta: any = None
    ) -> str:
        """Create a mock JWT access token."""
        # Mock JWT format: header.payload.signature
        import base64
        import json

        header = base64.b64encode(b'{"alg":"HS256","typ":"JWT"}').decode()
        payload_data = {"sub": str(user_id), "username": username, "type": "access"}
        payload = base64.b64encode(json.dumps(payload_data).encode()).decode()
        signature = base64.b64encode(b"mock_signature").decode()
        expired_marker = ""
        if expires_delta is not None and hasattr(expires_delta, "total_seconds"):
            if expires_delta.total_seconds() < 0:
                expired_marker = "-expired"
        return f"{header}.{payload}.{signature}{expired_marker}"

    def mock_create_refresh_token(user_id: any) -> str:
        """Create a mock refresh token."""
        import base64
        import json

        header = base64.b64encode(b'{"alg":"HS256","typ":"JWT"}').decode()
        payload_data = {"sub": str(user_id), "type": "refresh"}
        payload = base64.b64encode(json.dumps(payload_data).encode()).decode()
        signature = base64.b64encode(b"mock_refresh_sig").decode()
        return f"{header}.{payload}.{signature}"

    def mock_decode_token(token: str) -> dict[str, any]:
        """Decode a mock token."""
        import base64
        import json

        if "invalid" in token:
            raise AuthenticationError("Invalid token")
        if "expired" in token:
            raise AuthenticationError("Token has expired")
        if token.count(".") != 2:
            raise ValueError("Invalid token format")
        try:
            # Extract and decode payload
            parts = token.split(".")
            if len(parts) != 3:
                raise ValueError("Invalid token format")
            # Add padding if needed
            payload_part = parts[1]
            padding = 4 - len(payload_part) % 4
            if padding != 4:
                payload_part += "=" * padding
            payload_json = base64.b64decode(payload_part).decode()
            return json.loads(payload_json)
        except Exception as e:
            raise AuthenticationError(f"Token decode failed: {e}") from e

    # Mock API key operations
    def mock_hash_api_key(key: str) -> str:
        """Hash an API key."""
        salt = "".join(random.choices(string.ascii_letters, k=4))
        return f"hashed_{salt}_{key}"

    def mock_verify_api_key(key: str, hashed: str) -> bool:
        """Verify an API key."""
        return key in hashed

    manager.hash_password = Mock(side_effect=mock_hash)
    manager.verify_password = Mock(side_effect=mock_verify)
    manager.create_access_token = Mock(side_effect=mock_create_access_token)
    manager.create_refresh_token = Mock(side_effect=mock_create_refresh_token)
    manager.decode_token = Mock(side_effect=mock_decode_token)
    manager.hash_api_key = Mock(side_effect=mock_hash_api_key)
    manager.verify_api_key = Mock(side_effect=mock_verify_api_key)
    return manager


@pytest.fixture()
def mock_request() -> Mock:
    """Provide a mock FastAPI Request object.

    Returns:
        Mock request with common attributes
    """
    request = Mock()
    request.headers = {}
    return request


@pytest.fixture()
def mock_settings() -> Mock:
    """Provide mock application settings.

    Returns:
        Mock settings object with common config values
    """
    settings = Mock()
    settings.firebase_enabled = True
    settings.api_key_auth_enabled = True
    settings.cors_origins = ["*"]
    settings.cors_allow_credentials = True
    settings.cors_expose_headers = ["X-Request-ID"]
    settings.host = "127.0.0.1"  # Use localhost for tests
    settings.port = 8000
    settings.workers = 1
    settings.log_level = "info"
    settings.public_url = "http://localhost:8000"
    return settings


@pytest.fixture()
def mock_firebase_user() -> Mock:
    """Provide a mock Firebase user object.

    Returns:
        Mock Firebase user with standard attributes
    """
    user = Mock()
    user.uid = "test-uid-123"
    user.email = "test@example.com"
    user.display_name = "Test User"
    user.email_verified = True
    user.custom_claims = {"role": "user"}
    return user


@pytest.fixture()
def mock_firebase_auth() -> MagicMock:
    """Provide a mocked Firebase authentication module.

    Returns:
        Mock firebase_auth module with common methods
    """
    auth = MagicMock()
    auth.verify_id_token = Mock(return_value={"sub": "test-user", "email": "test@example.com"})
    auth.get_user = Mock(return_value=Mock(uid="test-uid", email="test@example.com"))
    auth.get_user_by_email = Mock(return_value=Mock(uid="test-uid", email="test@example.com"))
    auth.create_user = Mock(return_value=Mock(uid="new-uid", email="new@example.com"))
    auth.UserNotFoundError = Exception
    auth.ExpiredSignInError = Exception
    auth.RevokedSignInError = Exception
    auth.InvalidIdTokenError = Exception
    return auth


@pytest.fixture()
def mock_ollama_client() -> Mock:
    """Provide a mocked Ollama client.

    Returns:
        Mock Ollama client with model and inference methods
    """
    client = Mock()
    client.list = Mock(return_value={"models": [{"name": "llama2"}]})
    client.show = Mock(return_value={"name": "llama2", "size": "7b"})
    client.generate = Mock(return_value={"response": "test response"})
    client.embeddings = Mock(return_value={"embedding": [0.1, 0.2, 0.3]})
    client.pull = Mock(return_value={"status": "success"})
    # Add timeout attributes for client config tests
    client.timeout = 30
    return client


@pytest.fixture()
def mock_metrics_registry() -> Mock:
    """Provide a mocked Prometheus metrics registry.

    Returns:
        Mock registry with common metric collectors
    """
    registry = Mock()
    registry.collect = Mock(return_value=[])
    registry.register = Mock()
    registry.unregister = Mock()

    # Add mock metrics
    auth_metrics = {
        "auth_attempts_total": 0.0,
        "auth_failures_total": 0.0,
        "auth_successes_total": 0.0,
    }
    http_metrics = {
        "http_requests": 0.0,
        "cache_hits_total": 0.0,
        "cache_misses_total": 0.0,
        "rate_limit_exceeded_total": 0.0,
        "requests_total": 0.0,
    }

    all_metrics = {**auth_metrics, **http_metrics}
    registry.collect = Mock(return_value=[Mock(name=k, value=v) for k, v in all_metrics.items()])

    def mock_get_summary() -> dict[str, any]:
        """Get metrics summary."""
        return {**auth_metrics, **http_metrics}

    registry.get_summary = Mock(side_effect=mock_get_summary)
    return registry
