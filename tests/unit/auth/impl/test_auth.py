"""
Tests for Authentication Module
Tests Firebase OAuth, JWT token validation, role-based access control
"""

from datetime import timedelta
from uuid import uuid4

import pytest

from ollama.auth import (
    get_current_user,
    require_role,
    require_root_admin,
    revoke_user_tokens,
)
from ollama.auth_manager import (
    AuthManager,
    AuthenticationError,
    get_auth_manager,
)


def _random_value(prefix: str) -> str:
    """Generate a deterministic-looking but non-secret test value."""

    return f"{prefix}-{uuid4().hex}"


class TestFirebaseAuth:
    """Test Firebase authentication functions"""

    @pytest.mark.asyncio
    async def test_get_current_user_requires_token(self):
        """Test that get_current_user requires valid token"""
        # This will be tested with mock Firebase credentials
        pass

    def test_hash_password(self, auth_manager):
        """Test password hashing"""
        password = _random_value("password")
        hashed = auth_manager.hash_password(password)

        # Verify hashed password is different from original
        assert hashed != password
        assert len(hashed) > len(password)

    def test_verify_password(self, auth_manager):
        """Test password verification"""
        password = _random_value("password")
        hashed = auth_manager.hash_password(password)

        # Verify correct password
        assert auth_manager.verify_password(password, hashed)

        # Verify incorrect password
        assert not auth_manager.verify_password("wrong-password", hashed)

    def test_password_hashes_are_unique(self, auth_manager):
        """Test that same password produces different hashes"""
        password = _random_value("password")
        hash1 = auth_manager.hash_password(password)
        hash2 = auth_manager.hash_password(password)

        # Different hashes for same password (due to random salt)
        assert hash1 != hash2
        # But both verify correctly
        assert auth_manager.verify_password(password, hash1)
        assert auth_manager.verify_password(password, hash2)

    def test_create_access_token(self, auth_manager):
        """Test JWT access token creation"""
        user_id = uuid4()
        username = _random_value("user")

        token = auth_manager.create_access_token(user_id, username)

        # Token should be a string
        assert isinstance(token, str)
        assert len(token) > 0
        # JWT has three parts separated by dots
        assert token.count(".") == 2

    def test_create_refresh_token(self, auth_manager):
        """Test JWT refresh token creation"""
        user_id = uuid4()

        token = auth_manager.create_refresh_token(user_id)

        assert isinstance(token, str)
        assert len(token) > 0
        assert token.count(".") == 2

    def test_decode_access_token(self, auth_manager):
        """Test JWT access token decoding"""
        user_id = uuid4()
        username = _random_value("user")

        token = auth_manager.create_access_token(user_id, username)
        payload = auth_manager.decode_token(token)

        assert str(user_id) == payload["sub"]
        assert username == payload["username"]
        assert payload["type"] == "access"

    def test_decode_refresh_token(self, auth_manager):
        """Test JWT refresh token decoding"""
        user_id = uuid4()

        token = auth_manager.create_refresh_token(user_id)
        payload = auth_manager.decode_token(token)

        assert str(user_id) == payload["sub"]
        assert payload["type"] == "refresh"

    def test_decode_invalid_token(self, auth_manager):
        """Test decoding invalid token raises error"""
        with pytest.raises(AuthenticationError):
            auth_manager.decode_token("invalid.token.here")

    def test_decode_expired_token(self, auth_manager):
        """Test decoding expired token raises error"""
        # Create token with very short expiration
        user_id = uuid4()
        token = auth_manager.create_access_token(
            user_id, "testuser", expires_delta=timedelta(seconds=-1)  # Already expired
        )

        with pytest.raises(AuthenticationError):
            auth_manager.decode_token(token)

    def test_hash_api_key(self, auth_manager):
        """Test API key hashing"""
        api_key = _random_value("api-key")
        hashed = auth_manager.hash_api_key(api_key)

        assert hashed != api_key
        assert len(hashed) > len(api_key)

    def test_verify_api_key(self, auth_manager):
        """Test API key verification"""
        api_key = _random_value("api-key")
        hashed = auth_manager.hash_api_key(api_key)

        assert auth_manager.verify_api_key(api_key, hashed)
        assert not auth_manager.verify_api_key("wrong_key", hashed)

    def test_token_expiration(self, auth_manager):
        """Test token expiration"""
        user_id = uuid4()

        # Create token with 1 second expiration
        token = auth_manager.create_access_token(
            user_id, "testuser", expires_delta=timedelta(seconds=1)
        )

        # Should be valid immediately
        payload = auth_manager.decode_token(token)
        assert payload is not None

    def test_different_secrets_produce_different_tokens(self):
        """Test that different secrets produce different tokens"""
        manager1 = AuthManager(secret_key=_random_value("secret-one"))
        manager2 = AuthManager(secret_key=_random_value("secret-two"))

        user_id = uuid4()
        token1 = manager1.create_access_token(user_id, "testuser")
        token2 = manager2.create_access_token(user_id, "testuser")

        assert token1 != token2

        # token1 can be decoded by manager1
        payload1 = manager1.decode_token(token1)
        assert payload1 is not None

        # token1 cannot be decoded by manager2
        with pytest.raises(AuthenticationError):
            manager2.decode_token(token1)


class TestGetAuthManager:
    """Test auth manager singleton"""

    def test_get_auth_manager_returns_instance(self):
        """Test get_auth_manager returns AuthManager"""
        manager = get_auth_manager()
        assert isinstance(manager, AuthManager)

    def test_get_auth_manager_caches_instance(self):
        """Test get_auth_manager caches the instance"""
        manager1 = get_auth_manager()
        manager2 = get_auth_manager()
        assert manager1 is manager2
