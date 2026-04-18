"""Authentication and authorization manager.

Provides JWT token management, API key validation, and user authentication.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import bcrypt
import jwt

from ollama.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class AuthManager:
    """Authentication and authorization manager.

    Handles:
    - Password hashing and verification
    - JWT token generation and validation
    - API key validation
    - User session management
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256") -> None:
        """Initialize auth manager.

        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash.

        Args:
            password: Plain text password
            hashed: Hashed password

        Returns:
            True if password matches
        """
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    def create_access_token(
        self, user_id: UUID, username: str, expires_delta: timedelta | None = None
    ) -> str:
        """Create JWT access token.

        Args:
            user_id: User UUID
            username: Username
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        if expires_delta is None:
            expires_delta = timedelta(hours=1)

        expire = datetime.now(UTC) + expires_delta

        payload = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "access",
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def create_refresh_token(self, user_id: UUID, expires_delta: timedelta | None = None) -> str:
        """Create JWT refresh token.

        Args:
            user_id: User UUID
            expires_delta: Token expiration time

        Returns:
            Encoded JWT refresh token
        """
        if expires_delta is None:
            expires_delta = timedelta(days=7)

        expire = datetime.now(UTC) + expires_delta

        payload = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh",
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def decode_token(self, token: str) -> dict[str, Any]:
        """Decode and validate JWT token.

        Args:
            token: Encoded JWT token

        Returns:
            Token payload

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired") from None
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e!s}") from e
        except ValueError as e:
            raise AuthenticationError(f"Invalid token: {e!s}") from e

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key.

        Args:
            api_key: Plain API key

        Returns:
            Hashed API key
        """
        return self.hash_password(api_key)

    def verify_api_key(self, api_key: str, hashed: str) -> bool:
        """Verify an API key against its hash.

        Args:
            api_key: Plain API key
            hashed: Hashed API key

        Returns:
            True if API key matches
        """
        return self.verify_password(api_key, hashed)
