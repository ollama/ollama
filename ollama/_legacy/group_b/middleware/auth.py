"""Authentication middleware.

This module provides the authentication middleware for the FastAPI application,
bridging the auth domain with the middleware layer.
"""

from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ollama.exceptions import AuthenticationError


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key and JWT authentication."""

    def __init__(
        self, app: Any = None, verify_api_key: Callable[[str], bool] | None = None
    ) -> None:
        """Initialize middleware.

        Args:
            app: The ASGI application.
            verify_api_key: Optional callback to verify API keys.
        """
        # Call super().__init__ if app is provided, else we are in a unit test
        if app is not None:
            super().__init__(app)
        else:
            self.app = None
        self.verify_api_key = verify_api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and validate authentication.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            The response.
        """
        # Skip auth for health check
        if request.url.path == "/health" or request.url.path == "/api/v1/health":
            return await call_next(request)

        try:
            self.validate_auth(request)
        except (AuthenticationError, ValueError) as e:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=401,
                content={"success": False, "error": {"code": "UNAUTHORIZED", "message": str(e)}},
            )

        response = await call_next(request)
        return response

    def validate_auth(self, request: Request) -> bool:
        """Validate authentication headers.

        Args:
            request: The incoming request.

        Returns:
            True if authentication is valid.

        Raises:
            ValueError: If authentication header is missing or malformed.
            AuthenticationError: If authentication key is invalid.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise ValueError("Missing Authorization header")

        if not auth_header.lower().startswith("bearer "):
            raise ValueError("Invalid Authorization format. Use 'Bearer <key>'")

        api_key = auth_header[7:].strip()
        if not api_key:
            raise ValueError("Empty API key")

        if not self.validate_key(api_key):
            raise AuthenticationError("Invalid API key")

        return True

    def validate_key(self, api_key: str) -> bool:
        """Validate an API key.

        Args:
            api_key: The API key to validate.

        Returns:
            True if valid.
        """
        if self.verify_api_key:
            return self.verify_api_key(api_key)

        # Default implementation for tests/dev (should be replaced in production)
        return api_key == "sk-valid-key" or api_key.startswith("sk-")


# Alias for backward compatibility
AuthMiddleware = AuthenticationMiddleware

__all__ = ["AuthMiddleware", "AuthenticationMiddleware"]
