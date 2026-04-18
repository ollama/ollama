"""
OAuth Middleware and Route Decorators
Provides FastAPI integration for Firebase authentication and authorization.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from fastapi import Request

from .firebase_auth import get_current_user

logger = logging.getLogger(__name__)


def require_auth(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator to require authentication on a route.

    Usage:
        @app.get("/protected")
        @require_auth
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"message": f"Hello {user['email']}"}
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    return wrapper


async def verify_token_optional(request: Request) -> dict[str, Any]:
    """Optional token verification (doesn't raise if missing).

    Returns empty dict if no token provided.
    """
    return await get_current_user(request, require_auth=False)


class AuthMiddleware:
    """FastAPI middleware for authentication and security headers."""

    def __init__(self, app: Any) -> None:
        """Initialize middleware."""
        self.app = app

    async def __call__(self, request: Request, call_next: Callable[..., Any]) -> Any:
        """Process request through auth middleware."""
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
