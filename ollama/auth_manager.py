"""
Authentication and Authorization wiring for FastAPI dependencies.

This module provides dependency helpers that wire together the authentication
manager, repositories, and FastAPI security primitives. Core authentication
logic (AuthManager) and exceptions live in dedicated modules to comply with
single-class-per-file standards.
"""

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from fastapi import Depends, HTTPException, Security, status
    from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
else:
    try:
        from fastapi import Depends, HTTPException, Security, status
        from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
    except Exception:  # pragma: no cover
        # Fallbacks for environments without FastAPI
        def Depends(dependency: Callable[..., Any] | None = None, *, use_cache: bool = True) -> Any:
            return None

        class HTTPException(Exception):
            def __init__(
                self,
                status_code: int,
                detail: Any = None,
                headers: dict[str, Any] | None = None,
            ) -> None:
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail
                self.headers = headers

        def Security(
            dependency: Callable[..., Any] | None = None,
            *,
            scopes: Sequence[str] | None = None,
            use_cache: bool = True,
        ) -> Any:
            return None

        class status:
            HTTP_401_UNAUTHORIZED = 401
            HTTP_403_FORBIDDEN = 403

        class APIKeyHeader:
            def __init__(self, name: str, *, auto_error: bool = True) -> None:
                pass

        class HTTPAuthorizationCredentials:
            def __init__(self, scheme: str, credentials: str) -> None:
                self.scheme = scheme
                self.credentials = credentials

        class HTTPBearer:
            def __init__(self, *, auto_error: bool = True) -> None:
                pass


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
else:
    try:
        from sqlalchemy.ext.asyncio import AsyncSession
    except Exception:  # pragma: no cover

        class AsyncSession:
            pass


from ollama.auth import AuthManager
from ollama.config import get_settings
from ollama.exceptions import AuthenticationError
from ollama.models import User
from ollama.repositories import RepositoryFactory
from ollama.services import get_db

logger = logging.getLogger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Global auth manager instance
_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """
    Get global auth manager instance

    Returns:
        AuthManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        settings = get_settings()
        jwt_secret = getattr(settings, "jwt_secret", "development-secret-change-me")
        _auth_manager = AuthManager(secret_key=jwt_secret)
    return _auth_manager


# FastAPI Dependencies
async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current user from JWT token

    Args:
        credentials: HTTP authorization credentials
        db: Database session

    Returns:
        User model instance

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        auth_manager = get_auth_manager()
        payload = auth_manager.decode_token(credentials.credentials)

        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        user_id = UUID(str(sub))

        # Get user from database
        repo_factory = RepositoryFactory(db)
        user_repo = repo_factory.get_user_repository()
        user = await user_repo.get_by_id(user_id)

        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive"
            )

        return user

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
        ) from e


async def get_current_user_from_api_key(
    api_key: str | None = Security(api_key_header), db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current user from API key

    Args:
        api_key: API key from X-API-Key header
        db: Database session

    Returns:
        User model instance

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    try:
        # Get API key from database
        repo_factory = RepositoryFactory(db)
        api_key_repo = repo_factory.get_api_key_repository()

        # Verify and get user from API key
        key_record = await api_key_repo.verify_and_get_user(api_key)

        if key_record is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        # Mark API key as used
        await api_key_repo.mark_used(key_record.id)

        # Get user
        user_repo = repo_factory.get_user_repository()
        user = await user_repo.get_by_id(key_record.user_id)

        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive"
            )

        return user

    except Exception as e:
        logger.error(f"API key authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        ) from e


async def get_current_user(
    bearer_user: User = Depends(get_current_user_from_token),
    api_key_user: User = Depends(get_current_user_from_api_key),
) -> User:
    """
    Get current user from either JWT token or API key

    Tries bearer token first, falls back to API key

    Returns:
        User model instance
    """
    # This will never actually be called because the dependencies
    # will handle the authentication. This is just a marker.
    return bearer_user or api_key_user


def require_admin(current_user: User = Depends(get_current_user_from_token)) -> User:
    """
    Require admin privileges

    Args:
        current_user: Current authenticated user

    Returns:
        User model instance

    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_user


# Optional authentication (doesn't fail if not authenticated)
async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    Get current user if authenticated, None otherwise

    Useful for endpoints that work both authenticated and unauthenticated

    Returns:
        User model instance or None
    """
    try:
        if credentials:
            return await get_current_user_from_token(credentials, db)
        elif api_key:
            return await get_current_user_from_api_key(api_key, db)
    except HTTPException:
        pass

    return None
