"""
Firebase OAuth Authentication Module
Implements Firebase JWT token verification and role-based access control.
Mirrors Gov-AI-Scout authentication pattern for consistency across services.

CRITICAL: This implementation mirrors Gov-AI-Scout for seamless client compatibility.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import Depends, HTTPException, Request, status

logger = logging.getLogger(__name__)

# Firebase configuration
_firebase_initialized = False
_firebase_app: Any = None

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth
    from firebase_admin import credentials
except ImportError:
    logger.warning(
        "firebase-admin not installed. OAuth will be disabled. "
        "Install with: pip install firebase-admin"
    )


def init_firebase(credentials_path: str | None = None) -> None:
    """Initialize Firebase Admin SDK.

    Args:
        credentials_path: Path to Firebase service account JSON.
                         If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.

    Raises:
        RuntimeError: If Firebase initialization fails.
    """
    global _firebase_initialized, _firebase_app

    if _firebase_initialized:
        return

    try:
        if credentials_path:
            creds = credentials.Certificate(credentials_path)
            _firebase_app = firebase_admin.initialize_app(creds)
        else:
            # Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
            _firebase_app = firebase_admin.initialize_app()

        logger.info("✅ Firebase initialized successfully")
        _firebase_initialized = True

    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase: {e}")
        raise RuntimeError(f"Firebase initialization failed: {e}") from e


def _extract_token_from_header(auth_header: str) -> tuple[bool, str]:
    """Extract and validate Authorization header.

    Args:
        auth_header: Authorization header value

    Returns:
        Tuple of (is_valid, token)
    """
    if not auth_header.startswith("Bearer "):
        return False, ""
    return True, auth_header[7:]  # Remove "Bearer " prefix


def _check_firebase_available(require_auth: bool) -> None:
    """Check if Firebase is initialized.

    Args:
        require_auth: If True, raise error if not initialized

    Raises:
        HTTPException: 503 if Firebase not available and require_auth=True
    """
    if not _firebase_initialized:
        if require_auth:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available",
            )


def _verify_firebase_token(token: str) -> dict[str, Any]:
    """Verify Firebase JWT token.

    Args:
        token: JWT token to verify

    Returns:
        Decoded token claims

    Raises:
        Exception: If token verification fails
    """
    decoded_token = cast(dict[str, Any], firebase_auth.verify_id_token(token))
    logger.debug(f"✅ Token verified for user: {decoded_token.get('email')}")
    return decoded_token


def _handle_token_verification_error(error: Exception, require_auth: bool) -> dict[str, Any]:
    """Handle token verification errors.

    Args:
        error: The verification error
        require_auth: If True, raise HTTPException; if False, return empty dict

    Returns:
        Empty dict if not require_auth

    Raises:
        HTTPException: If require_auth=True
    """
    if isinstance(error, firebase_auth.ExpiredSignInError):
        logger.warning("⚠️ Expired authentication token")
        if require_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            ) from None
        return {}

    if isinstance(error, firebase_auth.RevokedSignInError):
        logger.warning("⚠️ Revoked authentication token - user must re-authenticate")
        if require_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication revoked. Please sign in again.",
            ) from None
        return {}

    if isinstance(error, firebase_auth.InvalidIdTokenError):
        logger.warning("⚠️ Invalid authentication token")
        if require_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from None
        return {}

    logger.error(f"❌ Token verification failed: {error}")
    if require_auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        ) from error
    return {}


async def get_current_user(request: Request, require_auth: bool = True) -> dict[str, Any]:
    """Extract and verify Firebase JWT token from request.

    Args:
        request: FastAPI request object
        require_auth: If True, raise error if auth fails. If False, return empty dict.

    Returns:
        User claims from verified JWT token

    Raises:
        HTTPException: 401 if token missing, invalid, or revoked
    """
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization", "")
    is_valid, token = _extract_token_from_header(auth_header)

    if not is_valid:
        if require_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
            )
        return {}

    # Check Firebase is initialized
    _check_firebase_available(require_auth)

    # Verify token
    try:
        return _verify_firebase_token(token)

    except Exception as e:
        return _handle_token_verification_error(e, require_auth)


def require_role(
    allowed_roles: list[str],
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Dependency to check if user has required role.

    Args:
        allowed_roles: List of role names that are authorized

    Returns:
        Dependency function for route protection
    """

    async def role_checker(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        """Verify user has required role."""
        user_roles = cast(list[str], user.get("roles", []))

        if not any(role in user_roles for role in allowed_roles):
            logger.warning(
                f"⚠️ User {user.get('email')} lacks required roles. "
                f"Has: {user_roles}, Required: {allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {allowed_roles}",
            )

        return user

    return role_checker


def require_root_admin(
    root_admin_email: str,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Dependency to check if user is root admin (specific email).

    Args:
        root_admin_email: Email address of root admin

    Returns:
        Dependency function for admin-only route protection
    """

    async def admin_checker(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        """Verify user is root admin."""
        user_email = user.get("email", "")

        if user_email != root_admin_email:
            logger.warning(f"⚠️ Non-admin {user_email} attempted admin operation")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
            )

        return user

    return admin_checker


def revoke_user_tokens(uid: str) -> None:
    """Revoke all tokens for a user (force re-authentication).

    Args:
        uid: Firebase user ID

    Note:
        This does NOT revoke existing tokens immediately (max 5 min delay).
        Existing valid tokens will be rejected after TTL expires.
    """
    if not _firebase_initialized:
        logger.warning("Firebase not initialized; token revocation skipped")
        return

    try:
        firebase_auth.revoke_refresh_tokens(uid)
        logger.info(f"✅ Revoked tokens for user: {uid}")

    except Exception as e:
        logger.error(f"❌ Failed to revoke tokens for {uid}: {e}")
        raise


def get_or_create_user(email: str, display_name: str | None = None) -> dict[str, Any]:
    """Get or create Firebase user.

    Args:
        email: User email address
        display_name: Optional display name

    Returns:
        User data dict

    Raises:
        RuntimeError: If Firebase not initialized or operation fails
    """
    if not _firebase_initialized:
        raise RuntimeError("Firebase not initialized")

    try:
        # Try to get existing user
        user = firebase_auth.get_user_by_email(email)
        logger.debug(f"✅ Found existing user: {email}")
        return {"uid": user.uid, "email": user.email, "display_name": user.display_name}

    except firebase_auth.UserNotFoundError:
        # Create new user
        try:
            user = firebase_auth.create_user(email=email, display_name=display_name)
            logger.info(f"✅ Created new user: {email}")
            return {"uid": user.uid, "email": user.email, "display_name": user.display_name}

        except Exception as e:
            logger.error(f"❌ Failed to create user {email}: {e}")
            raise RuntimeError(f"User creation failed: {e}") from e

    except Exception as e:
        logger.error(f"❌ Failed to get user {email}: {e}")
        raise RuntimeError(f"User lookup failed: {e}") from e
