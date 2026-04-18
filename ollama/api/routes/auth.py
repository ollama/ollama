"""
Authentication Routes
Handles user login, registration, API key management
"""

import logging
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ollama.api.schemas.auth import (
    APIKeyCreate,
    APIKeyList,
    APIKeyResponse,
    LoginRequest,
    PasswordChange,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from ollama.auth_manager import (
    get_auth_manager,
    get_current_user_from_token,
    require_admin,
)
from ollama.models import APIKey, User
from ollama.repositories import RepositoryFactory
from ollama.services import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)) -> UserResponse:
    """
    Register a new user

    Creates a new user account with hashed password.
    """
    repo_factory = RepositoryFactory(db)
    user_repo = repo_factory.get_user_repository()
    auth_manager = get_auth_manager()

    # Check if username exists
    existing_user = await user_repo.get_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists"
        )

    # Check if email exists
    existing_email = await user_repo.get_by_email(user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Hash password
    hashed_password = auth_manager.hash_password(user_data.password)

    # Create user
    created_user = await user_repo.create(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_admin=False,
    )
    logger.info(f"User registered: {created_user.username}")

    return UserResponse.model_validate(created_user)


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    """
    Login with username/password

    Returns JWT access and refresh tokens.
    """
    repo_factory = RepositoryFactory(db)
    user_repo = repo_factory.get_user_repository()
    auth_manager = get_auth_manager()

    # Get user by username
    user = await user_repo.get_by_username(credentials.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
        )

    # Verify password
    if not auth_manager.verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    # Update last login
    await user_repo.update(user.id, last_login=datetime.now(UTC))

    # Create tokens
    access_token = auth_manager.create_access_token(
        user_id=user.id, username=user.username, expires_delta=timedelta(hours=1)
    )

    refresh_token = auth_manager.create_refresh_token(
        user_id=user.id, expires_delta=timedelta(days=7)
    )

    logger.info(f"User logged in: {user.username}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=3600,  # 1 hour
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Refresh access token using refresh token
    """
    auth_manager = get_auth_manager()

    try:
        # Decode refresh token
        payload = auth_manager.decode_token(request.refresh_token)

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )

        user_id = UUID(payload.get("sub"))

        # Get user
        repo_factory = RepositoryFactory(db)
        user_repo = repo_factory.get_user_repository()
        user = await user_repo.get_by_id(user_id)

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive"
            )

        # Create new access token
        access_token = auth_manager.create_access_token(
            user_id=user.id,
            username=user.username,
            expires_delta=timedelta(hours=1),
        )

        return TokenResponse(access_token=access_token, token_type="bearer", expires_in=3600)

    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        ) from e


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: User = Depends(get_current_user_from_token),
) -> UserResponse:
    """
    Get current authenticated user
    """
    return UserResponse.model_validate(current_user)


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user_from_token),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """
    # Change user password
    """
    auth_manager = get_auth_manager()

    # Verify old password
    if not auth_manager.verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect password")

    # Hash new password
    new_hash = auth_manager.hash_password(password_data.new_password)

    # Update password
    repo_factory = RepositoryFactory(db)
    user_repo = repo_factory.get_user_repository()
    await user_repo.update(current_user.id, password_hash=new_hash, updated_at=datetime.now(UTC))

    logger.info(f"Password changed for user: {current_user.username}")

    return {"message": "Password changed successfully"}


# API Key Management
@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user_from_token),
    db: AsyncSession = Depends(get_db),
) -> APIKeyResponse:
    """
    Create a new API key

    Returns the plain text key ONCE - store it securely!
    """
    auth_manager = get_auth_manager()

    # Generate random API key
    plain_key = f"sk_{secrets.token_urlsafe(32)}"
    key_hash = auth_manager.hash_api_key(plain_key)
    key_prefix = plain_key[:8]

    # Calculate expiration
    expires_at = None
    if key_data.expires_in_days:
        expires_at = datetime.now(UTC) + timedelta(days=key_data.expires_in_days)

    # Create API key record
    repo_factory = RepositoryFactory(db)
    api_key_repo = repo_factory.get_api_key_repository()
    created_key = await api_key_repo.create(
        name=key_data.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        user_id=current_user.id,
        is_active=True,
        expires_at=expires_at,
    )

    logger.info(f"API key created for user {current_user.username}: {key_data.name}")

    # Return key with plain text (only time it's visible)
    response = APIKeyResponse.model_validate(created_key)
    response.key = plain_key

    return response


@router.get("/api-keys", response_model=APIKeyList)
async def list_api_keys(
    current_user: User = Depends(get_current_user_from_token), db: AsyncSession = Depends(get_db)
) -> APIKeyList:
    """
    List all API keys for current user
    """
    repo_factory = RepositoryFactory(db)
    api_key_repo = repo_factory.get_api_key_repository()

    keys: list[APIKey] = await api_key_repo.get_by_user_id(current_user.id)

    return APIKeyList(keys=[APIKeyResponse.model_validate(k) for k in keys], total=len(keys))


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: UUID,
    current_user: User = Depends(get_current_user_from_token),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """
    Revoke (deactivate) an API key
    """
    repo_factory = RepositoryFactory(db)
    api_key_repo = repo_factory.get_api_key_repository()

    # Get key
    key = await api_key_repo.get_by_id(key_id)
    if not key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    # Verify ownership
    if key.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to revoke this API key"
        )

    # Revoke key
    await api_key_repo.update(key_id, is_active=False)

    logger.info(f"API key revoked: {key.name} for user {current_user.username}")

    return {"message": "API key revoked successfully"}


# Admin endpoints
@router.get("/users", response_model=list[UserResponse])
async def list_users(
    admin_user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> list[UserResponse]:
    """
    List all users (admin only)
    """
    repo_factory = RepositoryFactory(db)
    user_repo = repo_factory.get_user_repository()

    users: list[User] = await user_repo.get_all()
    return [UserResponse.model_validate(u) for u in users]


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: UUID, admin_user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> dict[str, str]:
    """
    Deactivate a user account (admin only)
    """
    repo_factory = RepositoryFactory(db)
    user_repo = repo_factory.get_user_repository()

    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await user_repo.update(user_id, is_active=False, updated_at=datetime.now(UTC))

    logger.info(f"User deactivated by admin: {user.username}")

    return {"message": "User deactivated successfully"}
