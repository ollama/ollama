"""Authentication API Schemas.

Consolidated authentication and authorization schemas for the API layer.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    """Schema for user registration."""

    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    full_name: str | None = Field(None, description="User full name")


class UserResponse(BaseModel):
    """Schema for user information response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: str
    full_name: str | None = None
    is_active: bool = True
    created_at: datetime


class TokenResponse(BaseModel):
    """Schema for JWT token response."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int = 3600


class LoginRequest(BaseModel):
    """Schema for user login."""

    username: str = Field(..., description="Username or Email")
    password: str = Field(..., description="User password")


class APIKeyCreate(BaseModel):
    """Schema for creating an API key."""

    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """Schema for API key response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    key: str | None = None  # Only provided on creation
    prefix: str
    created_at: datetime
    expires_at: datetime | None = None


class APIKeyList(BaseModel):
    """Schema for a list of API keys."""

    keys: list[APIKeyResponse]
    total: int


class PasswordChange(BaseModel):
    """Schema for password change request."""

    current_password: str
    new_password: str = Field(..., min_length=8)


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh request."""

    refresh_token: str
