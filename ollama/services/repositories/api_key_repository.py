"""
API Key Repository - CRUD operations for APIKey model.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ollama.models import APIKey

from .base_repository import BaseRepository


class APIKeyRepository(BaseRepository[APIKey]):
    """Repository for APIKey model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(APIKey, session)

    async def get_by_key_hash(self, key_hash: str) -> APIKey | None:
        """Get API key by key hash.

        Args:
            key_hash: Hashed API key

        Returns:
            APIKey instance or None if not found
        """
        return await self.get_one(key_hash=key_hash)

    async def get_by_user_id(self, user_id: uuid.UUID) -> list[APIKey]:
        """Get all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of API keys
        """
        return await self.get_all(user_id=user_id)

    async def get_active_keys(self, user_id: uuid.UUID) -> list[APIKey]:
        """Get all active (non-expired) API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of active API keys
        """
        # Get all keys for user and filter by expiration
        keys = await self.get_all(user_id=user_id)
        now = datetime.now(UTC)
        return [k for k in keys if k.expires_at is None or k.expires_at > now]

    async def create_key(
        self,
        user_id: uuid.UUID,
        key_hash: str,
        name: str,
        scopes: list[str] | None = None,
        rate_limit: int = 100,
        expires_at: datetime | None = None,
    ) -> APIKey:
        """Create a new API key.

        Args:
            user_id: User ID
            key_hash: Hashed API key
            name: Human-readable key name
            scopes: List of permission scopes
            rate_limit: Requests per minute limit
            expires_at: Optional expiration datetime

        Returns:
            Created APIKey instance
        """
        key = await self.create(
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes or [],
            rate_limit=rate_limit,
            expires_at=expires_at,
        )
        await self.commit()
        return key

    async def revoke_key(self, key_id: uuid.UUID) -> APIKey | None:
        """Revoke (delete) an API key.

        Args:
            key_id: API key ID

        Returns:
            Deleted APIKey instance or None if not found
        """
        key = await self.get_by_id(key_id)
        if key:
            await self.delete(key_id)
            await self.commit()
        return key

    async def update_last_used(self, key_id: uuid.UUID) -> APIKey | None:
        """Update last_used timestamp for a key.

        Args:
            key_id: API key ID

        Returns:
            Updated APIKey instance or None if not found
        """
        key = await self.update(key_id, last_used=datetime.now(UTC))
        if key:
            await self.commit()
        return key

    async def search_keys(self, user_id: uuid.UUID, query: str) -> list[APIKey]:
        """Search API keys by name.

        Args:
            user_id: User ID
            query: Search query

        Returns:
            List of matching keys
        """
        keys = await self.get_all(user_id=user_id)
        return [k for k in keys if query.lower() in k.name.lower()]

    async def has_scope(self, key_hash: str, required_scope: str) -> bool:
        """Check if API key has required scope.

        Args:
            key_hash: Hashed API key
            required_scope: Required scope

        Returns:
            True if key has scope, False otherwise
        """
        key = await self.get_by_key_hash(key_hash)
        if not key:
            return False

        # Check if expired
        if key.expires_at and key.expires_at < datetime.now(UTC):
            return False

        # Check scope
        return required_scope in key.scopes

    async def is_rate_limited(self, key_hash: str, current_requests: int) -> bool:
        """Check if API key is rate limited.

        Args:
            key_hash: Hashed API key
            current_requests: Current request count

        Returns:
            True if rate limited, False otherwise
        """
        key = await self.get_by_key_hash(key_hash)
        if not key or key.rate_limit is None:
            return False
        return bool(current_requests >= key.rate_limit)

    async def verify_and_get_user(self, key_hash: str) -> APIKey | None:
        """Verify API key and return associated user record.

        Args:
            key_hash: Hashed API key

        Returns:
            APIKey record or None
        """
        key = await self.get_by_key_hash(key_hash)
        if not key:
            return None

        if not key.is_active:
            return None

        # Check expiration
        if key.expires_at and key.expires_at < datetime.now(UTC):
            return None

        return key

    async def mark_used(self, key_id: uuid.UUID) -> None:
        """Mark API key as used.

        Args:
            key_id: API key ID
        """
        await self.update_last_used(key_id)
