"""
User Repository - CRUD operations for User model.
"""

import uuid
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ollama.models import User

from .base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    """Repository for User model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username.

        Args:
            username: Username to search for

        Returns:
            User instance or None if not found
        """
        return await self.get_one(username=username)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address.

        Args:
            email: Email to search for

        Returns:
            User instance or None if not found
        """
        return await self.get_one(email=email)

    async def get_active_users(self) -> list[User]:
        """Get all active users.

        Returns:
            List of active users
        """
        return await self.get_all(is_active=True)

    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        is_active: bool = True,
        preferences: dict[str, Any] | None = None,
    ) -> User:
        """Create a new user.

        Args:
            username: Unique username
            email: User email
            password_hash: Hashed password
            is_active: Whether user is active
            preferences: User preferences dict

        Returns:
            Created user instance
        """
        user = await self.create(
            username=username,
            email=email,
            password_hash=password_hash,
            is_active=is_active,
            preferences=preferences or {},
        )
        await self.commit()
        return user

    async def deactivate_user(self, user_id: uuid.UUID) -> Optional[User]:
        """Deactivate a user.

        Args:
            user_id: User ID to deactivate

        Returns:
            Updated user instance or None if not found
        """
        user = await self.update(user_id, is_active=False)
        if user:
            await self.commit()
        return user

    async def update_user_preferences(
        self, user_id: uuid.UUID, preferences: dict[str, Any]
    ) -> Optional[User]:
        """Update user preferences.

        Args:
            user_id: User ID
            preferences: New preferences dict

        Returns:
            Updated user instance or None if not found
        """
        user = await self.update(user_id, preferences=preferences)
        if user:
            await self.commit()
        return user

    async def search_users(self, query: str) -> list[User]:
        """Search users by username or email.

        Args:
            query: Search query

        Returns:
            List of matching users
        """
        # Case-insensitive partial match on username or email
        search_query = select(User).where(
            (User.username.ilike(f"%{query}%")) | (User.email.ilike(f"%{query}%"))
        )
        result = await self.session.execute(search_query)
        return list(result.scalars().all())
