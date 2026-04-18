"""
Conversation Repository - CRUD operations for Conversation model.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ollama.models import Conversation
from ollama.repositories.base_repository import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):  # type: ignore[misc]
    """Repository for Conversation model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(Conversation, session)

    async def get_by_user_id(self, user_id: uuid.UUID) -> list[Conversation]:
        """Get all conversations for a user.

        Args:
            user_id: User ID

        Returns:
            List of conversations
        """
        return await self.get_all(user_id=user_id)  # type: ignore[no-any-return]

    async def get_active_conversations(self, user_id: uuid.UUID) -> list[Conversation]:
        """Get all active (non-archived) conversations for a user.

        Args:
            user_id: User ID

        Returns:
            List of active conversations
        """
        conversations = await self.get_all(user_id=user_id, is_archived=False)
        return sorted(conversations, key=lambda c: c.accessed_at, reverse=True)

    async def get_by_model(self, user_id: uuid.UUID, model: str) -> list[Conversation]:
        """Get conversations for a user using specific model.

        Args:
            user_id: User ID
            model: Model name

        Returns:
            List of conversations using that model
        """
        query = select(Conversation).where(
            and_(Conversation.user_id == user_id, Conversation.model == model)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create_conversation(
        self,
        user_id: uuid.UUID,
        model: str,
        title: str | None = None,
        system_prompt: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Conversation:
        """Create a new conversation.

        Args:
            user_id: User ID
            model: Model to use
            title: Conversation title
            system_prompt: System prompt/instructions
            parameters: Model parameters (temperature, top_p, etc.)

        Returns:
            Created Conversation instance
        """
        conversation = await self.create(
            user_id=user_id,
            model=model,
            title=title or f"Conversation with {model}",
            system_prompt=system_prompt,
            parameters=parameters or {},
            is_archived=False,
            accessed_at=datetime.now(UTC),
        )
        await self.commit()
        return conversation

    async def update_accessed_at(self, conversation_id: uuid.UUID) -> Conversation | None:
        """Update accessed_at timestamp (for sorting by recent).

        Args:
            conversation_id: Conversation ID

        Returns:
            Updated Conversation instance or None if not found
        """
        conversation = await self.update(conversation_id, accessed_at=datetime.now(UTC))
        if conversation:
            await self.commit()
        return conversation

    async def archive_conversation(self, conversation_id: uuid.UUID) -> Conversation | None:
        """Archive a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Updated Conversation instance or None if not found
        """
        conversation = await self.update(conversation_id, is_archived=True)
        if conversation:
            await self.commit()
        return conversation

    async def unarchive_conversation(self, conversation_id: uuid.UUID) -> Conversation | None:
        """Unarchive a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Updated Conversation instance or None if not found
        """
        conversation = await self.update(conversation_id, is_archived=False)
        if conversation:
            await self.commit()
        return conversation

    async def update_title(self, conversation_id: uuid.UUID, title: str) -> Conversation | None:
        """Update conversation title.

        Args:
            conversation_id: Conversation ID
            title: New title

        Returns:
            Updated Conversation instance or None if not found
        """
        conversation = await self.update(conversation_id, title=title)
        if conversation:
            await self.commit()
        return conversation

    async def search_conversations(self, user_id: uuid.UUID, query: str) -> list[Conversation]:
        """Search conversations by title or model.

        Args:
            user_id: User ID
            query: Search query

        Returns:
            List of matching conversations
        """
        conversations = await self.get_all(user_id=user_id)
        query_lower = query.lower()
        return [
            c
            for c in conversations
            if (c.title and query_lower in c.title.lower()) or query_lower in c.model.lower()
        ]

    async def get_recent_conversations(
        self, user_id: uuid.UUID, limit: int = 10
    ) -> list[Conversation]:
        """Get recent conversations for a user.

        Args:
            user_id: User ID
            limit: Number of conversations to return

        Returns:
            List of recent conversations
        """
        conversations = await self.get_by_user_id(user_id)
        # Sort by accessed_at descending
        sorted_convs = sorted(conversations, key=lambda c: c.accessed_at, reverse=True)
        return sorted_convs[:limit]

    async def count_conversations(self, user_id: uuid.UUID) -> int:
        """Count total conversations for a user.

        Args:
            user_id: User ID

        Returns:
            Number of conversations
        """
        return await self.count(user_id=user_id)  # type: ignore[no-any-return]

    async def count_active_conversations(self, user_id: uuid.UUID) -> int:
        """Count active conversations for a user.

        Args:
            user_id: User ID

        Returns:
            Number of active conversations
        """
        conversations = await self.get_all(user_id=user_id, is_archived=False)
        return len(conversations)
