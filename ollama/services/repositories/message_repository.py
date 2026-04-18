"""
Message Repository - CRUD operations for Message model.
"""

import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ollama.models import Message

from .base_repository import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Repository for Message model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(Message, session)

    async def get_by_conversation_id(self, conversation_id: uuid.UUID) -> list[Message]:
        """Get all messages in a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of messages sorted by creation
        """
        messages = await self.get_all(conversation_id=conversation_id)
        return sorted(messages, key=lambda m: m.created_at)

    async def get_conversation_messages_paginated(
        self, conversation_id: uuid.UUID, page: int = 1, page_size: int = 50
    ) -> tuple[list[Message], int]:
        """Get paginated messages for a conversation.

        Args:
            conversation_id: Conversation ID
            page: Page number (1-indexed)
            page_size: Messages per page

        Returns:
            Tuple of (messages, total_count)
        """
        return await self.get_paginated(
            page=page, page_size=page_size, order_by="created_at", conversation_id=conversation_id
        )

    async def get_last_n_messages(self, conversation_id: uuid.UUID, n: int = 10) -> list[Message]:
        """Get last N messages in a conversation.

        Args:
            conversation_id: Conversation ID
            n: Number of messages

        Returns:
            List of last N messages (oldest to newest)
        """
        messages = await self.get_by_conversation_id(conversation_id)
        return messages[-n:] if len(messages) > n else messages

    async def add_user_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        tokens: Optional[int] = None,
        parent_id: Optional[uuid.UUID] = None,
    ) -> Message:
        """Add a user message to conversation.

        Args:
            conversation_id: Conversation ID
            content: Message content
            tokens: Token count
            parent_id: Parent message ID for threading

        Returns:
            Created Message instance
        """
        message = await self.create(
            conversation_id=conversation_id,
            role="user",
            content=content,
            tokens=tokens,
            finish_reason=None,
            embedding=None,
            parent_id=parent_id,
        )
        await self.commit()
        return message

    async def add_assistant_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        parent_id: Optional[uuid.UUID] = None,
    ) -> Message:
        """Add an assistant message to conversation.

        Args:
            conversation_id: Conversation ID
            content: Message content
            tokens: Token count
            finish_reason: Why generation stopped (stop, length, etc.)
            embedding: Message embedding vector
            parent_id: Parent message ID for threading

        Returns:
            Created Message instance
        """
        message = await self.create(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            tokens=tokens,
            finish_reason=finish_reason,
            embedding=embedding,
            parent_id=parent_id,
        )
        await self.commit()
        return message

    async def add_system_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        parent_id: Optional[uuid.UUID] = None,
    ) -> Message:
        """Add a system message to conversation.

        Args:
            conversation_id: Conversation ID
            content: Message content
            parent_id: Parent message ID for threading

        Returns:
            Created Message instance
        """
        message = await self.create(
            conversation_id=conversation_id,
            role="system",
            content=content,
            tokens=None,
            finish_reason=None,
            embedding=None,
            parent_id=parent_id,
        )
        await self.commit()
        return message

    async def update_embedding(
        self, message_id: uuid.UUID, embedding: list[float]
    ) -> Optional[Message]:
        """Update message embedding.

        Args:
            message_id: Message ID
            embedding: Embedding vector

        Returns:
            Updated Message instance or None if not found
        """
        message = await self.update(message_id, embedding=embedding)
        if message:
            await self.commit()
        return message

    async def get_messages_by_role(self, conversation_id: uuid.UUID, role: str) -> list[Message]:
        """Get all messages with specific role in conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system)

        Returns:
            List of messages with that role
        """
        messages = await self.get_all(conversation_id=conversation_id, role=role)
        return sorted(messages, key=lambda m: m.created_at)

    async def count_conversation_messages(self, conversation_id: uuid.UUID) -> int:
        """Count total messages in conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Number of messages
        """
        return await self.count(conversation_id=conversation_id)

    async def count_messages_by_role(self, conversation_id: uuid.UUID, role: str) -> int:
        """Count messages by role in conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role

        Returns:
            Number of messages with that role
        """
        messages = await self.get_all(conversation_id=conversation_id, role=role)
        return len(messages)

    async def delete_conversation_messages(self, conversation_id: uuid.UUID) -> int:
        """Delete all messages in a conversation (caution!).

        Args:
            conversation_id: Conversation ID

        Returns:
            Number of messages deleted
        """
        count = await self.delete_where(conversation_id=conversation_id)
        await self.commit()
        return count

    async def search_messages(self, conversation_id: uuid.UUID, query: str) -> list[Message]:
        """Search messages by content in conversation.

        Args:
            conversation_id: Conversation ID
            query: Search query

        Returns:
            List of matching messages
        """
        messages = await self.get_by_conversation_id(conversation_id)
        query_lower = query.lower()
        return [m for m in messages if query_lower in m.content.lower()]
