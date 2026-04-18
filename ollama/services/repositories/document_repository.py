"""
Document Repository - CRUD operations for Document model.
"""

import uuid
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ollama.models import Document

from .base_repository import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(Document, session)

    async def get_by_user_id(self, user_id: uuid.UUID) -> list[Document]:
        """Get all documents for a user.

        Args:
            user_id: User ID

        Returns:
            List of documents
        """
        return await self.get_all(user_id=user_id)

    async def get_indexed_documents(self, user_id: uuid.UUID) -> list[Document]:
        """Get all indexed documents for a user.

        Args:
            user_id: User ID

        Returns:
            List of indexed documents
        """
        return await self.get_all(user_id=user_id, is_indexed=True)

    async def get_by_collection(self, user_id: uuid.UUID, collection_name: str) -> list[Document]:
        """Get documents in a specific Qdrant collection.

        Args:
            user_id: User ID
            collection_name: Collection name

        Returns:
            List of documents in collection
        """
        query = select(Document).where(
            and_(Document.user_id == user_id, Document.vector_collection == collection_name)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create_document(
        self,
        user_id: uuid.UUID,
        title: str,
        content: str,
        chunks: Optional[list[str]] = None,
        vector_collection: Optional[str] = None,
        is_indexed: bool = False,
    ) -> Document:
        """Create a new document.

        Args:
            user_id: User ID
            title: Document title
            content: Full document content
            chunks: List of document chunks
            vector_collection: Qdrant collection name
            is_indexed: Whether document is indexed in vector DB

        Returns:
            Created Document instance
        """
        document = await self.create(
            user_id=user_id,
            title=title,
            content=content,
            chunks=chunks or [],
            vector_collection=vector_collection,
            is_indexed=is_indexed,
        )
        await self.commit()
        return document

    async def mark_indexed(
        self, document_id: uuid.UUID, collection_name: str
    ) -> Optional[Document]:
        """Mark document as indexed and set collection.

        Args:
            document_id: Document ID
            collection_name: Qdrant collection name

        Returns:
            Updated Document instance or None if not found
        """
        document = await self.update(
            document_id, is_indexed=True, vector_collection=collection_name
        )
        if document:
            await self.commit()
        return document

    async def mark_not_indexed(self, document_id: uuid.UUID) -> Optional[Document]:
        """Mark document as not indexed.

        Args:
            document_id: Document ID

        Returns:
            Updated Document instance or None if not found
        """
        document = await self.update(document_id, is_indexed=False)
        if document:
            await self.commit()
        return document

    async def update_chunks(self, document_id: uuid.UUID, chunks: list[str]) -> Optional[Document]:
        """Update document chunks.

        Args:
            document_id: Document ID
            chunks: List of chunks

        Returns:
            Updated Document instance or None if not found
        """
        document = await self.update(document_id, chunks=chunks)
        if document:
            await self.commit()
        return document

    async def search_documents(self, user_id: uuid.UUID, query: str) -> list[Document]:
        """Search documents by title or content.

        Args:
            user_id: User ID
            query: Search query

        Returns:
            List of matching documents
        """
        documents = await self.get_by_user_id(user_id)
        query_lower = query.lower()
        return [
            d
            for d in documents
            if query_lower in d.title.lower() or query_lower in d.content.lower()
        ]

    async def get_documents_paginated(
        self, user_id: uuid.UUID, page: int = 1, page_size: int = 10
    ) -> tuple[list[Document], int]:
        """Get paginated documents for a user.

        Args:
            user_id: User ID
            page: Page number (1-indexed)
            page_size: Documents per page

        Returns:
            Tuple of (documents, total_count)
        """
        return await self.get_paginated(
            page=page, page_size=page_size, order_by="created_at", user_id=user_id
        )

    async def count_documents(self, user_id: uuid.UUID) -> int:
        """Count total documents for a user.

        Args:
            user_id: User ID

        Returns:
            Number of documents
        """
        return await self.count(user_id=user_id)

    async def count_indexed_documents(self, user_id: uuid.UUID) -> int:
        """Count indexed documents for a user.

        Args:
            user_id: User ID

        Returns:
            Number of indexed documents
        """
        documents = await self.get_indexed_documents(user_id)
        return len(documents)

    async def get_by_title(self, user_id: uuid.UUID, title: str) -> Optional[Document]:
        """Get document by title.

        Args:
            user_id: User ID
            title: Document title

        Returns:
            Document instance or None if not found
        """
        query = select(Document).where(and_(Document.user_id == user_id, Document.title == title))
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
