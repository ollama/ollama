"""
Base Repository - Abstract base class for all repository implementations.
Provides common CRUD operations with async SQLAlchemy support.
"""

import uuid
from typing import Any, Generic, TypeVar

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

T = TypeVar("T", bound=DeclarativeBase)


class BaseRepository(Generic[T]):
    """
    Generic async repository for SQLAlchemy ORM models.
    Provides common CRUD operations with type safety.
    """

    def __init__(self, model: type[T], session: AsyncSession):
        """Initialize repository with model and session.

        Args:
            model: SQLAlchemy ORM model class
            session: Async SQLAlchemy session
        """
        self.model = model
        self.session = session

    async def create(self, **kwargs: Any) -> T:
        """Create and persist a new record.

        Args:
            **kwargs: Field values for the model

        Returns:
            Created model instance
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def get_by_id(self, id: uuid.UUID) -> T | None:
        """Retrieve record by ID.

        Args:
            id: UUID primary key

        Returns:
            Model instance or None if not found
        """
        return await self.session.get(self.model, id)

    async def get_one(self, **filters: Any) -> T | None:
        """Retrieve single record matching filters.

        Args:
            **filters: Column=value pairs for WHERE clause

        Returns:
            Model instance or None if not found
        """
        query = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_all(self, **filters: Any) -> list[T]:
        """Retrieve all records matching filters.

        Args:
            **filters: Column=value pairs for WHERE clause

        Returns:
            List of model instances
        """
        query = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_paginated(
        self, page: int = 1, page_size: int = 10, order_by: str | None = None, **filters: Any
    ) -> tuple[list[T], int]:
        """Retrieve paginated records.

        Args:
            page: Page number (1-indexed)
            page_size: Records per page
            order_by: Column name to order by
            **filters: Column=value pairs for WHERE clause

        Returns:
            Tuple of (records, total_count)
        """
        # Get total count
        total = await self.count(**filters)

        query = select(self.model)

        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        # Apply ordering
        if order_by and hasattr(self.model, order_by):
            query = query.order_by(getattr(self.model, order_by).desc())

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        result = await self.session.execute(query)
        records: list[T] = list(result.scalars().all())

        return records, total

    async def update(self, id: uuid.UUID, **kwargs: Any) -> T | None:
        """Update record and return updated instance.

        Args:
            id: UUID primary key
            **kwargs: Fields to update

        Returns:
            Updated model instance or None if not found
        """
        instance = await self.get_by_id(id)
        if not instance:
            return None

        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        await self.session.flush()
        return instance

    async def update_where(self, values: dict[str, Any], **filters: Any) -> int:
        """Update multiple records matching filters.

        Args:
            values: Dict of column=value pairs to update
            **filters: Column=value pairs for WHERE clause

        Returns:
            Number of rows updated
        """
        query = update(self.model)

        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        query = query.values(**values)
        result: Any = await self.session.execute(query)
        return int(result.rowcount) if result.rowcount is not None else 0

    async def delete(self, id: uuid.UUID) -> bool:
        """Delete record by ID.

        Args:
            id: UUID primary key

        Returns:
            True if record deleted, False if not found
        """
        instance = await self.get_by_id(id)
        if not instance:
            return False

        await self.session.delete(instance)
        return True

    async def delete_where(self, **filters: Any) -> int:
        """Delete records matching filters.

        Args:
            **filters: Column=value pairs for WHERE clause

        Returns:
            Number of rows deleted
        """
        query = delete(self.model)

        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result: Any = await self.session.execute(query)
        return int(result.rowcount) if result.rowcount is not None else 0

    async def exists(self, **filters: Any) -> bool:
        """Check if record exists matching filters.

        Args:
            **filters: Column=value pairs for WHERE clause

        Returns:
            True if any record matches, False otherwise
        """
        query = select(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one_or_none() is not None

    async def count(self, **filters: Any) -> int:
        """Count records matching filters.

        Args:
            **filters: Column=value pairs for WHERE clause

        Returns:
            Number of matching records
        """
        query = select(func.count()).select_from(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return int(result.scalar() or 0)

    async def commit(self) -> None:
        """Commit transaction."""
        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback transaction."""
        await self.session.rollback()
