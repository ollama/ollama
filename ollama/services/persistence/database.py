"""
Database Service - PostgreSQL Connection Management
Provides SQLAlchemy connection pooling and async database operations
"""

import logging
from collections.abc import AsyncGenerator

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connection pool and session lifecycle"""

    def __init__(self, database_url: str, echo: bool = False) -> None:
        """
        Initialize database manager

        Args:
            database_url: PostgreSQL connection string (postgresql+asyncpg://...)
            echo: Enable SQL echo for debugging
        """
        self.database_url = database_url
        self._initialized = False

        # Create async engine with connection pool
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            echo=echo,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "server_settings": {"application_name": "ollama-api", "jit": "off"},
                "timeout": 30,
                "command_timeout": 30,
            },
        )

        # Create async session factory
        self.SessionLocal = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    async def initialize(self) -> None:
        """Initialize database connection (called on startup)"""
        try:
            from sqlalchemy import text

            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            self._initialized = True
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise

    async def close(self) -> None:
        """Close database connection pool (called on shutdown)"""
        await self.engine.dispose()
        logger.info("✅ Database connection pool closed")

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session for dependency injection"""
        async with self.SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Global database manager instance
_db_manager: DatabaseManager | None = None


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """Initialize global database manager"""
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo=echo)
    return _db_manager


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency"""
    if _db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database manager not initialized",
        )
    async for session in _db_manager.get_session():
        yield session


def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return _db_manager
