#!/usr/bin/env python
"""
Initialize Ollama Database - Create all tables
Run this once to set up the database schema
"""

import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine

from ollama.config import get_settings
from ollama.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_database():
    """Create all tables in the database"""
    settings = get_settings()
    
    # Ensure we use asyncpg for async operations
    database_url = settings.database_url
    if "postgresql://" in database_url and "asyncpg" not in database_url:
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    logger.info("🗄️  Initializing database...")
    logger.info(f"Database URL: {database_url}")
    
    # Create async engine
    engine = create_async_engine(
        database_url,
        echo=False,
    )
    
    try:
        # Create all tables
        async with engine.begin() as conn:
            logger.info("📝 Creating tables...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ All tables created successfully")
        
        # List created tables
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
                ).fetchall()
            )
            logger.info(f"\n📊 Created tables:")
            for (table,) in result:
                logger.info(f"   - {table}")
        
        logger.info("\n✅ Database initialization complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_database())
