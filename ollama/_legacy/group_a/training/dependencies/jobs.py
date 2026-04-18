"""Training job dependencies.

Provides factory functions for creating training-related services and managers.
"""

from typing import Annotated

from fastapi import Depends
from ollama.training.services.job_manager import TrainingJobManager
from sqlalchemy.ext.asyncio import AsyncSession

from ollama.repositories import TrainingJobRepository
from ollama.services.persistence import get_db


async def get_training_job_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> TrainingJobRepository:
    """Dependency for TrainingJobRepository."""
    return TrainingJobRepository(session)


async def get_training_job_manager(
    repo: Annotated[TrainingJobRepository, Depends(get_training_job_repository)],
) -> TrainingJobManager:
    """Dependency for TrainingJobManager."""
    from ollama.main import get_training_engine

    engine = await get_training_engine()
    return TrainingJobManager(repo, engine=engine)
