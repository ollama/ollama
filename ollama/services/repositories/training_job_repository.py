"""Training Job Repository.

CRUD operations for persisting and retrieving training job states.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ollama.models import TrainingJob
from ollama.repositories.base_repository import BaseRepository


class TrainingJobRepository(BaseRepository[TrainingJob]):  # type: ignore[misc]
    """Repository for TrainingJob model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(TrainingJob, session)

    async def get_active_jobs(self) -> list[TrainingJob]:
        """Retrieve all jobs currently in a non-terminal state.

        Returns:
            List of active training jobs.
        """
        from ollama.training.schemas import TrainingStatus

        active_statuses = [
            TrainingStatus.PENDING,
            TrainingStatus.PROVISIONING,
            TrainingStatus.TRAINING,
            TrainingStatus.EVALUATING,
            TrainingStatus.EXPORTING,
        ]
        query = select(TrainingJob).where(TrainingJob.status.in_(active_statuses))
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def list_jobs(self, status: str | None = None) -> list[TrainingJob]:
        """List jobs with optional status filter.

        Args:
            status: Optional filter by training status.

        Returns:
            List of matching training jobs.
        """
        if status:
            return await self.get_all(status=status)  # type: ignore[no-any-return]
        return await self.get_all()  # type: ignore[no-any-return]

    async def update_status(self, job_id: uuid.UUID, status: str) -> None:
        """Update the status of a specific job.

        Args:
            job_id: The ID of the job to update.
            status: The new status string.
        """
        await self.update(job_id, status=status)
