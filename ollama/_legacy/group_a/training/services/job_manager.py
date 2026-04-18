"""Training job management service.

Orchestrates the lifecycle of training jobs, including validation,
scheduling, and state transitions.
"""

from uuid import UUID

import structlog
from ollama.training.schemas import (
    TrainingJob,
    TrainingJobCreate,
    TrainingStatus,
)
from ollama.training.services.engine import TrainingEngine

from ollama.repositories import TrainingJobRepository

log = structlog.get_logger(__name__)


class TrainingJobManager:
    """Manages model training job lifecycle.

    Handles job creation, status tracking, and coordinates with the
    training engine for local execution.
    """

    def __init__(
        self, repository: TrainingJobRepository, engine: TrainingEngine | None = None
    ) -> None:
        """Initialize job manager with training job repository.

        Args:
            repository: Data access layer for training jobs.
            engine: Training engine instance for process control.
        """
        self.repo = repository
        self.engine = engine

    async def create_job(self, request: TrainingJobCreate) -> TrainingJob:
        """Initialize a new training job and persist it.

        Args:
            request: Initial job request parameters.

        Returns:
            The persisted training job instance.
        """
        log.info("creating_training_job", model=request.base_model, dataset=request.dataset_id)

        job_data = {
            "base_model": request.base_model,
            "dataset_id": request.dataset_id,
            "config": request.config.model_dump(),
            "status": TrainingStatus.PENDING,
        }

        # Persist to database
        db_job = await self.repo.create(**job_data)

        # TODO: Dispatch to async worker queue or background task
        # await self._dispatch_job(db_job.id)

        return TrainingJob.model_validate(db_job)

    async def get_job_status(self, job_id: UUID) -> TrainingJob:
        """Retrieve current job status from the repository.

        Args:
            job_id: Unique identifier for the job.

        Returns:
            Updated job state.

        Raises:
            ValueError: If job not found.
        """
        db_job = await self.repo.get_by_id(job_id)
        if not db_job:
            log.error("job_not_found", job_id=str(job_id))
            raise ValueError(f"Job {job_id} not found")

        return TrainingJob.model_validate(db_job)

    async def cancel_job(self, job_id: UUID) -> bool:
        """Attempt to cancel a running or pending training job.

        Args:
            job_id: Unique identifier for the job.

        Returns:
            True if cancellation was initiated, False otherwise.
        """
        log.info("cancelling_training_job", job_id=str(job_id))

        db_job = await self.repo.get_by_id(job_id)
        if not db_job:
            log.warning("cancel_requested_not_found", job_id=str(job_id))
            return False

        current_status = TrainingStatus(db_job.status)

        # If already terminal, nothing to do
        if current_status in (
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        ):
            log.info("cancel_ignored_already_terminal", job_id=str(job_id), status=current_status)
            return False

        # If it's currently training, try to signal the engine
        if current_status == TrainingStatus.TRAINING and self.engine:
            await self.engine.stop(str(job_id))

        # Update database status
        await self.repo.update_status(job_id, TrainingStatus.CANCELLED)
        # Note: caller is responsible for commit if they handle the transaction
        # But here repository.update_status might need a session.commit()

        return True
