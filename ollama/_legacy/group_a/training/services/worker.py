"""Training background worker.

Handles the asynchronous execution of training jobs by picking up PENDING jobs from
the database and executing them through the TrainingEngine.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, cast

from ollama.training.schemas import TrainingConfig, TrainingStatus
from ollama.training.services.engine import TrainingEngine

from ollama.repositories import TrainingJobRepository
from ollama.services.persistence.database import DatabaseManager
from ollama.services.resources.manager import ResourceManager
from ollama.services.resources.types import WorkloadType

log = logging.getLogger(__name__)


class TrainingWorker:
    """Background worker for processing training jobs."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        engine: TrainingEngine,
        resources: ResourceManager,
        poll_interval: float = 10.0,
    ) -> None:
        """Initialize worker.

        Args:
            db_manager: Database manager for session creation.
            engine: ML training engine.
            resources: GPU resource manager.
            poll_interval: Seconds between checking for new jobs.
        """
        self.db_manager = db_manager
        self.engine = engine
        self.resources = resources
        self.poll_interval = poll_interval
        self._running = False
        self._task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        """Start the background worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("Training worker started.")

    async def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("Training worker stopped.")

    async def _run_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                await self._process_next_job()
            except Exception as e:
                log.exception("Error in training worker loop: %s", e)

            await asyncio.sleep(self.poll_interval)

    async def _process_next_job(self) -> None:
        """Pick up and process the next pending job."""
        async with self.db_manager.SessionLocal() as session:
            repository = TrainingJobRepository(session)

            # 1. Look for pending jobs
            pending_jobs = await repository.list_jobs(status=TrainingStatus.PENDING)
            if not pending_jobs:
                return

            job = pending_jobs[0]
            job_id = cast(uuid.UUID, job.id)
            log.info("Starting processing for job %s: %s", job_id, job.base_model)

            # 2. Acquire GPU Training Lock
            success = await self.resources.acquire(WorkloadType.TRAINING, timeout=60.0)
            if not success:
                log.warning(
                    "Could not acquire GPU lock for training job %s. Retrying later.", job_id
                )
                return

            try:
                # 3. Mark as running
                await repository.update_status(job_id, TrainingStatus.TRAINING)
                await session.commit()

                # 4. Execute (this will block until done)
                try:
                    # Parse config back to pydantic model for type safety in engine
                    config = TrainingConfig(**cast(dict[str, Any], job.config))
                    dataset_path = Path(cast(str, job.dataset_id))

                    await self.engine.train(
                        job_id=str(job_id),
                        config=config,
                        dataset_path=dataset_path,
                    )
                    await repository.update_status(job_id, TrainingStatus.COMPLETED)
                    log.info("Job %s completed successfully.", job_id)
                except Exception as e:
                    log.error("Job %s failed with error: %s", job_id, e)
                    await repository.update_status(job_id, TrainingStatus.FAILED)

                await session.commit()

            finally:
                # 5. Always release the lock
                self.resources.release()
