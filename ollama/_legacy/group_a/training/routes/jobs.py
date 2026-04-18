"""Training job route handlers.

Implementation of REST endpoints for training job lifecycle management.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from ollama.training.dependencies.jobs import get_training_job_manager
from ollama.training.schemas import (
    TrainingJob,
    TrainingJobCreate,
)
from ollama.training.services.job_manager import TrainingJobManager

router = APIRouter(prefix="/jobs", tags=["Training"])


@router.post(
    "/",
    response_model=TrainingJob,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new training job",
)
async def create_job(
    request: TrainingJobCreate,
    manager: Annotated[TrainingJobManager, Depends(get_training_job_manager)],
) -> TrainingJob:
    """Create a new asynchronous training job for model fine-tuning.

    Args:
        request: Training job configuration and base model details.
        manager: Training job manager service.

    Returns:
        The created training job with initial status.
    """
    return await manager.create_job(request)


@router.get(
    "/{job_id}",
    response_model=TrainingJob,
    summary="Get training job status",
)
async def get_job(
    job_id: UUID,
    manager: Annotated[TrainingJobManager, Depends(get_training_job_manager)],
) -> TrainingJob:
    """Retrieve the status and metrics for a specific training job.

    Args:
        job_id: Unique identifier for the training job.
        manager: Training job manager service.

    Returns:
        Current state of the training job.

    Raises:
        HTTPException: If the job is not found.
    """
    try:
        return await manager.get_job_status(job_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a training job",
)
async def cancel_job(
    job_id: UUID,
    manager: Annotated[TrainingJobManager, Depends(get_training_job_manager)],
) -> None:
    """Cancel a pending or running training job.

    Args:
        job_id: Unique identifier for the training job.
        manager: Training job manager service.

    Raises:
        HTTPException: If the job is not found or cannot be cancelled.
    """
    success = await manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to cancel job {job_id}",
        )
