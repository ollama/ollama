"""
Batch processing API for bulk text generation, embeddings, and document processing.
Supports job queuing, progress tracking, and result retrieval.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import logging
import asyncio
from datetime import datetime
from sqlalchemy import Column, String, JSON, Float, DateTime, Enum as SQLEnum

from app.core.db import Base, get_db
from app.core.auth import get_current_user
from app.schemas import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/batch", tags=["batch-processing"])


# ==================== Models ====================

class JobStatus(str, Enum):
    """Job status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobType(str, Enum):
    """Batch job type enum."""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDINGS = "embeddings"
    DOCUMENT_PROCESSING = "document_processing"


class BatchItem(BaseModel):
    """Single item in batch request."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Item ID")
    prompt: Optional[str] = Field(None, description="Prompt for generation")
    messages: Optional[List[dict]] = Field(None, description="Messages for chat")
    text: Optional[str] = Field(None, description="Text for embeddings")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class BatchRequest(BaseModel):
    """Batch processing request."""
    name: str = Field(description="Job name")
    job_type: BatchJobType = Field(description="Type of batch job")
    model: str = Field(description="Model to use")
    items: List[BatchItem] = Field(description="Items to process")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    priority: int = Field(default=0, ge=0, le=10, description="Priority level (0-10)")


class BatchItemResult(BaseModel):
    """Result for single batch item."""
    item_id: str
    status: JobStatus
    result: Optional[dict]
    error: Optional[str]
    tokens_used: Optional[int]
    processing_time: Optional[float]


class BatchJobResponse(BaseModel):
    """Batch job response."""
    job_id: str
    name: str
    job_type: BatchJobType
    status: JobStatus
    user_id: str
    priority: int
    total_items: int
    processed_items: int
    failed_items: int
    progress: float
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    results: Optional[List[BatchItemResult]]
    error: Optional[str]


# ==================== Database Models ====================

class BatchJob(Base):
    """Batch job database model."""
    __tablename__ = "batch_jobs"
    
    job_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    name = Column(String)
    job_type = Column(SQLEnum(BatchJobType))
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    priority = Column(Float, default=0)
    
    total_items = Column(Float)
    processed_items = Column(Float, default=0)
    failed_items = Column(Float, default=0)
    
    items = Column(JSON)
    results = Column(JSON, default=list)
    config = Column(JSON)
    error = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


# ==================== Job Queue ====================

class JobQueue:
    """In-memory job queue with priority support."""
    
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    
    async def submit(self, job: BatchJobResponse):
        """Submit job to queue."""
        # Use negative priority for max-heap behavior
        priority = -job.priority
        await self.queue.put((priority, job.job_id, job))
        self.jobs[job.job_id] = job
        logger.info(f"Job {job.job_id} submitted with priority {job.priority}")
    
    async def get_next(self) -> Optional[BatchJobResponse]:
        """Get next job from queue."""
        if self.queue.empty():
            return None
        try:
            _, job_id, job = self.queue.get_nowait()
            return job
        except asyncio.QueueEmpty:
            return None
    
    def get_status(self, job_id: str) -> Optional[BatchJobResponse]:
        """Get job status."""
        return self.jobs.get(job_id)
    
    def update_status(self, job_id: str, status: JobStatus):
        """Update job status."""
        if job_id in self.jobs:
            self.jobs[job_id].status = status
    
    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()


job_queue = JobQueue()


# ==================== API Endpoints ====================

@router.post("/submit", response_model=BatchJobResponse)
async def submit_batch_job(
    request: BatchRequest,
    current_user: User = Depends(get_current_user),
) -> BatchJobResponse:
    """
    Submit a batch job for processing.
    
    Supports:
    - Text generation (single prompt, multiple parameters)
    - Chat completion (conversation batches)
    - Embeddings (bulk embedding generation)
    - Document processing (PDF/text parsing)
    """
    job_id = str(uuid.uuid4())
    
    job = BatchJobResponse(
        job_id=job_id,
        name=request.name,
        job_type=request.job_type,
        status=JobStatus.PENDING,
        user_id=current_user.id,
        priority=request.priority,
        total_items=len(request.items),
        processed_items=0,
        failed_items=0,
        progress=0.0,
        created_at=datetime.utcnow(),
        started_at=None,
        completed_at=None,
        estimated_completion=None,
        results=None,
        error=None,
    )
    
    await job_queue.submit(job)
    logger.info(f"Batch job {job_id} created for user {current_user.id}")
    
    return job


@router.get("/status/{job_id}", response_model=BatchJobResponse)
async def get_batch_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> BatchJobResponse:
    """Get batch job status and progress."""
    job = job_queue.get_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return job


@router.get("/results/{job_id}", response_model=List[BatchItemResult])
async def get_batch_results(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
) -> List[BatchItemResult]:
    """Get batch job results with pagination."""
    job = job_queue.get_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if not job.results:
        return []
    
    return job.results[offset:offset + limit]


@router.delete("/{job_id}")
async def cancel_batch_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Cancel a pending or processing batch job."""
    job = job_queue.get_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")
    
    job_queue.update_status(job_id, JobStatus.CANCELLED)
    logger.info(f"Batch job {job_id} cancelled")
    
    return {"status": "cancelled", "job_id": job_id}


@router.get("/list")
async def list_batch_jobs(
    status: Optional[JobStatus] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
):
    """List batch jobs for current user."""
    jobs = [
        job for job in job_queue.jobs.values()
        if job.user_id == current_user.id
    ]
    
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    
    return {
        "total": len(jobs),
        "offset": offset,
        "limit": limit,
        "items": jobs[offset:offset + limit],
    }


# ==================== Background Worker ====================

async def process_batch_worker():
    """Background worker to process batch jobs."""
    logger.info("Batch processing worker started")
    
    while True:
        try:
            job = await job_queue.get_next()
            
            if not job:
                await asyncio.sleep(1)
                continue
            
            logger.info(f"Processing batch job {job.job_id}")
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            
            results = []
            processed = 0
            failed = 0
            
            # Process each item
            for item in job.items:
                try:
                    # Simulate processing (replace with actual implementation)
                    await asyncio.sleep(0.1)
                    
                    result = BatchItemResult(
                        item_id=item.id,
                        status=JobStatus.COMPLETED,
                        result={"text": f"Processed: {item.prompt or item.text}"},
                        error=None,
                        tokens_used=100,
                        processing_time=0.1,
                    )
                    results.append(result)
                    processed += 1
                
                except Exception as e:
                    logger.error(f"Error processing item {item.id}: {e}")
                    result = BatchItemResult(
                        item_id=item.id,
                        status=JobStatus.FAILED,
                        result=None,
                        error=str(e),
                        tokens_used=0,
                        processing_time=0.0,
                    )
                    results.append(result)
                    failed += 1
                
                # Update progress
                job.processed_items = processed + failed
                job.failed_items = failed
                job.progress = (job.processed_items / job.total_items) * 100
            
            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.results = results
            
            logger.info(f"Batch job {job.job_id} completed: {processed} processed, {failed} failed")
        
        except Exception as e:
            logger.error(f"Batch worker error: {e}")
            await asyncio.sleep(1)


# ==================== Batch Analytics ====================

@router.get("/analytics")
async def get_batch_analytics(
    current_user: User = Depends(get_current_user),
):
    """Get batch processing analytics for user."""
    jobs = [
        job for job in job_queue.jobs.values()
        if job.user_id == current_user.id
    ]
    
    completed = [j for j in jobs if j.status == JobStatus.COMPLETED]
    failed = [j for j in jobs if j.status == JobStatus.FAILED]
    processing = [j for j in jobs if j.status == JobStatus.PROCESSING]
    
    total_items = sum(j.total_items for j in jobs)
    total_processed = sum(j.processed_items for j in jobs)
    
    avg_progress = (total_processed / total_items * 100) if total_items > 0 else 0
    
    return {
        "total_jobs": len(jobs),
        "completed": len(completed),
        "failed": len(failed),
        "processing": len(processing),
        "pending": len(jobs) - len(completed) - len(failed) - len(processing),
        "total_items": total_items,
        "total_processed": total_processed,
        "average_progress": avg_progress,
        "success_rate": (len(completed) / len(jobs) * 100) if jobs else 0,
    }
