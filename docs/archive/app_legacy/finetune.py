"""
Model fine-tuning API for training custom models.
Supports dataset upload, training configuration, progress monitoring, and inference.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import logging
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.core.auth import get_current_user
from app.schemas import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/finetune", tags=["fine-tuning"])

# Training data storage directory
TRAINING_DATA_DIR = Path("/data/finetune/datasets")
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("/data/finetune/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_user_model_dir(current_user: User, model_name: str | None = None) -> Path:
    """Resolve a safe path under the user's model directory."""

    safe_user = Path(str(current_user.id)).name
    if safe_user != str(current_user.id) or ".." in str(current_user.id):
        raise HTTPException(status_code=400, detail="Invalid user identifier")

    base_dir = (MODELS_DIR / safe_user).resolve()

    if model_name is None:
        return base_dir

    safe_model = Path(model_name).name
    if safe_model != model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail="Invalid model path")

    target_dir = (base_dir / safe_model).resolve()
    if not target_dir.is_relative_to(base_dir):
        raise HTTPException(status_code=400, detail="Invalid model path")

    return target_dir


# ==================== Models ====================

class TrainingStatus(str, Enum):
    """Training job status."""
    CREATED = "created"
    VALIDATING = "validating"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetFormat(str, Enum):
    """Dataset format types."""
    JSONL = "jsonl"  # {"instruction": "...", "output": "..."}
    CSV = "csv"      # instruction,output columns
    PARQUET = "parquet"


class TrainingConfig(BaseModel):
    """Fine-tuning training configuration."""
    learning_rate: float = Field(default=1e-5, ge=1e-6, le=1e-3)
    batch_size: int = Field(default=8, ge=1, le=128)
    num_epochs: int = Field(default=3, ge=1, le=50)
    max_seq_length: int = Field(default=512, ge=64, le=2048)
    warmup_steps: int = Field(default=100, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.1)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=32)
    evaluation_strategy: str = Field(default="epoch")  # "no", "steps", "epoch"
    save_strategy: str = Field(default="epoch")
    logging_steps: int = Field(default=100)


class DatasetInfo(BaseModel):
    """Dataset information."""
    dataset_id: str
    name: str
    format: DatasetFormat
    size_mb: float
    num_samples: int
    created_at: datetime
    validation_split: float = 0.1


class FineTuneJob(BaseModel):
    """Fine-tuning job."""
    job_id: str
    base_model: str
    dataset_id: str
    output_model_name: str
    status: TrainingStatus
    progress: float
    
    config: TrainingConfig
    user_id: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    training_loss: Optional[float]
    eval_loss: Optional[float]
    eval_accuracy: Optional[float]
    
    error: Optional[str]
    logs: List[str] = Field(default_factory=list)


# ==================== Dataset Management ====================

@router.post("/datasets")
async def create_dataset(
    file: UploadFile = File(...),
    name: str = Query(...),
    format: DatasetFormat = Query(DatasetFormat.JSONL),
    current_user: User = Depends(get_current_user),
) -> DatasetInfo:
    """
    Upload training dataset for fine-tuning.
    
    Supports formats:
    - JSONL: One JSON object per line with "instruction" and "output" fields
    - CSV: Two columns: "instruction" and "output"
    - Parquet: Binary format with instruction/output columns
    """
    dataset_id = str(uuid.uuid4())
    user_dir = TRAINING_DATA_DIR / current_user.id
    user_dir.mkdir(exist_ok=True)
    
    file_path = user_dir / f"{dataset_id}.{format.value}"
    
    # Save file
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Validate and count samples
    num_samples = 0
    if format == DatasetFormat.JSONL:
        for line in contents.decode().strip().split("\n"):
            if line:
                num_samples += 1
    
    logger.info(f"Dataset {dataset_id} created by user {current_user.id}")
    
    return DatasetInfo(
        dataset_id=dataset_id,
        name=name,
        format=format,
        size_mb=file_size_mb,
        num_samples=num_samples,
        created_at=datetime.utcnow(),
        validation_split=0.1,
    )


@router.get("/datasets")
async def list_datasets(
    current_user: User = Depends(get_current_user),
) -> List[DatasetInfo]:
    """List all datasets for current user."""
    user_dir = TRAINING_DATA_DIR / current_user.id
    
    if not user_dir.exists():
        return []
    
    datasets = []
    for file_path in user_dir.glob("*"):
        # Parse file name to get dataset ID
        dataset_id = file_path.stem
        format = DatasetFormat(file_path.suffix[1:])
        
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        datasets.append(
            DatasetInfo(
                dataset_id=dataset_id,
                name=file_path.name,
                format=format,
                size_mb=size_mb,
                num_samples=0,  # Would need to count
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                validation_split=0.1,
            )
        )
    
    return datasets


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
):
    """Delete a training dataset."""
    user_dir = TRAINING_DATA_DIR / current_user.id
    
    # Find and delete the dataset file
    for file_path in user_dir.glob(f"{dataset_id}.*"):
        file_path.unlink()
        logger.info(f"Dataset {dataset_id} deleted by user {current_user.id}")
        return {"status": "deleted", "dataset_id": dataset_id}
    
    raise HTTPException(status_code=404, detail="Dataset not found")


# ==================== Fine-Tuning Jobs ====================

jobs_registry: dict[str, FineTuneJob] = {}


@router.post("/train", response_model=FineTuneJob)
async def start_fine_tuning(
    base_model: str = Query(...),
    dataset_id: str = Query(...),
    output_model_name: str = Query(...),
    config: TrainingConfig = Query(default=TrainingConfig()),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
) -> FineTuneJob:
    """
    Start fine-tuning a model with custom dataset.
    
    Steps:
    1. Validate dataset exists
    2. Prepare training data
    3. Start training in background
    4. Monitor progress
    5. Save trained model
    """
    job_id = str(uuid.uuid4())
    
    # Validate dataset exists
    user_dir = TRAINING_DATA_DIR / current_user.id
    dataset_files = list(user_dir.glob(f"{dataset_id}.*"))
    
    if not dataset_files:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    job = FineTuneJob(
        job_id=job_id,
        base_model=base_model,
        dataset_id=dataset_id,
        output_model_name=output_model_name,
        status=TrainingStatus.CREATED,
        progress=0.0,
        config=config,
        user_id=current_user.id,
        created_at=datetime.utcnow(),
        started_at=None,
        completed_at=None,
        training_loss=None,
        eval_loss=None,
        eval_accuracy=None,
        error=None,
        logs=[],
    )
    
    jobs_registry[job_id] = job
    
    # Start training in background
    background_tasks.add_task(
        run_fine_tuning,
        job_id=job_id,
        base_model=base_model,
        dataset_path=dataset_files[0],
        output_dir=_safe_user_model_dir(current_user, output_model_name),
        config=config,
        job=job,
    )
    
    logger.info(f"Fine-tuning job {job_id} started for user {current_user.id}")
    
    return job


@router.get("/jobs/{job_id}", response_model=FineTuneJob)
async def get_training_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> FineTuneJob:
    """Get fine-tuning job status and progress."""
    job = jobs_registry.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return job


@router.get("/jobs")
async def list_training_jobs(
    status: Optional[TrainingStatus] = None,
    current_user: User = Depends(get_current_user),
):
    """List all fine-tuning jobs for current user."""
    jobs = [
        job for job in jobs_registry.values()
        if job.user_id == current_user.id
    ]
    
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    return {
        "total": len(jobs),
        "items": jobs,
    }


@router.delete("/jobs/{job_id}")
async def cancel_training(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Cancel a fine-tuning job."""
    job = jobs_registry.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")
    
    job.status = TrainingStatus.CANCELLED
    logger.info(f"Training job {job_id} cancelled")
    
    return {"status": "cancelled", "job_id": job_id}


# ==================== Model Management ====================

@router.get("/models")
async def list_trained_models(
    current_user: User = Depends(get_current_user),
):
    """List all trained models for current user."""
    user_dir = _safe_user_model_dir(current_user)
    
    if not user_dir.exists():
        return {"models": []}
    
    models = []
    for model_dir in user_dir.iterdir():
        if model_dir.is_dir():
            # Check for model files
            if (model_dir / "adapter_config.json").exists():
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "size_gb": sum(
                        f.stat().st_size for f in model_dir.rglob("*")
                    ) / (1024**3),
                    "created_at": datetime.fromtimestamp(model_dir.stat().st_ctime),
                })
    
    return {"models": models}


@router.delete("/models/{model_name}")
async def delete_trained_model(
    model_name: str,
    current_user: User = Depends(get_current_user),
):
    """Delete a trained model."""
    # This legacy endpoint is disabled to avoid unsafe filesystem operations.
    raise HTTPException(status_code=410, detail="Model deletion is disabled in legacy API")


# ==================== Background Training ====================

async def run_fine_tuning(
    job_id: str,
    base_model: str,
    dataset_path: Path,
    output_dir: Path,
    config: TrainingConfig,
    job: FineTuneJob,
):
    """Run fine-tuning in background."""
    try:
        job.status = TrainingStatus.VALIDATING
        job.logs.append("Validating dataset...")
        await asyncio.sleep(0.5)
        
        job.status = TrainingStatus.PREPARING
        job.logs.append("Preparing training data...")
        job.progress = 10.0
        await asyncio.sleep(0.5)
        
        job.status = TrainingStatus.TRAINING
        job.started_at = datetime.utcnow()
        job.logs.append(f"Starting training with {config.num_epochs} epochs...")
        
        # Simulate training
        for epoch in range(config.num_epochs):
            job.progress = 20.0 + (epoch / config.num_epochs) * 60.0
            job.logs.append(f"Epoch {epoch + 1}/{config.num_epochs}")
            
            # Simulate training loss
            training_loss = 2.0 - (epoch * 0.3)
            job.logs.append(f"Training loss: {training_loss:.4f}")
            
            await asyncio.sleep(1)
        
        job.status = TrainingStatus.EVALUATING
        job.progress = 80.0
        job.logs.append("Evaluating model...")
        
        # Simulate evaluation
        job.eval_loss = 1.0
        job.eval_accuracy = 0.85
        job.training_loss = 0.5
        job.logs.append(f"Eval loss: {job.eval_loss:.4f}")
        job.logs.append(f"Eval accuracy: {job.eval_accuracy:.2%}")
        
        await asyncio.sleep(0.5)
        
        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "adapter_config.json"
        config_path.write_text(config.json())
        
        job.status = TrainingStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress = 100.0
        job.logs.append("Training completed successfully!")
        
        logger.info(f"Fine-tuning job {job_id} completed")
    
    except Exception as e:
        logger.error(f"Fine-tuning error for {job_id}: {e}")
        job.status = TrainingStatus.FAILED
        job.error = str(e)
        job.logs.append(f"ERROR: {str(e)}")


# ==================== Inference with Fine-Tuned Models ====================

@router.post("/inference")
async def inference_with_finetuned_model(
    model_name: str = Query(...),
    prompt: str = Query(...),
    temperature: float = Query(default=0.7, ge=0.0, le=2.0),
    max_tokens: int = Query(default=256, ge=1),
    current_user: User = Depends(get_current_user),
):
    """
    Run inference with a fine-tuned model.
    
    The model must be in the trained models directory.
    """
    model_dir = _safe_user_model_dir(current_user, model_name)
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Load adapter config
    config_path = model_dir / "adapter_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=400, detail="Invalid model structure")
    
    # Simulate inference (replace with actual ollama inference)
    await asyncio.sleep(0.5)
    
    return {
        "model": model_name,
        "prompt": prompt,
        "output": f"Response from fine-tuned {model_name}",
        "tokens_used": 50,
        "temperature": temperature,
    }
