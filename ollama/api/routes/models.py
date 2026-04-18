"""Model management API routes.

Provides endpoints for discovering, retrieving, and managing
available AI models.
"""

from dataclasses import asdict
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ollama.api.dependencies import get_model_manager
from ollama.auth_manager import get_current_user_from_api_key
from ollama.models import User
from ollama.services.models.models import OllamaModelManager

log = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/")
async def list_models(
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
    _current_user: Annotated[User, Depends(get_current_user_from_api_key)],
) -> dict[str, Any]:
    """List all models available in the system."""
    try:
        models = await manager.list_available_models()
        return {"models": [asdict(m) for m in models]}
    except Exception as e:
        log.error("list_models_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list models") from e


@router.get("/{name}")
async def get_model(
    name: str,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
    _current_user: Annotated[User, Depends(get_current_user_from_api_key)],
) -> dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        model = await manager.get_model(name)
        return asdict(model)
    except Exception as e:
        log.error("get_model_failed", model=name, error=str(e))
        raise HTTPException(status_code=404, detail=f"Model {name} not found") from e
