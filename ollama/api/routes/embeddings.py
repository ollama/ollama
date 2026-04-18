"""Embeddings API routes.

Provides endpoints for generating vector embeddings from text
for semantic search and indexing.
"""

import time
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ollama.api.dependencies import get_model_manager
from ollama.api.schemas.inference import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from ollama.auth_manager import get_current_user_from_api_key
from ollama.models import User
from ollama.repositories import RepositoryFactory, get_repositories
from ollama.services.models.models import OllamaModelManager

log = structlog.get_logger(__name__)

router = APIRouter()


def get_embedding_model(name: str) -> Any:
    """Return a sentence-transformers model instance for embeddings.

    Raises a RuntimeError if `sentence-transformers` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        ) from e

    return SentenceTransformer(name)


@router.post("/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
    current_user: Annotated[User, Depends(get_current_user_from_api_key)],
    repos: Annotated[RepositoryFactory, Depends(get_repositories)],
) -> EmbeddingResponse:
    """Generate embeddings for text using specified model.

    Telemetry is logged for vector generation tracking.
    """
    start_time = time.time()

    try:
        embedding = await manager.generate_embedding(model_name=request.model, prompt=request.text)

        # Log usage for embeddings
        await repos.get_usage_repository().log_usage(
            user_id=current_user.id,
            endpoint="/api/v1/embeddings",
            method="POST",
            response_time_ms=int((time.time() - start_time) * 1000),
            status_code=200,
            input_tokens=0,  # Ollama doesn't return tokens for embeddings easily
            output_tokens=len(embedding),
            usage_metadata={"model": request.model, "type": "embedding"},
        )

        return EmbeddingResponse(
            embedding=embedding, model=request.model, dimensions=len(embedding)
        )

    except Exception as e:
        log.error("embeddings_failed", model=request.model, error=str(e))
        raise HTTPException(status_code=500, detail="Embeddings generation failed") from e
