"""Ollama inference API routes.

Provides endpoints for text generation, embeddings, model management,
and conversation history with streaming support.
"""

import dataclasses
import hashlib
import json
import time
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import StreamingResponse

from ollama.api.cache_decorators import cached_inference
from ollama.api.dependencies import get_model_manager
from ollama.api.dependencies.cache import get_cache_manager
from ollama.api.dependencies.resource_manager import ResourceDependency
from ollama.api.dependencies.security import DLPDependency
from ollama.api.dependencies.semantic_cache import get_semantic_cache
from ollama.api.schemas.inference import (
    ConversationRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    ListModelsResponse,
    ModelPullRequest,
)
from ollama.auth_manager import get_current_user_from_api_key
from ollama.models import User
from ollama.repositories import RepositoryFactory, get_repositories
from ollama.services.cache.semantic_cache import SemanticCache
from ollama.services.models.models import (
    GenerateRequest as OllamaGenerateRequest,
)
from ollama.services.models.models import (
    Model,
    OllamaModelManager,
)
from ollama.services.persistence.cache import CacheManager

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inference"])


def _generate_cache_key(request: GenerateRequest) -> str:
    """Generate a unique cache key for an inference request."""
    # Build payload for hashing
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "system": request.system,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "num_predict": request.num_predict,
    }
    dump = json.dumps(payload, sort_keys=True)
    h = hashlib.sha256(dump.encode()).hexdigest()
    return f"inference:v1:gen:{request.model}:{h}"


# Request/Response Schemas moved to ollama.api.schemas.*


# Endpoints


@router.get("/models")
async def list_models(
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
) -> ListModelsResponse:
    """List all available Ollama models.

    Returns:
        Response with available models and metadata
    """
    try:
        models = await manager.list_available_models()
        return ListModelsResponse(models=[dataclasses.asdict(m) for m in models], total=len(models))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to list models: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to list models") from e


@router.get("/models/{model_name}")
async def get_model(
    model_name: str,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
) -> Model:
    """Get details for a specific model.

    Args:
        model_name: Model identifier to retrieve

    Returns:
        Model details and metadata
    """
    try:
        models = await manager.list_available_models()
        for model in models:
            if model.name == model_name:
                return model
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get model details: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to get model details") from e


@router.post("/generate", response_model=None)
@cached_inference(ttl=3600)
async def generate(
    request: GenerateRequest,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
    cache: Annotated[CacheManager, Depends(get_cache_manager)],
    semantic_cache: Annotated[SemanticCache, Depends(get_semantic_cache)],
    dlp: DLPDependency,
    current_user: Annotated[User, Depends(get_current_user_from_api_key)],
    repos: Annotated[RepositoryFactory, Depends(get_repositories)],
    resources: ResourceDependency = None,
) -> StreamingResponse | GenerateResponse:
    """Generate text using specified model.

    Supports both streaming and non-streaming responses based on request.
    Non-streaming requests are cached in Redis (exact) and Qdrant (semantic).
    PII redaction is applied via Google Cloud DLP.
    """
    from fastapi import status

    from ollama.services.resources.types import WorkloadType

    # Start timing
    start_time = time.time()

    # FAANG-Grade: PII Redaction (Inbound)
    request.prompt = await dlp.redact(request.prompt)

    # FAANG-Grade: GPU Resource Arbitration
    if resources and not await resources.acquire(WorkloadType.INFERENCE, timeout=5.0):
        log.warning("inference_resource_conflict", model=request.model)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU resources currently occupied by training job. Try again later.",
        )

    # Elite Optimization: Cache Lookup for non-streaming requests
    if not request.stream:
        # 1. Exact match lookup (Redis)
        cache_key = _generate_cache_key(request)
        cached_data = await cache.get(cache_key)
        if cached_data:
            log.info("inference_cache_hit_exact", model=request.model, key=cache_key)

            # Log usage for cache hit
            await repos.get_usage_repository().log_usage(
                user_id=current_user.id,
                endpoint="/api/v1/generate",
                method="POST",
                response_time_ms=int((time.time() - start_time) * 1000),
                status_code=200,
                input_tokens=int(cached_data.get("prompt_eval_count") or 0),
                output_tokens=int(cached_data.get("eval_count") or 0),
                usage_metadata={
                    "cache_hit": True,
                    "cache_hit_type": "exact",
                    "model": request.model,
                },
            )
            return GenerateResponse(**cached_data)

        # 2. Semantic match lookup (Qdrant)
        semantic_data = await semantic_cache.get(request.prompt)
        if semantic_data:
            log.info("inference_cache_hit_semantic", model=request.model)

            # Log usage for semantic cache hit
            await repos.get_usage_repository().log_usage(
                user_id=current_user.id,
                endpoint="/api/v1/generate",
                method="POST",
                response_time_ms=int((time.time() - start_time) * 1000),
                status_code=200,
                input_tokens=int(semantic_data.get("prompt_eval_count") or 0),
                output_tokens=int(semantic_data.get("eval_count") or 0),
                usage_metadata={
                    "cache_hit": True,
                    "cache_hit_type": "semantic",
                    "model": request.model,
                },
            )
            return GenerateResponse(**semantic_data)

    try:
        ollama_request = OllamaGenerateRequest(
            model=request.model,
            prompt=request.prompt,
            system=request.system,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            num_predict=request.num_predict,
            stream=request.stream,
        )

        if request.stream:
            return _handle_streaming_response(manager, ollama_request)

        # Pass everything to populate caches
        response_data = await _handle_non_streaming_response(
            manager, ollama_request, request, cache, semantic_cache, dlp
        )

        # Log usage for fresh inference
        await repos.get_usage_repository().log_usage(
            user_id=current_user.id,
            endpoint="/api/v1/generate",
            method="POST",
            response_time_ms=int((time.time() - start_time) * 1000),
            status_code=200,
            input_tokens=int(response_data.prompt_eval_count or 0),
            output_tokens=int(response_data.eval_count or 0),
            usage_metadata={"cache_hit": False, "model": request.model},
        )

        return response_data

    except ValueError as e:
        log.warning("invalid_request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.error("generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Generation failed") from e


def _handle_streaming_response(
    manager: OllamaModelManager, ollama_request: OllamaGenerateRequest
) -> StreamingResponse:
    """Handle generation with streaming support."""

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Stream generation responses as Server-Sent Events."""
        async for response in manager.generate(ollama_request):
            yield f"data: {json.dumps(dataclasses.asdict(response))}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def _handle_non_streaming_response(
    manager: OllamaModelManager,
    ollama_request: OllamaGenerateRequest,
    request: GenerateRequest,
    cache: CacheManager | None,
    semantic_cache: SemanticCache | None,
    dlp: Any,
) -> GenerateResponse:
    """Handle generation with result collection, caching, and redaction."""
    full_response = ""
    final_data = None
    async for response in manager.generate(ollama_request):
        full_response += response.response
        if response.done:
            final_data = response

    # FAANG-Grade: PII Redaction (Outbound)
    if dlp:
        full_response = await dlp.redact(full_response)

    result = GenerateResponse(
        model=final_data.model if final_data else request.model,
        prompt=request.prompt,
        response=full_response,
        done=True,
        total_duration=final_data.total_duration if final_data else 0,
        prompt_eval_count=final_data.prompt_eval_count if final_data else 0,
        eval_count=final_data.eval_count if final_data else 0,
    )

    # Elite Optimization: Populate Caches
    if cache:
        cache_key = _generate_cache_key(request)
        await cache.set(cache_key, dataclasses.asdict(result), ttl=3600)
        log.info("inference_cache_populated", model=request.model, key=cache_key)

    if semantic_cache:
        await semantic_cache.set(request.prompt, dataclasses.asdict(result), ttl=3600)
        log.info("inference_cache_semantic_populated", model=request.model)

    return result


@router.post("/embeddings")
@cached_inference(ttl=1800)
async def create_embedding(
    request: EmbeddingRequest,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
    current_user: Annotated[User, Depends(get_current_user_from_api_key)],
    repos: Annotated[RepositoryFactory, Depends(get_repositories)],
) -> EmbeddingResponse:
    """Generate embeddings for input text.

    Args:
        request: Embedding request with text and model
    """
    try:
        start_time = time.time()
        embedding = await manager.generate_embedding(model_name=request.model, prompt=request.text)

        # Log usage
        await repos.get_usage_repository().log_usage(
            user_id=current_user.id,
            endpoint="/api/v1/embeddings",
            method="POST",
            response_time_ms=int((time.time() - start_time) * 1000),
            status_code=200,
            input_tokens=len(request.text.split()),  # Simple estimation
            output_tokens=0,
            usage_metadata={"model": request.model},
        )

        return EmbeddingResponse(
            embedding=embedding,
            model=request.model,
            dimensions=len(embedding),
        )
    except Exception as e:
        log.error("embedding_generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Embedding generation failed") from e


@router.post("/models/pull")
async def pull_model(
    request: ModelPullRequest,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
) -> dict[str, Any]:
    """Download and prepare a model.

    Args:
        request: Pull request with model identifier
        manager: Model manager instance

    Returns:
        Status of pull operation
    """
    try:
        await manager.pull_model(request.model_name)
        return {
            "status": "success",
            "message": f"Model {request.model_name} pulled successfully",
        }
    except Exception as e:
        log.error("model_pull_failed", model=request.model_name, error=str(e))
        raise HTTPException(status_code=500, detail="Model pull failed") from e


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
) -> dict[str, Any]:
    """Delete a model from storage.

    Args:
        model_name: Model identifier to delete
        manager: Model manager instance

    Returns:
        Status of delete operation
    """
    try:
        await manager.delete_model(model_name)
        return {
            "status": "success",
            "message": f"Model {model_name} deleted successfully",
        }
    except Exception as e:
        log.error("model_delete_failed", model=model_name, error=str(e))
        raise HTTPException(status_code=500, detail="Model delete failed") from e


@router.post("/chat")
async def chat_completion(
    request: ConversationRequest,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
    current_user: Annotated[User, Depends(get_current_user_from_api_key)],
    repos: Annotated[RepositoryFactory, Depends(get_repositories)],
) -> StreamingResponse:
    """Chat completion with conversation history."""
    try:
        # Start timing
        start_time = time.time()

        # Build prompt from conversation history
        system_prompt = "You are a helpful assistant."
        conversation_text = ""

        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                conversation_text += f"{msg.role}: {msg.content}\n"

        conversation_text += "assistant: "

        ollama_request = OllamaGenerateRequest(
            model=request.model,
            prompt=conversation_text,
            system=system_prompt,
            temperature=request.temperature,
            stream=request.stream,
        )

        async def chat_stream() -> AsyncGenerator[str, None]:
            """Stream chat responses as Server-Sent Events."""
            async for response in manager.generate(ollama_request):
                yield f"data: {json.dumps(dataclasses.asdict(response))}\n\n"
                if response.done:
                    # Log usage at the end of stream
                    # Note: Background tasks could be used here to avoid slowing down the response
                    await repos.get_usage_repository().log_usage(
                        user_id=current_user.id,
                        endpoint="/api/v1/chat",
                        method="POST",
                        response_time_ms=int((time.time() - start_time) * 1000),
                        status_code=200,
                        input_tokens=response.prompt_eval_count,
                        output_tokens=response.eval_count,
                        usage_metadata={"model": request.model, "stream": True},
                    )

        return StreamingResponse(
            chat_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        log.error("chat_completion_failed", model=request.model, error=str(e))
        raise HTTPException(status_code=500, detail="Chat completion failed") from e
