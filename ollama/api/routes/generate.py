"""Text generation API routes.

Provides endpoints for text generation using specified models
with caching, telemetry, and resource arbitration.
"""

import dataclasses
import hashlib
import json
import time
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.responses import StreamingResponse

from ollama.api.dependencies import get_model_manager
from ollama.api.dependencies.cache import get_cache_manager
from ollama.api.dependencies.resource_manager import ResourceDependency
from ollama.api.dependencies.security import DLPDependency
from ollama.api.dependencies.semantic_cache import get_semantic_cache
from ollama.api.schemas.inference import (
    GenerateRequest,
    GenerateResponse,
)
from ollama.auth_manager import get_current_user_from_api_key
from ollama.models import User
from ollama.repositories import RepositoryFactory, get_repositories
from ollama.services.cache.semantic_cache import SemanticCache
from ollama.services.models.models import (
    GenerateRequest as OllamaGenerateRequest,
)
from ollama.services.models.models import (
    OllamaModelManager,
)
from ollama.services.persistence.cache import CacheManager
from ollama.services.resources.types import WorkloadType

log = structlog.get_logger(__name__)

router = APIRouter()


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


@router.post("/generate", response_model=None)
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
                input_tokens=cached_data.get("prompt_eval_count", 0),
                output_tokens=cached_data.get("eval_count", 0),
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
                input_tokens=semantic_data.get("prompt_eval_count", 0),
                output_tokens=semantic_data.get("eval_count", 0),
                usage_metadata={
                    "cache_hit": True,
                    "cache_hit_type": "semantic",
                    "model": request.model,
                },
            )
            return GenerateResponse(**semantic_data)

    try:
        # Align request construction with the service-level GenerateRequest dataclass
        ollama_request = OllamaGenerateRequest(
            model=request.model,
            prompt=request.prompt,
            system=request.system,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=getattr(request, "repeat_penalty", 1.1),
            num_predict=getattr(request, "num_predict", 100),
            stop=getattr(request, "stop", None),
            context_length=getattr(request, "context_length", 2048),
            stream=request.stream,
        )

        if request.stream:
            return _handle_streaming_response(manager, ollama_request)
        else:
            result = await _handle_non_streaming_response(
                manager, ollama_request, request, cache, semantic_cache, dlp
            )

            # Log usage for fresh inference
            await repos.get_usage_repository().log_usage(
                user_id=current_user.id,
                endpoint="/api/v1/generate",
                method="POST",
                response_time_ms=int((time.time() - start_time) * 1000),
                status_code=200,
                input_tokens=int(result.prompt_eval_count or 0),
                output_tokens=int(result.eval_count or 0),
                usage_metadata={
                    "cache_hit": False,
                    "model": request.model,
                },
            )
            return result

    except HTTPException:
        raise
    except Exception as e:
        log.error("generation_failed", model=request.model, error=str(e))
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
        cache_key = f"inference:v1:gen:{request.model}:{h}"
        await cache.set(cache_key, dataclasses.asdict(result), ttl=3600)
        log.info("inference_cache_populated", model=request.model, key=cache_key)

    if semantic_cache:
        await semantic_cache.set(request.prompt, dataclasses.asdict(result), ttl=3600)

    return result
