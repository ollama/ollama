"""Chat API routes.

Provides endpoints for conversational AI interactions with
context management, caching, and telemetry.
"""

import dataclasses
import json
import time
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.responses import StreamingResponse

from ollama.api.dependencies import get_model_manager
from ollama.api.dependencies.resource_manager import ResourceDependency
from ollama.api.dependencies.security import DLPDependency
from ollama.api.schemas.inference import (
    ChatRequest,
    ChatResponse,
)
from ollama.auth_manager import get_current_user_from_api_key
from ollama.models import User
from ollama.repositories import RepositoryFactory, get_repositories
from ollama.services.models.models import GenerateRequest as OllamaGenerateRequest
from ollama.services.models.models import OllamaModelManager
from ollama.services.resources.types import WorkloadType

log = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/chat", response_model=None)
async def chat(
    request: ChatRequest,
    manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
    dlp: DLPDependency,
    current_user: Annotated[User, Depends(get_current_user_from_api_key)],
    repos: Annotated[RepositoryFactory, Depends(get_repositories)],
    resources: ResourceDependency = None,
) -> StreamingResponse | ChatResponse:
    """Chat with specified model using message history.

    Supports streaming and non-streaming responses.
    Telemetry is logged for inference tracking.
    """
    start_time = time.time()

    # FAANG-Grade: Redact all messages
    for msg in request.messages:
        msg.content = await dlp.redact(msg.content)

    # FAANG-Grade: GPU Resource Arbitration
    if resources and not await resources.acquire(WorkloadType.INFERENCE, timeout=5.0):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU resources currently occupied.",
        )

    # Build a single prompt from the conversation messages and system role
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
        temperature=getattr(request, "temperature", 0.7),
        top_p=getattr(request, "top_p", 0.9),
        top_k=getattr(request, "top_k", 40),
        num_predict=getattr(request, "num_predict", 128),
        stop=getattr(request, "stop", None),
        stream=request.stream,
    )

    try:
        if request.stream:
            return _handle_streaming_chat(manager, ollama_request)
        else:
            result = None
            # Collect non-streaming response from manager.generate
            async for response in manager.generate(ollama_request):
                result = response

            # Redact output
            if dlp and result is not None:
                # result.response is plain text
                result.response = await dlp.redact(result.response)

            # Log usage for non-streaming chat
            await repos.get_usage_repository().log_usage(
                user_id=current_user.id,
                endpoint="/api/v1/chat",
                method="POST",
                response_time_ms=int((time.time() - start_time) * 1000),
                status_code=200,
                input_tokens=int(result.prompt_eval_count or 0) if result else 0,
                output_tokens=int(result.eval_count or 0) if result else 0,
                usage_metadata={"model": request.model, "stream": False},
            )
            return ChatResponse(
                model=result.model if result else request.model,
                message={"role": "assistant", "content": result.response if result else ""},
                done=True,
                total_duration=result.total_duration if result else 0,
                prompt_eval_count=result.prompt_eval_count if result else 0,
                eval_count=result.eval_count if result else 0,
            )

    except Exception as e:
        log.error("chat_failed", model=request.model, error=str(e))
        raise HTTPException(status_code=500, detail="Chat failed") from e


def _handle_streaming_chat(
    manager: OllamaModelManager, ollama_request: OllamaGenerateRequest
) -> StreamingResponse:
    """Handle chat with streaming support using the generate API."""

    async def chat_stream() -> AsyncGenerator[str, None]:
        async for response in manager.generate(ollama_request):
            yield f"data: {json.dumps(dataclasses.asdict(response))}\n\n"

    return StreamingResponse(
        chat_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _handle_non_streaming_chat(
    manager: OllamaModelManager,
    ollama_request: OllamaGenerateRequest,
    request: ChatRequest,
    dlp: Any,
) -> ChatResponse:
    """Handle chat with result collection and redaction using generate API."""
    full_content = ""
    final_data = None
    async for response in manager.generate(ollama_request):
        full_content += response.response
        if response.done:
            final_data = response

    # Redact output
    if dlp:
        full_content = await dlp.redact(full_content)

    return ChatResponse(
        model=final_data.model if final_data else request.model,
        message={"role": "assistant", "content": full_content},
        done=True,
        total_duration=final_data.total_duration if final_data else 0,
        prompt_eval_count=final_data.prompt_eval_count if final_data else 0,
        eval_count=final_data.eval_count if final_data else 0,
    )
