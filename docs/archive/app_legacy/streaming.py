"""
Streaming responses using WebSocket and Server-Sent Events (SSE).
Supports real-time text generation, chat completions, and embeddings.
"""

from typing import AsyncGenerator, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
from datetime import datetime

from app.core.ollama_client import ollama_client
from app.core.auth import get_current_user
from app.schemas import ChatRequest, GenerateRequest, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])


# ==================== Server-Sent Events (SSE) ====================

async def text_generation_stream(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Stream text generation responses using SSE format."""
    try:
        async for chunk in ollama_client.generate_stream(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        ):
            # SSE format: "data: {json}\n\n"
            event_data = {
                "type": "text_delta",
                "text": chunk,
                "timestamp": datetime.utcnow().isoformat(),
            }
            yield f"data: {json.dumps(event_data)}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
        yield f"data: {json.dumps(error_data)}\n\n"

    # Send completion event
    completion_data = {
        "type": "complete",
        "timestamp": datetime.utcnow().isoformat(),
    }
    yield f"data: {json.dumps(completion_data)}\n\n"


async def chat_stream(
    model: str,
    messages: list,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses using SSE format."""
    try:
        async for chunk in ollama_client.chat_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
        ):
            event_data = {
                "type": "message_delta",
                "text": chunk,
                "timestamp": datetime.utcnow().isoformat(),
            }
            yield f"data: {json.dumps(event_data)}\n\n"

    except Exception as e:
        logger.error(f"Chat streaming error: {e}")
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
        yield f"data: {json.dumps(error_data)}\n\n"

    completion_data = {
        "type": "complete",
        "timestamp": datetime.utcnow().isoformat(),
    }
    yield f"data: {json.dumps(completion_data)}\n\n"


@router.post("/generate")
async def stream_generate(
    request: GenerateRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream text generation using SSE.
    
    Example:
    ```javascript
    const eventSource = new EventSource('/api/v1/stream/generate?model=llama2&prompt=Hello');
    eventSource.addEventListener('message', (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'text_delta') {
            console.log(data.text);
        } else if (data.type === 'complete') {
            eventSource.close();
        }
    });
    ```
    """
    logger.info(f"Stream generate for user {current_user.id} with model {request.model}")

    return StreamingResponse(
        text_generation_stream(
            model=request.model,
            prompt=request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/chat")
async def stream_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream chat completion using SSE.
    
    Example:
    ```javascript
    const eventSource = new EventSource('/api/v1/stream/chat?model=llama2');
    eventSource.addEventListener('message', (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'message_delta') {
            console.log(data.text);
        }
    });
    ```
    """
    logger.info(f"Stream chat for user {current_user.id} with model {request.model}")

    return StreamingResponse(
        chat_stream(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            system_prompt=request.system_prompt,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ==================== WebSocket Support ====================

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.message_queue: dict[str, asyncio.Queue] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.message_queue[client_id] = asyncio.Queue()
        logger.info(f"Client {client_id} connected. Active: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        """Unregister a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.message_queue:
            del self.message_queue[client_id]
        logger.info(f"Client {client_id} disconnected. Active: {len(self.active_connections)}")

    async def send_personal(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")


manager = ConnectionManager()


@router.websocket("/ws/chat/{client_id}")
async def websocket_chat(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time chat streaming.
    
    Expected message format:
    {
        "type": "message",
        "model": "llama2",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7
    }
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                model = data.get("model", "llama2")
                messages = data.get("messages", [])
                temperature = data.get("temperature", 0.7)
                
                logger.info(f"WebSocket chat from {client_id} using {model}")
                
                # Send start event
                await manager.send_personal(
                    {
                        "type": "start",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    client_id,
                )
                
                # Stream response
                try:
                    async for chunk in ollama_client.chat_stream(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    ):
                        await manager.send_personal(
                            {
                                "type": "message_delta",
                                "text": chunk,
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                            client_id,
                        )
                        # Allow client to interrupt
                        await asyncio.sleep(0.01)
                    
                    # Send completion
                    await manager.send_personal(
                        {
                            "type": "complete",
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        client_id,
                    )
                
                except Exception as e:
                    logger.error(f"Chat streaming error: {e}")
                    await manager.send_personal(
                        {
                            "type": "error",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        client_id,
                    )
            
            elif data.get("type") == "ping":
                # Keep-alive
                await manager.send_personal(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    client_id,
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


@router.websocket("/ws/generate/{client_id}")
async def websocket_generate(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time text generation streaming.
    
    Expected message format:
    {
        "type": "generate",
        "model": "llama2",
        "prompt": "Write a story...",
        "temperature": 0.7,
        "max_tokens": 500
    }
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "generate":
                model = data.get("model", "llama2")
                prompt = data.get("prompt", "")
                temperature = data.get("temperature", 0.7)
                max_tokens = data.get("max_tokens")
                
                logger.info(f"WebSocket generate from {client_id} using {model}")
                
                await manager.send_personal(
                    {"type": "start", "timestamp": datetime.utcnow().isoformat()},
                    client_id,
                )
                
                try:
                    async for chunk in ollama_client.generate_stream(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        await manager.send_personal(
                            {
                                "type": "text_delta",
                                "text": chunk,
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                            client_id,
                        )
                        await asyncio.sleep(0.01)
                    
                    await manager.send_personal(
                        {
                            "type": "complete",
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        client_id,
                    )
                
                except Exception as e:
                    logger.error(f"Generate streaming error: {e}")
                    await manager.send_personal(
                        {
                            "type": "error",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        client_id,
                    )
            
            elif data.get("type") == "ping":
                await manager.send_personal(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    client_id,
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)
