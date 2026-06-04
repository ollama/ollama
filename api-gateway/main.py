import json
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Header, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware

import auth


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "604800"))  # 7 days

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

app = FastAPI(title="Ollama API Gateway", docs_url="/docs", redoc_url=None)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Redis client (initialized on startup)
redis_client: redis.Redis | None = None


class Message(BaseModel):
    role: str
    content: str


class SessionCreate(BaseModel):
    title: str | None = None
    model: str = "llama3.2:1b"


class SessionChatRequest(BaseModel):
    message: str
    model: str | None = None
    stream: bool = False


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    model: str
    message_count: int
    created_at: str
    updated_at: str


# Admin request/response models
class AdminLoginRequest(BaseModel):
    username: str
    password: str


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreateRequest(BaseModel):
    user_id: str
    display_name: str | None = None


class ApiKeyCreateRequest(BaseModel):
    user_id: str
    label: str | None = None


class ApiKeyResponse(BaseModel):
    key_id: str
    user_id: str
    label: str
    prefix: str
    created_by: str
    created_at: str
    last_used_at: str
    revoked: str


class ApiKeyCreateResponse(BaseModel):
    api_key: str
    metadata: ApiKeyResponse


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    
    # Initialize bootstrap admin
    admin_created = await auth.initialize_bootstrap_admin(redis_client)
    if admin_created:
        print("✓ Bootstrap admin user created")
    
    # Migrate existing .env API keys to Redis
    migrated = await auth.migrate_env_api_keys(redis_client)
    if migrated > 0:
        print(f"✓ Migrated {migrated} API key(s) from .env to Redis")


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


async def require_api_key(authorization: str | None) -> str:
    """Validate Bearer token and return user_id."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Use Authorization: Bearer <api-key>",
        )

    # Verify API key from Redis
    user_id = await auth.verify_api_key(redis_client, token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )

    return user_id


async def require_admin_token(authorization: str | None = Header(default=None)) -> dict:
    """Validate admin JWT token and return payload."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin token",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Use Authorization: Bearer <admin-token>",
        )

    payload = auth.verify_jwt_token(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin token",
        )

    return payload


def forward_request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in request.headers.items():
        lower_name = name.lower()
        if lower_name in HOP_BY_HOP_HEADERS:
            continue
        if lower_name in {"authorization", "host", "content-length", "x-session-id"}:
            continue
        headers[name] = value
    return headers


def forward_response_headers(response: httpx.Response) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in response.headers.items():
        lower_name = name.lower()
        if lower_name in HOP_BY_HOP_HEADERS:
            continue
        if lower_name == "content-length":
            continue
        headers[name] = value
    return headers


async def close_upstream(response: httpx.Response, client: httpx.AsyncClient) -> None:
    await response.aclose()
    await client.aclose()


async def stream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in response.aiter_raw():
        yield chunk


def session_key(user_id: str, session_id: str) -> str:
    """Redis key for session data."""
    return f"session:{user_id}:{session_id}"


def session_index_key(user_id: str) -> str:
    """Redis key for user's session list."""
    return f"sessions:{user_id}"


async def create_session(user_id: str, title: str | None, model: str) -> str:
    """Create a new session and return session_id."""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "title": title or "New chat",
        "model": model,
        "messages": json.dumps([]),
        "created_at": now,
        "updated_at": now,
    }

    # Store session
    key = session_key(user_id, session_id)
    await redis_client.hset(key, mapping=session_data)
    await redis_client.expire(key, SESSION_TTL_SECONDS)

    # Add to user's session index
    await redis_client.sadd(session_index_key(user_id), session_id)

    return session_id


async def get_session(user_id: str, session_id: str) -> dict[str, Any] | None:
    """Load session data."""
    key = session_key(user_id, session_id)
    data = await redis_client.hgetall(key)
    if not data:
        return None

    # Parse messages JSON
    data["messages"] = json.loads(data.get("messages", "[]"))
    return data


async def update_session(user_id: str, session_id: str, messages: list[dict], model: str | None = None):
    """Update session messages and refresh TTL."""
    key = session_key(user_id, session_id)
    updates = {
        "messages": json.dumps(messages),
        "updated_at": datetime.utcnow().isoformat(),
    }
    if model:
        updates["model"] = model

    await redis_client.hset(key, mapping=updates)
    await redis_client.expire(key, SESSION_TTL_SECONDS)


async def list_sessions(user_id: str) -> list[SessionResponse]:
    """List all sessions for a user."""
    session_ids = await redis_client.smembers(session_index_key(user_id))
    sessions = []

    for sid in session_ids:
        data = await get_session(user_id, sid)
        if data:
            sessions.append(
                SessionResponse(
                    session_id=data["session_id"],
                    user_id=data["user_id"],
                    title=data["title"],
                    model=data["model"],
                    message_count=len(data["messages"]),
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                )
            )

    return sorted(sessions, key=lambda s: s.updated_at, reverse=True)


async def delete_session(user_id: str, session_id: str):
    """Delete a session."""
    await redis_client.delete(session_key(user_id, session_id))
    await redis_client.srem(session_index_key(user_id), session_id)


# ============================================================================
# Session management endpoints
# ============================================================================


@app.get("/health")
async def health() -> dict[str, str | bool]:
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    return {
        "status": "ok",
        "ollama_url": OLLAMA_URL,
        "redis_connected": redis_ok,
    }


# ============================================================================
# Admin endpoints
# ============================================================================


@app.post("/admin/auth/login", response_model=AdminLoginResponse)
async def admin_login(body: AdminLoginRequest):
    """Admin login endpoint - returns JWT token."""
    is_valid = await auth.verify_admin_credentials(
        redis_client, body.username, body.password
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Create JWT token
    token = auth.create_jwt_token(
        {"username": body.username, "role": "admin"},
        expires_delta=timedelta(hours=24),
    )

    return AdminLoginResponse(access_token=token)


@app.post("/admin/users", status_code=201)
async def create_user_endpoint(
    body: UserCreateRequest,
    admin: dict = Depends(require_admin_token),
):
    """Create a new user (admin only)."""
    user_data = await auth.create_user(
        redis_client,
        user_id=body.user_id,
        display_name=body.display_name,
    )
    return user_data


@app.get("/admin/users")
async def list_users_endpoint(admin: dict = Depends(require_admin_token)):
    """List all users (admin only)."""
    users = await auth.list_users(redis_client)
    return {"users": users}


@app.patch("/admin/users/{user_id}/disable")
async def disable_user_endpoint(
    user_id: str,
    admin: dict = Depends(require_admin_token),
):
    """Disable a user (admin only)."""
    await auth.disable_user(redis_client, user_id)
    return {"status": "disabled", "user_id": user_id}


@app.post("/admin/api-keys", response_model=ApiKeyCreateResponse, status_code=201)
async def create_api_key_endpoint(
    body: ApiKeyCreateRequest,
    admin: dict = Depends(require_admin_token),
):
    """Generate a new API key for a user (admin only)."""
    # Check if user exists
    user = await auth.get_user(redis_client, body.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    raw_key, metadata = await auth.create_api_key(
        redis_client,
        user_id=body.user_id,
        label=body.label,
        created_by=admin["username"],
    )

    return ApiKeyCreateResponse(
        api_key=raw_key,
        metadata=ApiKeyResponse(**metadata),
    )


@app.get("/admin/api-keys")
async def list_api_keys_endpoint(
    user_id: str | None = None,
    admin: dict = Depends(require_admin_token),
):
    """List API keys, optionally filtered by user_id (admin only)."""
    keys = await auth.list_api_keys(redis_client, user_id)
    return {"api_keys": keys}


@app.delete("/admin/api-keys/{key_id}", status_code=204)
async def revoke_api_key_endpoint(
    key_id: str,
    admin: dict = Depends(require_admin_token),
):
    """Revoke an API key (admin only)."""
    await auth.revoke_api_key(redis_client, key_id)


@app.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session_endpoint(
    body: SessionCreate,
    authorization: str | None = Header(default=None),
):
    """Create a new chat session."""
    user_id = await require_api_key(authorization)
    session_id = await create_session(user_id, body.title, body.model)
    session = await get_session(user_id, session_id)

    return SessionResponse(
        session_id=session["session_id"],
        user_id=session["user_id"],
        title=session["title"],
        model=session["model"],
        message_count=0,
        created_at=session["created_at"],
        updated_at=session["updated_at"],
    )


@app.get("/sessions", response_model=list[SessionResponse])
async def list_sessions_endpoint(authorization: str | None = Header(default=None)):
    """List all sessions for the authenticated user."""
    user_id = await require_api_key(authorization)
    return await list_sessions(user_id)


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    authorization: str | None = Header(default=None),
):
    """Get message history for a session."""
    user_id = await require_api_key(authorization)
    session = await get_session(user_id, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "messages": session["messages"]}


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session_endpoint(
    session_id: str,
    authorization: str | None = Header(default=None),
):
    """Delete a session."""
    user_id = await require_api_key(authorization)
    await delete_session(user_id, session_id)


@app.post("/sessions/{session_id}/chat")
async def session_chat(
    session_id: str,
    body: SessionChatRequest,
    authorization: str | None = Header(default=None),
):
    """Send a message in a session (server manages context)."""
    user_id = await require_api_key(authorization)
    session = await get_session(user_id, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build messages with session history
    messages = session["messages"]
    messages.append({"role": "user", "content": body.message})

    model = body.model or session["model"]

    # Call Ollama
    ollama_body = {
        "model": model,
        "messages": messages,
        "stream": body.stream,
    }

    client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0))

    try:
        response = await client.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json=ollama_body,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama request failed: {exc}",
        )
    finally:
        await client.aclose()

    # Extract assistant reply
    assistant_message = result["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": assistant_message})

    # Update session
    await update_session(user_id, session_id, messages, model)

    return result


# ============================================================================
# OpenAI-compatible proxy (optional X-Session-Id)
# ============================================================================


class OllamaProxyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        authorization = request.headers.get("authorization")
        session_id_header = request.headers.get("x-session-id")

        try:
            user_id = await require_api_key(authorization)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail},
            )

        # Optional: load session context if X-Session-Id provided
        if session_id_header:
            session = await get_session(user_id, session_id_header)
            if session:
                # Inject session messages into request body
                body_bytes = await request.body()
                try:
                    body = json.loads(body_bytes)
                    # Merge session history with incoming messages
                    session_messages = session["messages"]
                    if "messages" in body:
                        # Append new user message to session history
                        session_messages.extend(body["messages"])
                        body["messages"] = session_messages
                        body_bytes = json.dumps(body).encode()
                except json.JSONDecodeError:
                    pass

                # Rebuild request with merged context
                request._body = body_bytes

        path = request.url.path.removeprefix("/v1/")
        target_url = f"{OLLAMA_URL}/v1/{path}"
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"

        body = await request.body()
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0)
        )
        upstream_request = client.build_request(
            request.method,
            target_url,
            headers=forward_request_headers(request),
            content=body,
        )

        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"error": f"Ollama upstream request failed: {exc}"},
            )

        # Optionally save response to session
        if session_id_header and request.method == "POST" and "/chat/completions" in request.url.path:
            # Read response to extract assistant message
            response_bytes = b""
            async for chunk in upstream_response.aiter_raw():
                response_bytes += chunk

            try:
                response_data = json.loads(response_bytes)
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    assistant_content = response_data["choices"][0]["message"]["content"]
                    session = await get_session(user_id, session_id_header)
                    if session:
                        messages = session["messages"]
                        messages.append({"role": "assistant", "content": assistant_content})
                        await update_session(user_id, session_id_header, messages)

                # Return buffered response
                return JSONResponse(
                    status_code=upstream_response.status_code,
                    content=response_data,
                    headers=dict(forward_response_headers(upstream_response)),
                )
            except Exception:
                pass

        return StreamingResponse(
            stream_response(upstream_response),
            status_code=upstream_response.status_code,
            headers=forward_response_headers(upstream_response),
            background=BackgroundTask(close_upstream, upstream_response, client),
        )


app.add_middleware(OllamaProxyMiddleware)
