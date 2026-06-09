import json
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "604800"))  # 7 days
SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD", "")
DEFAULT_REDIRECT_URI = "http://localhost:8080/api/auth/github/callback"

FORMAT_SYSTEM_PROMPT = os.getenv(
    "FORMAT_SYSTEM_PROMPT",
    (
        "You are a helpful assistant. Format all answers in Markdown.\n"
        "- Use headings and bullet points when helpful.\n"
        "- Preserve indentation and line breaks.\n"
        "- For code, use fenced code blocks with the correct language tag.\n"
        "- When rewriting text, output only the rewritten text.\n"
    ),
)

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

# Redis keys for admin config
CONFIG_KEY = "gateway:config"
ADMIN_SESSION_PREFIX = "admin:session:"


# ============================================================================
# Configuration Store (Redis with .env fallback)
# ============================================================================

async def get_runtime_config() -> dict[str, str]:
    """Get runtime config from Redis."""
    if not redis_client:
        return {}
    data = await redis_client.hgetall(CONFIG_KEY)
    return data or {}


async def get_config_value(name: str, env_fallback: str = "") -> str:
    """Get config value from Redis, fallback to environment variable."""
    runtime = await get_runtime_config()
    return runtime.get(name) or os.getenv(name, env_fallback)


async def seed_config_from_env():
    """Seed Redis config from .env on first startup if empty."""
    if not redis_client:
        return
    
    existing = await redis_client.exists(CONFIG_KEY)
    if existing:
        return  # Already seeded
    
    # Seed from .env
    seed_data = {}
    if os.getenv("API_KEYS"):
        seed_data["API_KEYS"] = os.getenv("API_KEYS", "")
    if os.getenv("GITHUB_CLIENT_ID"):
        seed_data["GITHUB_CLIENT_ID"] = os.getenv("GITHUB_CLIENT_ID", "")
    if os.getenv("GITHUB_CLIENT_SECRET"):
        seed_data["GITHUB_CLIENT_SECRET"] = os.getenv("GITHUB_CLIENT_SECRET", "")
    if os.getenv("GITHUB_REDIRECT_URI"):
        seed_data["GITHUB_REDIRECT_URI"] = os.getenv("GITHUB_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    
    if seed_data:
        await redis_client.hset(CONFIG_KEY, mapping=seed_data)


async def configured_api_keys() -> dict[str, str]:
    """Returns {api_key: user_id} mapping from Redis or .env fallback."""
    raw_keys = await get_config_value("API_KEYS", os.getenv("API_KEYS", ""))
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    return {key: f"user_{abs(hash(key)) % 10000:04d}" for key in keys}


# ============================================================================
# Pydantic Models
# ============================================================================

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


# Admin models
class AdminLoginRequest(BaseModel):
    password: str


class AdminLoginResponse(BaseModel):
    admin_token: str
    expires_in: int


class AdminConfigResponse(BaseModel):
    github_client_id: str
    github_redirect_uri: str
    github_client_secret_set: bool
    api_keys: list[dict[str, str]]


class AdminConfigUpdate(BaseModel):
    github_client_id: str | None = None
    github_client_secret: str | None = None
    github_redirect_uri: str | None = None


class GeneratedApiKeyResponse(BaseModel):
    api_key: str
    preview: str


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # Seed config from .env on first startup
    await seed_config_from_env()


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()


# ============================================================================
# Authentication
# ============================================================================

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

    # Use constant-time comparison
    key_map = await configured_api_keys()
    if not key_map:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gateway API keys are not configured",
        )

    for key, user_id in key_map.items():
        if secrets.compare_digest(token, key):
            return user_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )


async def require_admin(authorization: str | None = Header(default=None)) -> None:
    """Validate admin Bearer token."""
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
    
    # Check if session exists in Redis
    session_key = f"{ADMIN_SESSION_PREFIX}{token}"
    if not await redis_client.exists(session_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin session",
        )


def mask_key(key: str) -> str:
    """Mask an API key for display (show last 4 chars)."""
    if len(key) <= 4:
        return "••••"
    return f"{'•' * (len(key) - 4)}{key[-4:]}"


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

    api_keys = await configured_api_keys()
    return {
        "status": "ok",
        "ollama_url": OLLAMA_URL,
        "redis_connected": redis_ok,
        "api_keys_configured": bool(api_keys),
    }


# ============================================================================
# Admin endpoints
# ============================================================================

@app.post("/admin/login", response_model=AdminLoginResponse)
async def admin_login(body: AdminLoginRequest):
    """Admin login - returns session token."""
    if not SUPER_ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Super admin is not configured",
        )
    
    if not secrets.compare_digest(body.password, SUPER_ADMIN_PASSWORD):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin password",
        )
    
    # Generate admin session token
    token = secrets.token_urlsafe(32)
    session_key = f"{ADMIN_SESSION_PREFIX}{token}"
    await redis_client.setex(session_key, 3600, "1")  # 1 hour
    
    return AdminLoginResponse(admin_token=token, expires_in=3600)


@app.get("/admin/config", response_model=AdminConfigResponse)
async def get_admin_config(authorization: str | None = Header(default=None)):
    """Get current admin configuration."""
    await require_admin(authorization)
    
    cfg = await get_runtime_config()
    
    # Parse API keys
    raw_keys = cfg.get("API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    api_keys_list = [
        {"id": str(i), "preview": mask_key(k)}
        for i, k in enumerate(keys)
    ]
    
    return AdminConfigResponse(
        github_client_id=cfg.get("GITHUB_CLIENT_ID", ""),
        github_redirect_uri=cfg.get("GITHUB_REDIRECT_URI", DEFAULT_REDIRECT_URI),
        github_client_secret_set=bool(cfg.get("GITHUB_CLIENT_SECRET")),
        api_keys=api_keys_list,
    )


@app.put("/admin/config")
async def update_admin_config(
    body: AdminConfigUpdate,
    authorization: str | None = Header(default=None),
):
    """Update admin configuration."""
    await require_admin(authorization)
    
    updates: dict[str, str] = {}
    
    if body.github_client_id is not None:
        updates["GITHUB_CLIENT_ID"] = body.github_client_id.strip()
    
    if body.github_client_secret is not None and body.github_client_secret.strip():
        updates["GITHUB_CLIENT_SECRET"] = body.github_client_secret.strip()
    
    if body.github_redirect_uri is not None:
        updates["GITHUB_REDIRECT_URI"] = body.github_redirect_uri.strip()
    
    if updates:
        await redis_client.hset(CONFIG_KEY, mapping=updates)
    
    return {"status": "ok", "updated_fields": list(updates.keys())}


@app.post("/admin/api-keys/generate", response_model=GeneratedApiKeyResponse)
async def generate_api_key(authorization: str | None = Header(default=None)):
    """Generate a new API key."""
    await require_admin(authorization)
    
    # Generate secure random key
    new_key = secrets.token_urlsafe(32)
    
    # Get existing keys
    cfg = await get_runtime_config()
    existing = [k.strip() for k in cfg.get("API_KEYS", "").split(",") if k.strip()]
    existing.append(new_key)
    
    # Save updated list
    await redis_client.hset(CONFIG_KEY, "API_KEYS", ",".join(existing))
    
    return GeneratedApiKeyResponse(api_key=new_key, preview=mask_key(new_key))


@app.delete("/admin/api-keys/{index}")
async def revoke_api_key(
    index: int,
    authorization: str | None = Header(default=None),
):
    """Revoke an API key by index."""
    await require_admin(authorization)
    
    cfg = await get_runtime_config()
    keys = [k.strip() for k in cfg.get("API_KEYS", "").split(",") if k.strip()]
    
    if index < 0 or index >= len(keys):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    # Remove the key
    keys.pop(index)
    await redis_client.hset(CONFIG_KEY, "API_KEYS", ",".join(keys))
    
    return {"status": "ok", "remaining_keys": len(keys)}


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
    # Ensure a consistent formatting/system prompt at the start of the session.
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": FORMAT_SYSTEM_PROMPT})
    elif messages[0].get("content") != FORMAT_SYSTEM_PROMPT:
        # Keep the session stable even if the prompt changes across deployments.
        messages[0]["content"] = FORMAT_SYSTEM_PROMPT
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
