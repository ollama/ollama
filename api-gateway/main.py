import base64
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
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "604800"))  # 7 days
SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD", "")
DEFAULT_REDIRECT_URI = "http://localhost:8080/api/auth/github/callback"

MAX_REPO_CONTEXT_CHARS = int(os.getenv("MAX_REPO_CONTEXT_CHARS", "16000"))
MAX_REPO_FILES = int(os.getenv("MAX_REPO_FILES", "25"))
MAX_REPO_FILE_BYTES = int(os.getenv("MAX_REPO_FILE_BYTES", "100000"))

GITHUB_SKIP_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "vendor",
    "coverage",
    ".next",
    "target",
}
GITHUB_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".pdf", ".zip", ".gz",
    ".tar", ".mp4", ".mp3", ".wasm", ".bin", ".exe", ".dll",
    ".so", ".dylib", ".lock",
}

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
OAUTH_STATE_PREFIX = "oauth:state:"
GITHUB_TOKEN_PREFIX = "github:token:"


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


class GitHubContextSet(BaseModel):
    owner: str
    repo: str
    ref: str | None = None
    paths: list[str] = []


class GitHubContextFiles(BaseModel):
    paths: list[str]


class GitHubContextResponse(BaseModel):
    linked: bool
    owner: str | None = None
    repo: str | None = None
    full_name: str | None = None
    ref: str | None = None
    paths: list[str] = []


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


# GitHub OAuth models
class GitHubStatusResponse(BaseModel):
    connected: bool
    login: str | None = None
    avatar_url: str | None = None


class GitHubRepo(BaseModel):
    id: int
    name: str
    full_name: str
    description: str | None
    private: bool
    html_url: str
    default_branch: str
    updated_at: str


class GitHubBranch(BaseModel):
    name: str
    commit: dict


class GitHubContent(BaseModel):
    name: str
    path: str
    type: str
    size: int | None = None
    sha: str
    download_url: str | None = None
    html_url: str


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
        "github_owner": "",
        "github_repo": "",
        "github_ref": "",
        "github_paths": json.dumps([]),
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
    data["github_paths"] = json.loads(data.get("github_paths", "[]") or "[]")
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
# GitHub OAuth helpers
# ============================================================================

async def get_github_token(user_id: str) -> str | None:
    """Get GitHub access token for a user."""
    token_key = f"{GITHUB_TOKEN_PREFIX}{user_id}"
    token_data = await redis_client.get(token_key)
    if not token_data:
        return None
    try:
        data = json.loads(token_data)
        return data.get("access_token")
    except json.JSONDecodeError:
        return None


async def save_github_token(user_id: str, access_token: str, token_type: str, scope: str):
    """Save GitHub access token for a user."""
    token_key = f"{GITHUB_TOKEN_PREFIX}{user_id}"
    token_data = json.dumps({
        "access_token": access_token,
        "token_type": token_type,
        "scope": scope,
        "saved_at": datetime.utcnow().isoformat(),
    })
    await redis_client.set(token_key, token_data)


async def delete_github_token(user_id: str):
    """Delete GitHub access token for a user."""
    token_key = f"{GITHUB_TOKEN_PREFIX}{user_id}"
    await redis_client.delete(token_key)


async def github_api_request(user_id: str, path: str, method: str = "GET", params: dict | None = None) -> dict | list:
    """Make authenticated GitHub API request."""
    token = await get_github_token(user_id)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="GitHub not connected. Connect GitHub first.",
        )
    
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params or {})
            else:
                response = await client.request(method, url, headers=headers, params=params or {})
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="GitHub token expired or invalid. Please reconnect.",
                )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"GitHub API error: {exc.response.text}",
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"GitHub request failed: {str(exc)}",
            )


def session_has_github_context(session: dict[str, Any]) -> bool:
    return bool(session.get("github_owner") and session.get("github_repo"))


def github_context_response(session: dict[str, Any]) -> GitHubContextResponse:
    if not session_has_github_context(session):
        return GitHubContextResponse(linked=False)
    owner = session["github_owner"]
    repo = session["github_repo"]
    return GitHubContextResponse(
        linked=True,
        owner=owner,
        repo=repo,
        full_name=f"{owner}/{repo}",
        ref=session.get("github_ref") or None,
        paths=session.get("github_paths", []),
    )


async def set_session_github_context(
    user_id: str,
    session_id: str,
    owner: str,
    repo: str,
    ref: str | None,
    paths: list[str],
) -> None:
    key = session_key(user_id, session_id)
    updates = {
        "github_owner": owner.strip(),
        "github_repo": repo.strip(),
        "github_ref": (ref or "").strip(),
        "github_paths": json.dumps(paths),
        "updated_at": datetime.utcnow().isoformat(),
    }
    await redis_client.hset(key, mapping=updates)
    await redis_client.expire(key, SESSION_TTL_SECONDS)


async def clear_session_github_context(user_id: str, session_id: str) -> None:
    key = session_key(user_id, session_id)
    updates = {
        "github_owner": "",
        "github_repo": "",
        "github_ref": "",
        "github_paths": json.dumps([]),
        "updated_at": datetime.utcnow().isoformat(),
    }
    await redis_client.hset(key, mapping=updates)
    await redis_client.expire(key, SESSION_TTL_SECONDS)


def _should_skip_repo_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in GITHUB_SKIP_DIRS:
            return True
    lower = path.lower()
    for ext in GITHUB_SKIP_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


def _prioritize_repo_paths(paths: list[str]) -> list[str]:
    priority_names = (
        "readme.md", "readme", "package.json", "pyproject.toml",
        "go.mod", "cargo.toml", "dockerfile", "makefile",
        "main.py", "app.py", "index.ts", "index.js",
    )

    def score(path: str) -> tuple[int, str]:
        lower = path.lower()
        for i, name in enumerate(priority_names):
            if lower == name or lower.endswith(f"/{name}"):
                return (i, path)
        return (len(priority_names), path)

    return sorted(paths, key=score)


async def _decode_github_file(
    user_id: str,
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> str | None:
    try:
        file_data = await github_api_request(
            user_id,
            f"/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )
    except HTTPException:
        return None

    if not isinstance(file_data, dict) or file_data.get("type") != "file":
        return None

    size = file_data.get("size", 0)
    if size and size > MAX_REPO_FILE_BYTES:
        return None

    try:
        raw = base64.b64decode(file_data["content"])
        return raw.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return None


async def _collect_repo_file_paths(
    user_id: str,
    owner: str,
    repo: str,
    ref: str,
    extra_paths: list[str],
) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()

    for path in extra_paths:
        normalized = path.strip().lstrip("/")
        if normalized and not _should_skip_repo_path(normalized) and normalized not in seen:
            seen.add(normalized)
            collected.append(normalized)

    try:
        branch_data = await github_api_request(
            user_id,
            f"/repos/{owner}/{repo}/branches/{ref}",
        )
        tree_sha = branch_data["commit"]["commit"]["tree"]["sha"]
        tree_data = await github_api_request(
            user_id,
            f"/repos/{owner}/{repo}/git/trees/{tree_sha}",
            params={"recursive": "1"},
        )
        blobs = [
            item["path"]
            for item in tree_data.get("tree", [])
            if item.get("type") == "blob" and not _should_skip_repo_path(item["path"])
        ]
        blobs = _prioritize_repo_paths(blobs)
        for path in blobs:
            if path not in seen:
                seen.add(path)
                collected.append(path)
    except HTTPException:
        for readme_name in ("README.md", "README.MD", "readme.md", "README"):
            if readme_name not in seen:
                seen.add(readme_name)
                collected.insert(0, readme_name)

    return collected[:MAX_REPO_FILES]


async def build_github_context_for_chat(user_id: str, session: dict[str, Any]) -> str:
    """Build repository context text for the LLM (not stored in message history)."""
    if not session_has_github_context(session):
        return ""

    owner = session["github_owner"]
    repo = session["github_repo"]
    ref = session.get("github_ref") or ""
    extra_paths = session.get("github_paths", [])

    try:
        repo_meta = await github_api_request(user_id, f"/repos/{owner}/{repo}")
        if not ref:
            ref = repo_meta.get("default_branch", "main")
    except HTTPException:
        if not ref:
            ref = "main"

    file_paths = await _collect_repo_file_paths(user_id, owner, repo, ref, extra_paths)

    parts: list[str] = [
        f"You are helping the user with the GitHub repository {owner}/{repo} (branch: {ref}).",
        "Use the following repository files as context when answering. "
        "If the answer is not in the context, say what is missing.",
        "",
    ]
    total_chars = 0

    for path in file_paths:
        content = await _decode_github_file(user_id, owner, repo, path, ref)
        if not content:
            continue
        block = f"### File: {path}\n```\n{content}\n```\n"
        if total_chars + len(block) > MAX_REPO_CONTEXT_CHARS:
            remaining = MAX_REPO_CONTEXT_CHARS - total_chars
            if remaining > 200:
                truncated = content[:remaining - 50]
                block = f"### File: {path} (truncated)\n```\n{truncated}\n...\n```\n"
                parts.append(block)
            break
        parts.append(block)
        total_chars += len(block)

    if len(parts) <= 3:
        return ""

    return "\n".join(parts)


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


@app.get("/sessions/{session_id}/github-context", response_model=GitHubContextResponse)
async def get_session_github_context(
    session_id: str,
    authorization: str | None = Header(default=None),
):
    """Get linked GitHub repository context for a session."""
    user_id = await require_api_key(authorization)
    session = await get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return github_context_response(session)


@app.post("/sessions/{session_id}/github-context", response_model=GitHubContextResponse)
async def set_session_github_context_endpoint(
    session_id: str,
    body: GitHubContextSet,
    authorization: str | None = Header(default=None),
):
    """Link a GitHub repository to a chat session."""
    user_id = await require_api_key(authorization)
    session = await get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    owner = body.owner.strip()
    repo = body.repo.strip()
    if not owner or not repo:
        raise HTTPException(status_code=400, detail="owner and repo are required")

    # Verify repo access
    await github_api_request(user_id, f"/repos/{owner}/{repo}")

    ref = body.ref
    if not ref:
        repo_meta = await github_api_request(user_id, f"/repos/{owner}/{repo}")
        ref = repo_meta.get("default_branch", "main")

    paths = [p.strip().lstrip("/") for p in body.paths if p.strip()]
    await set_session_github_context(user_id, session_id, owner, repo, ref, paths)

    key = session_key(user_id, session_id)
    await redis_client.hset(
        key,
        mapping={
            "title": f"{owner}/{repo}",
            "updated_at": datetime.utcnow().isoformat(),
        },
    )

    updated = await get_session(user_id, session_id)
    return github_context_response(updated)


@app.post("/sessions/{session_id}/github-context/files", response_model=GitHubContextResponse)
async def add_session_github_files_endpoint(
    session_id: str,
    body: GitHubContextFiles,
    authorization: str | None = Header(default=None),
):
    """Add file paths to the session's GitHub context."""
    user_id = await require_api_key(authorization)
    session = await get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session_has_github_context(session):
        raise HTTPException(
            status_code=400,
            detail="No repository linked to this session. Link a repo first.",
        )

    existing = list(session.get("github_paths", []))
    seen = set(existing)
    for path in body.paths:
        normalized = path.strip().lstrip("/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            existing.append(normalized)

    await set_session_github_context(
        user_id,
        session_id,
        session["github_owner"],
        session["github_repo"],
        session.get("github_ref") or None,
        existing,
    )
    updated = await get_session(user_id, session_id)
    return github_context_response(updated)


@app.delete("/sessions/{session_id}/github-context")
async def clear_session_github_context_endpoint(
    session_id: str,
    authorization: str | None = Header(default=None),
):
    """Remove GitHub repository context from a session."""
    user_id = await require_api_key(authorization)
    session = await get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await clear_session_github_context(user_id, session_id)
    return {"status": "ok"}


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

    system_content = FORMAT_SYSTEM_PROMPT
    if session_has_github_context(session):
        github_ctx = await build_github_context_for_chat(user_id, session)
        if github_ctx:
            system_content = f"{FORMAT_SYSTEM_PROMPT}\n\n--- GitHub Repository Context ---\n{github_ctx}"

    ollama_messages = list(messages)
    if ollama_messages and ollama_messages[0].get("role") == "system":
        ollama_messages[0] = {"role": "system", "content": system_content}
    else:
        ollama_messages.insert(0, {"role": "system", "content": system_content})

    # Call Ollama
    ollama_body = {
        "model": model,
        "messages": ollama_messages,
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
# GitHub OAuth endpoints
# ============================================================================

@app.get("/auth/github/login")
async def github_oauth_login(authorization: str | None = Header(default=None)):
    """Start GitHub OAuth flow."""
    user_id = await require_api_key(authorization)
    
    # Get GitHub OAuth config
    client_id = await get_config_value("GITHUB_CLIENT_ID")
    redirect_uri = await get_config_value("GITHUB_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub OAuth is not configured. Please configure in admin settings.",
        )
    
    # Generate state token and store with user_id
    state = secrets.token_urlsafe(32)
    state_key = f"{OAUTH_STATE_PREFIX}{state}"
    await redis_client.setex(state_key, 600, user_id)  # 10 minutes
    
    # Build GitHub authorization URL
    auth_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope=repo read:user"
        f"&state={state}"
    )
    
    return {"authorization_url": auth_url}


@app.get("/auth/github/callback")
async def github_oauth_callback(code: str, state: str):
    """Handle GitHub OAuth callback."""
    # Validate state
    state_key = f"{OAUTH_STATE_PREFIX}{state}"
    user_id = await redis_client.get(state_key)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state",
        )
    
    # Delete state after use
    await redis_client.delete(state_key)
    
    # Exchange code for access token
    client_id = await get_config_value("GITHUB_CLIENT_ID")
    client_secret = await get_config_value("GITHUB_CLIENT_SECRET")
    redirect_uri = await get_config_value("GITHUB_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub OAuth is not properly configured",
        )
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            response.raise_for_status()
            token_data = response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to exchange OAuth code: {str(exc)}",
            )
    
    if "error" in token_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"GitHub OAuth error: {token_data.get('error_description', token_data['error'])}",
        )
    
    # Save token
    await save_github_token(
        user_id,
        token_data["access_token"],
        token_data.get("token_type", "bearer"),
        token_data.get("scope", ""),
    )
    
    # Redirect to frontend with success message
    return RedirectResponse(url="/?github=connected")


@app.get("/auth/github/status", response_model=GitHubStatusResponse)
async def github_status(authorization: str | None = Header(default=None)):
    """Check GitHub connection status."""
    user_id = await require_api_key(authorization)
    
    token = await get_github_token(user_id)
    if not token:
        return GitHubStatusResponse(connected=False)
    
    try:
        user_data = await github_api_request(user_id, "/user")
        return GitHubStatusResponse(
            connected=True,
            login=user_data.get("login"),
            avatar_url=user_data.get("avatar_url"),
        )
    except HTTPException:
        return GitHubStatusResponse(connected=False)


@app.delete("/auth/github/disconnect")
async def github_disconnect(authorization: str | None = Header(default=None)):
    """Disconnect GitHub account."""
    user_id = await require_api_key(authorization)
    await delete_github_token(user_id)
    return {"status": "ok", "message": "GitHub disconnected"}


# ============================================================================
# GitHub Repository endpoints
# ============================================================================

@app.get("/github/repos", response_model=list[GitHubRepo])
async def list_github_repos(
    authorization: str | None = Header(default=None),
    per_page: int = 100,
    page: int = 1,
):
    """List user's GitHub repositories."""
    user_id = await require_api_key(authorization)
    
    repos_data = await github_api_request(
        user_id,
        "/user/repos",
        params={"per_page": per_page, "page": page, "sort": "updated"}
    )
    
    return [
        GitHubRepo(
            id=repo["id"],
            name=repo["name"],
            full_name=repo["full_name"],
            description=repo.get("description"),
            private=repo["private"],
            html_url=repo["html_url"],
            default_branch=repo.get("default_branch", "main"),
            updated_at=repo["updated_at"],
        )
        for repo in repos_data
    ]


@app.get("/github/repos/{owner}/{repo}")
async def get_github_repo(
    owner: str,
    repo: str,
    authorization: str | None = Header(default=None),
):
    """Get repository details."""
    user_id = await require_api_key(authorization)
    return await github_api_request(user_id, f"/repos/{owner}/{repo}")


@app.get("/github/repos/{owner}/{repo}/branches", response_model=list[GitHubBranch])
async def list_github_branches(
    owner: str,
    repo: str,
    authorization: str | None = Header(default=None),
):
    """List repository branches."""
    user_id = await require_api_key(authorization)
    branches_data = await github_api_request(user_id, f"/repos/{owner}/{repo}/branches")
    
    return [
        GitHubBranch(
            name=branch["name"],
            commit=branch["commit"],
        )
        for branch in branches_data
    ]


@app.get("/github/repos/{owner}/{repo}/contents/{path:path}", response_model=list[GitHubContent] | GitHubContent)
async def get_github_contents(
    owner: str,
    repo: str,
    path: str,
    authorization: str | None = Header(default=None),
    ref: str | None = None,
):
    """Get repository contents at path."""
    user_id = await require_api_key(authorization)
    
    params = {}
    if ref:
        params["ref"] = ref
    
    content_data = await github_api_request(
        user_id,
        f"/repos/{owner}/{repo}/contents/{path}",
        params=params
    )
    
    # Handle single file or directory
    if isinstance(content_data, list):
        return [
            GitHubContent(
                name=item["name"],
                path=item["path"],
                type=item["type"],
                size=item.get("size"),
                sha=item["sha"],
                download_url=item.get("download_url"),
                html_url=item["html_url"],
            )
            for item in content_data
        ]
    else:
        return GitHubContent(
            name=content_data["name"],
            path=content_data["path"],
            type=content_data["type"],
            size=content_data.get("size"),
            sha=content_data["sha"],
            download_url=content_data.get("download_url"),
            html_url=content_data["html_url"],
        )


@app.get("/github/repos/{owner}/{repo}/file/{path:path}")
async def get_github_file_content(
    owner: str,
    repo: str,
    path: str,
    authorization: str | None = Header(default=None),
    ref: str | None = None,
):
    """Get raw file content from repository."""
    user_id = await require_api_key(authorization)
    
    params = {}
    if ref:
        params["ref"] = ref
    
    file_data = await github_api_request(
        user_id,
        f"/repos/{owner}/{repo}/contents/{path}",
        params=params
    )
    
    if file_data.get("type") != "file":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path is not a file",
        )
    
    # Decode base64 content
    content = base64.b64decode(file_data["content"]).decode("utf-8")
    
    return {
        "path": path,
        "name": file_data["name"],
        "size": file_data["size"],
        "content": content,
        "sha": file_data["sha"],
    }


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
