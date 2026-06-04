"""
Authentication and authorization module for super admin and API key management.
"""
import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

import redis.asyncio as redis
from jose import JWTError, jwt
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256 for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new secure API key."""
    return f"ollama_{secrets.token_urlsafe(32)}"


def create_jwt_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict | None:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# ============================================================================
# Redis Keys
# ============================================================================


def admin_user_key(username: str) -> str:
    """Redis key for admin user data."""
    return f"admin:user:{username}"


def user_key(user_id: str) -> str:
    """Redis key for user data."""
    return f"user:{user_id}"


def api_key_hash_key(key_hash: str) -> str:
    """Redis key for API key lookup by hash."""
    return f"apikey:hash:{key_hash}"


def api_key_metadata_key(key_id: str) -> str:
    """Redis key for API key metadata."""
    return f"apikey:meta:{key_id}"


def user_api_keys_key(user_id: str) -> str:
    """Redis key for user's API key list."""
    return f"user:{user_id}:apikeys"


def all_users_key() -> str:
    """Redis key for set of all user IDs."""
    return "users:all"


def all_api_keys_key() -> str:
    """Redis key for set of all API key IDs."""
    return "apikeys:all"


# ============================================================================
# Bootstrap Admin
# ============================================================================


async def initialize_bootstrap_admin(redis_client: redis.Redis) -> bool:
    """
    Create the bootstrap admin user if it doesn't exist.
    Returns True if created, False if already exists.
    """
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD")

    if not admin_password:
        # No bootstrap admin configured
        return False

    admin_key = admin_user_key(admin_username)
    exists = await redis_client.exists(admin_key)

    if exists:
        return False

    # Create admin user
    admin_data = {
        "username": admin_username,
        "password_hash": hash_password(admin_password),
        "role": "admin",
        "created_at": datetime.utcnow().isoformat(),
    }

    await redis_client.hset(admin_key, mapping=admin_data)
    return True


async def verify_admin_credentials(
    redis_client: redis.Redis, username: str, password: str
) -> bool:
    """Verify admin username and password."""
    admin_key = admin_user_key(username)
    admin_data = await redis_client.hgetall(admin_key)

    if not admin_data:
        return False

    if admin_data.get("role") != "admin":
        return False

    password_hash = admin_data.get("password_hash")
    if not password_hash:
        return False

    return verify_password(password, password_hash)


# ============================================================================
# User Management
# ============================================================================


async def create_user(
    redis_client: redis.Redis,
    user_id: str,
    display_name: str | None = None,
    role: str = "user",
) -> dict[str, Any]:
    """Create a new user."""
    now = datetime.utcnow().isoformat()

    user_data = {
        "user_id": user_id,
        "display_name": display_name or user_id,
        "role": role,
        "created_at": now,
        "disabled": "false",
    }

    await redis_client.hset(user_key(user_id), mapping=user_data)
    await redis_client.sadd(all_users_key(), user_id)

    return user_data


async def get_user(redis_client: redis.Redis, user_id: str) -> dict[str, Any] | None:
    """Get user data."""
    data = await redis_client.hgetall(user_key(user_id))
    if not data:
        return None
    return data


async def list_users(redis_client: redis.Redis) -> list[dict[str, Any]]:
    """List all users."""
    user_ids = await redis_client.smembers(all_users_key())
    users = []

    for uid in user_ids:
        user_data = await get_user(redis_client, uid)
        if user_data:
            users.append(user_data)

    return sorted(users, key=lambda u: u.get("created_at", ""), reverse=True)


async def disable_user(redis_client: redis.Redis, user_id: str):
    """Disable a user account."""
    await redis_client.hset(user_key(user_id), "disabled", "true")


# ============================================================================
# API Key Management
# ============================================================================


async def create_api_key(
    redis_client: redis.Redis,
    user_id: str,
    label: str | None = None,
    created_by: str = "system",
) -> tuple[str, dict[str, Any]]:
    """
    Create a new API key for a user.
    Returns (raw_api_key, metadata_dict).
    The raw key is only returned once.
    """
    # Generate key
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_id = secrets.token_urlsafe(16)
    now = datetime.utcnow().isoformat()

    # Store hash -> user_id mapping for fast lookup
    await redis_client.set(api_key_hash_key(key_hash), user_id)

    # Store key metadata
    metadata = {
        "key_id": key_id,
        "user_id": user_id,
        "label": label or "API Key",
        "prefix": raw_key[:13] + "...",
        "created_by": created_by,
        "created_at": now,
        "last_used_at": "",
        "revoked": "false",
        "key_hash": key_hash,  # Store hash for revocation
    }

    await redis_client.hset(api_key_metadata_key(key_id), mapping=metadata)
    await redis_client.sadd(user_api_keys_key(user_id), key_id)
    await redis_client.sadd(all_api_keys_key(), key_id)

    return raw_key, metadata


async def verify_api_key(redis_client: redis.Redis, api_key: str) -> str | None:
    """
    Verify an API key and return the associated user_id.
    Returns None if invalid or revoked.
    """
    key_hash = hash_api_key(api_key)
    user_id = await redis_client.get(api_key_hash_key(key_hash))

    if not user_id:
        return None

    # Check if user is disabled
    user_data = await get_user(redis_client, user_id)
    if not user_data or user_data.get("disabled") == "true":
        return None

    # Find key metadata to check if revoked
    key_ids = await redis_client.smembers(user_api_keys_key(user_id))
    for key_id in key_ids:
        meta = await redis_client.hgetall(api_key_metadata_key(key_id))
        if meta.get("key_hash") == key_hash:
            if meta.get("revoked") == "true":
                return None
            # Update last used timestamp
            await redis_client.hset(
                api_key_metadata_key(key_id),
                "last_used_at",
                datetime.utcnow().isoformat(),
            )
            return user_id

    return None


async def list_api_keys(
    redis_client: redis.Redis, user_id: str | None = None
) -> list[dict[str, Any]]:
    """List API keys, optionally filtered by user_id."""
    if user_id:
        key_ids = await redis_client.smembers(user_api_keys_key(user_id))
    else:
        key_ids = await redis_client.smembers(all_api_keys_key())

    keys = []
    for key_id in key_ids:
        meta = await redis_client.hgetall(api_key_metadata_key(key_id))
        if meta and meta.get("revoked") == "false":
            keys.append(meta)

    return sorted(keys, key=lambda k: k.get("created_at", ""), reverse=True)


async def revoke_api_key(redis_client: redis.Redis, key_id: str):
    """Revoke an API key."""
    meta = await redis_client.hgetall(api_key_metadata_key(key_id))
    if meta:
        # Mark as revoked
        await redis_client.hset(api_key_metadata_key(key_id), "revoked", "true")
        # Remove hash mapping
        key_hash = meta.get("key_hash")
        if key_hash:
            await redis_client.delete(api_key_hash_key(key_hash))


# ============================================================================
# Migration from .env API_KEYS
# ============================================================================


async def migrate_env_api_keys(redis_client: redis.Redis) -> int:
    """
    Migrate API keys from .env to Redis.
    Returns the number of keys migrated.
    """
    raw_keys = os.getenv("API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not keys:
        return 0

    migrated = 0
    for api_key in keys:
        # Check if already migrated
        key_hash = hash_api_key(api_key)
        exists = await redis_client.exists(api_key_hash_key(key_hash))
        if exists:
            continue

        # Create synthetic user_id (same as before)
        user_id = f"user_{abs(hash(api_key)) % 10000:04d}"

        # Create user if doesn't exist
        user_exists = await redis_client.exists(user_key(user_id))
        if not user_exists:
            await create_user(redis_client, user_id, display_name=f"Migrated {user_id}")

        # Store the key hash
        await redis_client.set(api_key_hash_key(key_hash), user_id)

        # Create metadata
        key_id = secrets.token_urlsafe(16)
        metadata = {
            "key_id": key_id,
            "user_id": user_id,
            "label": "Migrated from .env",
            "prefix": api_key[:13] + "..." if len(api_key) > 13 else api_key + "...",
            "created_by": "migration",
            "created_at": datetime.utcnow().isoformat(),
            "last_used_at": "",
            "revoked": "false",
            "key_hash": key_hash,
        }

        await redis_client.hset(api_key_metadata_key(key_id), mapping=metadata)
        await redis_client.sadd(user_api_keys_key(user_id), key_id)
        await redis_client.sadd(all_api_keys_key(), key_id)

        migrated += 1

    return migrated
