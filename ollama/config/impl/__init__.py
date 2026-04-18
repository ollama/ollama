"""Configuration management for Ollama.

Centralized configuration handling with Pydantic BaseSettings.
Exposes `get_settings()` used across the app. Defaults honor the
GCP Load Balancer topology and Docker-internal networking.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of workers")
    log_level: str = Field(default="INFO", description="Logging level")
    public_url: str = Field(
        default="https://elevatediq.ai/ollama",
        description=(
            "Public API endpoint (development: use real IP/DNS, production: GCP Load Balancer)"
        ),
    )

    # Database - Docker service name
    database_url: str = Field(
        default="postgresql+asyncpg://ollama:changeme@postgres:5432/ollama",
        description="PostgreSQL connection URL (use 'postgres' service name in Docker)",
    )
    database_pool_size: int = Field(default=20, description="DB pool size")

    # Redis - Docker service name
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL (use 'redis' service name in Docker, NOT localhost)",
    )
    redis_password: str | None = Field(default=None, description="Redis password")

    # Qdrant - Docker service name
    qdrant_host: str = Field(
        default="qdrant",
        description="Qdrant host (use 'qdrant' service name in Docker, NOT localhost)",
    )
    qdrant_port: int = Field(default=6333, description="Qdrant port")

    # Ollama - Docker service name
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        description="Ollama inference engine URL (use 'ollama' service name in Docker, NOT localhost)",
    )

    # Authentication
    admin_key: str = Field(default="ollama-admin-secret-key", description="Admin authorization key")
    jwt_secret: str = Field(
        default="development-secret-change-me",
        description="JWT signing secret key. REQUIRED in production. See .env.example",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=60, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration")

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, description="Global rate limit per minute")
    rate_limit_burst: int = Field(default=100, description="Rate limit burst size")
    ollama_request_timeout: float = Field(default=300.0, description="Request timeout")
    ollama_connect_timeout: float = Field(default=10.0, description="Connection timeout")

    # CORS
    cors_origins: list[str] = Field(
        default=["https://elevatediq.ai", "https://elevatediq.ai/ollama"],
        description="Allowed CORS origins (production: GCP LB only, development: add real IP/DNS)",
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_expose_headers: list[str] = Field(default=["Content-Type"])

    # Security
    trusted_hosts: list[str] | None = Field(default=None, description="Trusted host names")
    api_key_auth_enabled: bool = Field(default=False)

    # Firebase OAuth (optional)
    firebase_enabled: bool = Field(default=False)
    firebase_credentials_path: str | None = Field(default=None)
    firebase_project_id: str = Field(default="project-131055855980")
    root_admin_email: str = Field(default="admin@example.com")

    # GCP OAuth 2.0 (optional)
    gcp_oauth_client_id: str = Field(default="")
    gcp_project_id: str = Field(default="")
    gcp_service_account_email: str = Field(default="")

    # Models
    models_path: str = Field(default="/models", description="Path to model files")

    # GPU
    cuda_visible_devices: str = Field(default="0", description="CUDA device IDs")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
