"""Configuration module for Ollama."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of workers")
    log_level: str = Field(default="INFO", description="Logging level")
    # PUBLIC_API_URL: Use real IP/DNS in development, GCP LB in production
    public_url: str = Field(
        default="https://elevatediq.ai/ollama",
        description="Public API endpoint (development: use real IP/DNS, production: GCP Load Balancer)",
    )

    # Database - Use Docker service name (internal communication)
    database_url: str = Field(
        default="postgresql+asyncpg://ollama:changeme@postgres:5432/ollama",
        description="PostgreSQL connection URL (use 'postgres' service name in Docker)",
    )
    database_pool_size: int = Field(default=20, description="DB pool size")

    # Redis - Use Docker service name (internal communication)
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL (use 'redis' service name in Docker, NOT localhost)",
    )
    redis_password: str | None = Field(default=None, description="Redis password")

    # Qdrant - Use Docker service name (internal communication)
    qdrant_host: str = Field(
        default="qdrant",
        description="Qdrant host (use 'qdrant' service name in Docker, NOT localhost)",
    )
    qdrant_port: int = Field(default=6333, description="Qdrant port")

    # Ollama - Use Docker service name (internal communication)
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        description="Ollama inference engine URL (use 'ollama' service name in Docker, NOT localhost)",
    )

    # Authentication
    jwt_secret: str = Field(
        default="development-secret-change-me",
        description="JWT signing secret key. REQUIRED in production. See .env.example",
    )
    admin_key: str = Field(
        default="dev-admin-key",
        description="Admin key for management tasks. REQUIRED in production.",
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
    # MANDATE: Never use ["*"] - must explicitly allow origins
    # Development: Allow real IP/DNS and localhost testing
    # Production: Only allow GCP Load Balancer
    cors_origins: list[str] = Field(
        default=["https://elevatediq.ai", "https://elevatediq.ai/ollama"],
        description="Allowed CORS origins (production: GCP LB only, development: add real IP/DNS)",
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_expose_headers: list[str] = Field(default=["Content-Type"])

    # Security
    trusted_hosts: list[str] = Field(default_factory=list, description="Trusted host names")
    api_key_auth_enabled: bool = Field(default=False)

    # Firebase OAuth (mirrored from Gov-AI-Scout)
    firebase_enabled: bool = Field(
        default=False, description="Enable Firebase OAuth authentication (dev: false, prod: true)"
    )
    firebase_credentials_path: str | None = Field(
        default=None,
        description="Path to Firebase service account JSON (uses GOOGLE_APPLICATION_CREDENTIALS if not set)",
    )
    firebase_project_id: str = Field(
        default="project-131055855980", description="Firebase/GCP project ID"
    )
    root_admin_email: str = Field(
        default="akushnir@bioenergystrategies.com",
        description="Root admin email (has all permissions)",
    )

    # GCP OAuth 2.0 Configuration
    gcp_oauth_client_id: str = Field(
        default="131055855980-6e0t5fjcnq4akk9rfgjsea801ceu8lcd.apps.googleusercontent.com",
        description="GCP OAuth 2.0 Client ID for authentication",
    )
    gcp_project_id: str = Field(default="project-131055855980", description="GCP Project ID")
    gcp_service_account_email: str = Field(
        default="ollama-service@project-131055855980.iam.gserviceaccount.com",
        description="GCP Service Account email",
    )

    # Models
    models_path: str = Field(default="/models", description="Path to model files")

    # GPU
    cuda_visible_devices: str = Field(default="0", description="CUDA device IDs")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
