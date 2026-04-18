"""Consolidated configuration management using Pydantic Settings.

Provides environment-aware configuration with GCP Secret Manager integration,
strict validation, and secrets protection.
"""

import logging
from enum import Enum
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """Database configuration.

    Attributes:
        host: Database host
        port: Database port
        username: Database username
        password: Database password (secret)
        database: Database name
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
    """

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    username: str = Field(default="postgres")
    # Use SecretStr for defaults to satisfy type checker
    password: SecretStr = Field(default=SecretStr(""))
    database: str = Field(default="ollama")
    pool_size: int = Field(default=20, ge=5, le=100)
    max_overflow: int = Field(default=10, ge=0, le=50)
    ssl_mode: str = Field(default="prefer")
    timeout: int = Field(default=10, ge=5, le=60)

    @property
    def url(self) -> str:
        """Get SQLAlchemy database URL.

        Returns:
            Database connection URL
        """
        return (
            f"postgresql://{self.username}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    class Config:
        """Pydantic config."""

        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration.

    Attributes:
        host: Redis host
        port: Redis port
        password: Redis password (secret)
        db: Database number
        ssl: Use SSL/TLS
        socket_timeout: Socket timeout in seconds
    """

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: SecretStr | None = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)
    ssl: bool = Field(default=False)
    socket_timeout: int = Field(default=5, ge=1, le=30)

    @property
    def url(self) -> str:
        """Get Redis connection URL.

        Returns:
            Redis connection URL
        """
        password_part = f":{self.password.get_secret_value()}@" if self.password else ""
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{password_part}{self.host}:{self.port}/{self.db}"

    class Config:
        """Pydantic config."""

        env_prefix = "REDIS_"


class APISettings(BaseSettings):
    """API configuration.

    Attributes:
        host: API host
        port: API port
        workers: Number of worker processes
        reload: Enable reload on code change (dev only)
        cors_origins: CORS allowed origins
        rate_limit_requests: Rate limit requests per window
        rate_limit_window: Rate limit time window in seconds
    """

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=16)
    reload: bool = Field(default=False)
    cors_origins: list[str] = Field(
        default=["https://elevatediq.ai", "https://elevatediq.ai/ollama"]
    )
    rate_limit_requests: int = Field(default=100, ge=10)
    rate_limit_window: int = Field(default=60, ge=1)

    class Config:
        """Pydantic config."""

        env_prefix = "API_"


class OllamaSettings(BaseSettings):
    """Ollama inference engine configuration.

    Attributes:
        base_url: Ollama server base URL
        timeout: Inference timeout in seconds
        model_timeout: Model load timeout in seconds
        default_model: Default model name
        models: Available models
    """

    base_url: str = Field(default="http://ollama:11434")
    timeout: int = Field(default=300, ge=30, le=3600)
    model_timeout: int = Field(default=60, ge=10, le=600)
    default_model: str = Field(default="llama3.2")
    models: list[str] = Field(default=["llama3.2", "mistral", "neural-chat"])

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate Ollama base URL.

        Args:
            v: URL to validate

        Returns:
            Validated URL

        Raises:
            ValueError: If URL is invalid
        """
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    class Config:
        """Pydantic config."""

        env_prefix = "OLLAMA_"


class VectorDBSettings(BaseSettings):
    """Vector database (Qdrant) configuration.

    Attributes:
        host: Qdrant host
        port: Qdrant port
        api_key: Qdrant API key (secret)
        collection_name: Default collection name
        vector_size: Vector dimension size
    """

    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)
    api_key: SecretStr | None = Field(default=None)
    collection_name: str = Field(default="ollama_embeddings")
    vector_size: int = Field(default=768, ge=64, le=2048)

    @property
    def url(self) -> str:
        """Get Qdrant connection URL.

        Returns:
            Qdrant connection URL
        """
        return f"http://{self.host}:{self.port}"

    class Config:
        """Pydantic config."""

        env_prefix = "QDRANT_"


class GCPSettings(BaseSettings):
    """Google Cloud Platform configuration.

    Attributes:
        project_id: GCP project ID
        region: GCP region
        secret_manager_enabled: Enable Secret Manager
        use_workload_identity: Use Workload Identity
    """

    project_id: str | None = Field(default=None)
    region: str = Field(default="us-central1")
    secret_manager_enabled: bool = Field(default=False)
    use_workload_identity: bool = Field(default=False)

    class Config:
        """Pydantic config."""

        env_prefix = "GCP_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration.

    Attributes:
        prometheus_enabled: Enable Prometheus metrics
        jaeger_enabled: Enable Jaeger tracing
        jaeger_host: Jaeger agent host
        jaeger_port: Jaeger agent port
        log_level: Logging level
    """

    prometheus_enabled: bool = Field(default=True)
    jaeger_enabled: bool = Field(default=True)
    jaeger_host: str = Field(default="localhost")
    jaeger_port: int = Field(default=6831, ge=1, le=65535)
    log_level: str = Field(default="info")

    class Config:
        """Pydantic config."""

        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Complete application settings.

    Consolidates all configuration into a single source of truth
    with validation, secret protection, and environment awareness.

    Attributes:
        environment: Deployment environment
        debug: Enable debug mode
        database: Database configuration
        redis: Redis configuration
        api: API configuration
        ollama: Ollama configuration
        vector_db: Vector database configuration
        gcp: GCP configuration
        monitoring: Monitoring configuration
    """

    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    version: str = Field(default="1.0.0")

    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    vector_db: VectorDBSettings = Field(
        default_factory=VectorDBSettings,
        alias="vector_db",
    )
    gcp: GCPSettings = Field(default_factory=GCPSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize settings with GCP Secret Manager support.

        Args:
            **data: Configuration data
        """
        super().__init__(**data)

        # Load secrets from GCP Secret Manager if enabled
        if self.gcp.secret_manager_enabled:
            self._load_from_secret_manager()

        logger.info(
            "Settings initialized",
            extra={
                "environment": self.environment,
                "debug": self.debug,
            },
        )

    def _load_from_secret_manager(self) -> None:
        """Load secrets from GCP Secret Manager.

        Requires GCP authentication and appropriate IAM permissions.
        """
        try:
            from google.cloud import secretmanager

            client = secretmanager.SecretManagerServiceClient()

            secrets_to_load = [
                ("database-password", "database.password"),
                ("redis-password", "redis.password"),
                ("api-key", "api.api_key"),
            ]

            for secret_name, field_path in secrets_to_load:
                try:
                    name = (
                        f"projects/{self.gcp.project_id}/" f"secrets/{secret_name}/versions/latest"
                    )
                    response = client.access_secret_version(request={"name": name})
                    secret_value = response.payload.data.decode("UTF-8")

                    # Set the secret value
                    parts = field_path.split(".")
                    obj = self
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], SecretStr(secret_value))

                except Exception as e:
                    logger.warning(f"Failed to load {secret_name} from Secret Manager: {e}")

        except ImportError:
            logger.warning(
                "google-cloud-secret-manager not installed. " "Skipping Secret Manager integration."
            )

    def is_production(self) -> bool:
        """Check if running in production.

        Returns:
            True if environment is production
        """
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development.

        Returns:
            True if environment is development
        """
        return self.environment == Environment.DEVELOPMENT


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance.

    Returns:
        Settings instance

    Raises:
        RuntimeError: If settings not initialized
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def init_settings() -> Settings:
    """Initialize global settings instance.

    Returns:
        Initialized Settings instance
    """
    global _settings
    _settings = Settings()
    return _settings
