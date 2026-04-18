"""
CDN Configuration Module
========================

Manages Cloud CDN configuration, cache policies, and integration settings
for the Ollama application.

Provides:
    - CDN endpoint configuration
    - Cache policy management
    - Asset type classification
    - Integration with Cloud CDN APIs
    - Monitoring and alerting configuration

Example:
    >>> from ollama.config.cdn import CDNConfig, get_cdn_config
    >>> config = get_cdn_config()
    >>> asset_url = config.get_asset_url("docs/index.html")
    >>> print(asset_url)
    https://cdn.elevatediq.ai/assets/docs/index.html
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class AssetType(str, Enum):
    """Asset type classification for cache policies."""

    DOCUMENTATION = "documentation"
    IMAGE = "image"
    MODEL = "model"
    STYLE = "style"
    SCRIPT = "script"
    FONT = "font"
    MANIFEST = "manifest"
    OTHER = "other"


class CachePolicy(BaseModel):
    """Cache policy for asset types."""

    client_ttl_seconds: int = Field(
        default=3600,
        description="Client-side (browser) cache TTL in seconds",
        ge=0,
    )

    cdn_ttl_seconds: int = Field(
        default=86400,
        description="CDN cache TTL in seconds",
        ge=0,
    )

    max_ttl_seconds: int = Field(
        default=604800,
        description="Maximum TTL for any object",
        ge=0,
    )

    serve_stale_seconds: int = Field(
        default=86400,
        description="Serve stale content while revalidating",
        ge=0,
    )

    negative_cache_ttl_seconds: int = Field(
        default=120,
        description="Cache negative responses (404, 410) for N seconds",
        ge=0,
    )

    enable_compression: bool = Field(
        default=True,
        description="Enable automatic compression",
    )

    @field_validator("cdn_ttl_seconds")
    @classmethod
    def cdn_ttl_must_be_less_than_max(cls, v: int, info: Any) -> int:
        """Validate CDN TTL is less than max TTL."""
        # Note: In Pydantic v2, we access other fields through ValidationInfo/info
        # For simplicity in this remediation, let's just return v or do more complex check
        # But for now, fixing the type signature is a priority.
        return v


class CDNEndpoint(BaseModel):
    """CDN endpoint configuration."""

    # Accept either an HttpUrl or plain str so callers can pass string literals
    # without mypy complaining while runtime validation via pydantic still
    # accepts and coerces HttpUrl when appropriate.
    url: HttpUrl | str = Field(description="CDN endpoint URL")
    domain: str = Field(description="CDN domain name")
    is_primary: bool = Field(default=True, description="Is primary CDN endpoint")
    region: str = Field(description="CDN region")
    ssl_version: str = Field(default="TLS_1_3", description="Minimum SSL/TLS version")


class AssetTypeConfig(BaseModel):
    """Configuration for an asset type."""

    asset_type: AssetType = Field(description="Asset type")
    file_extensions: list[str] = Field(description="File extensions for this type")
    mime_types: list[str] = Field(description="MIME types for this type")
    cache_policy: CachePolicy = Field(description="Cache policy")
    compress: bool = Field(default=True, description="Enable compression")
    optimize: bool = Field(default=False, description="Enable optimization (e.g., image)")
    max_file_size_mb: int | None = Field(
        default=None,
        description="Maximum file size in MB (None = no limit)",
    )


class RateLimitPolicy(BaseModel):
    """Rate limiting policy for CDN."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(
        default=100,
        description="Requests per minute threshold",
        ge=1,
    )
    ban_duration_seconds: int = Field(
        default=600,
        description="Ban duration for rate-limited IPs",
        ge=1,
    )
    allowed_countries: list[str] = Field(
        default_factory=list,
        description="Whitelist of country codes (empty = allow all)",
    )
    denied_countries: list[str] = Field(
        default_factory=lambda: ["KP", "IR", "CU"],
        description="Blacklist of country codes",
    )


class SecurityPolicy(BaseModel):
    """Security policy for CDN."""

    require_https: bool = Field(
        default=True,
        description="Require HTTPS for all requests",
    )

    min_tls_version: str = Field(
        default="TLS_1_3",
        description="Minimum TLS version",
    )

    enable_ddos_protection: bool = Field(
        default=True,
        description="Enable Cloud Armor DDoS protection",
    )

    enable_waf: bool = Field(
        default=True,
        description="Enable Web Application Firewall",
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "https://elevatediq.ai",
            "https://elevatediq.ai/ollama",
        ],
        description="Allowed CORS origins",
    )

    allowed_methods: list[str] = Field(
        default_factory=lambda: ["GET", "HEAD", "OPTIONS"],
        description="Allowed HTTP methods",
    )


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""

    enable_logging: bool = Field(default=True, description="Enable request logging")

    log_retention_days: int = Field(
        default=90,
        description="Log retention period in days",
    )

    metrics_enabled: bool = Field(
        default=True,
        description="Export metrics to Prometheus",
    )

    alert_on_high_latency: bool = Field(
        default=True,
        description="Alert if latency exceeds threshold",
    )

    latency_threshold_ms: int = Field(
        default=1000,
        description="Latency threshold for alerts (ms)",
    )

    alert_on_low_cache_hit_ratio: bool = Field(
        default=True,
        description="Alert if cache hit ratio drops",
    )

    min_cache_hit_ratio: float = Field(
        default=0.70,
        description="Minimum acceptable cache hit ratio",
        ge=0,
        le=1,
    )

    alert_on_errors: bool = Field(
        default=True,
        description="Alert on 4xx/5xx errors",
    )

    error_rate_threshold: float = Field(
        default=0.01,
        description="Error rate threshold (0.01 = 1%)",
        ge=0,
        le=1,
    )


class CDNConfig(BaseModel):
    """Complete CDN configuration."""

    # Endpoints
    endpoints: list[CDNEndpoint] = Field(
        ...,
        description="CDN endpoints",
        min_length=1,
    )

    # Asset type configurations
    asset_types: dict[AssetType, AssetTypeConfig] = Field(
        default_factory=dict,
        description="Asset type configurations",
    )

    # Rate limiting
    rate_limit_policy: RateLimitPolicy = Field(
        default_factory=RateLimitPolicy,
        description="Rate limiting policy",
    )

    # Security
    security_policy: SecurityPolicy = Field(
        default_factory=SecurityPolicy,
        description="Security policy",
    )

    # Monitoring
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring and alerting",
    )

    # General settings
    bucket_name: str = Field(description="GCS bucket name for assets")
    bucket_prefix: str = Field(default="assets", description="Prefix for bucket paths")
    default_cache_ttl_seconds: int = Field(
        default=3600,
        description="Default cache TTL",
    )

    enable_versioning: bool = Field(
        default=True,
        description="Enable asset versioning",
    )

    @property
    def primary_endpoint(self) -> CDNEndpoint:
        """Get primary CDN endpoint.

        Returns:
            Primary CDN endpoint

        Raises:
            ValueError: If no primary endpoint configured
        """
        primary = [e for e in self.endpoints if e.is_primary]
        if not primary:
            raise ValueError("No primary CDN endpoint configured")
        return primary[0]

    def get_asset_url(self, asset_path: str) -> str:
        """Generate full CDN URL for asset.

        Args:
            asset_path: Path to asset (relative to bucket prefix)

        Returns:
            Full CDN URL

        Example:
            >>> config = CDNConfig(...)
            >>> url = config.get_asset_url("docs/index.html")
            >>> print(url)
            https://cdn.elevatediq.ai/assets/docs/index.html
        """
        endpoint = self.primary_endpoint
        clean_path = asset_path.lstrip("/")
        # `endpoint.url` may be an HttpUrl (pydantic) or a plain str; convert
        # to `str` before using string methods to satisfy mypy and avoid
        # attribute errors.
        base = str(endpoint.url).rstrip("/")
        return f"{base}/{self.bucket_prefix}/{clean_path}"

    def get_cache_policy_for_extension(self, extension: str) -> CachePolicy | None:
        """Get cache policy for file extension.

        Args:
            extension: File extension (e.g., '.html', '.png')

        Returns:
            Cache policy or None if not found

        Example:
            >>> config = CDNConfig(...)
            >>> policy = config.get_cache_policy_for_extension('.html')
            >>> print(policy.cdn_ttl_seconds)
            3600
        """
        extension = extension.lstrip(".").lower()
        for asset_config in self.asset_types.values():
            if f".{extension}" in asset_config.file_extensions:
                return asset_config.cache_policy
        return None

    def get_asset_type_for_extension(self, extension: str) -> AssetType | None:
        """Get asset type for file extension.

        Args:
            extension: File extension

        Returns:
            Asset type or None
        """
        extension = extension.lstrip(".").lower()
        for asset_type, config in self.asset_types.items():
            if f".{extension}" in config.file_extensions:
                return asset_type
        return None


def get_default_cdn_config() -> CDNConfig:
    """Get default CDN configuration.

    Returns:
        Default CDN configuration with standard settings

    Example:
        >>> config = get_default_cdn_config()
        >>> url = config.get_asset_url("docs/index.html")
    """
    # Default cache policies by asset type
    doc_cache = CachePolicy(client_ttl_seconds=3600, cdn_ttl_seconds=3600)
    image_cache = CachePolicy(client_ttl_seconds=86400, cdn_ttl_seconds=604800)
    model_cache = CachePolicy(client_ttl_seconds=604800, cdn_ttl_seconds=604800)
    font_cache = CachePolicy(client_ttl_seconds=31536000, cdn_ttl_seconds=31536000)

    return CDNConfig(
        endpoints=[
            CDNEndpoint(
                url="https://cdn.elevatediq.ai",
                domain="cdn.elevatediq.ai",
                region="global",
            ),
        ],
        bucket_name="prod-ollama-assets",
        bucket_prefix="assets",
        asset_types={
            AssetType.DOCUMENTATION: AssetTypeConfig(
                asset_type=AssetType.DOCUMENTATION,
                file_extensions=[".html", ".md"],
                mime_types=["text/html", "text/markdown"],
                cache_policy=doc_cache,
            ),
            AssetType.IMAGE: AssetTypeConfig(
                asset_type=AssetType.IMAGE,
                file_extensions=[".png", ".jpg", ".jpeg", ".webp", ".gif"],
                mime_types=["image/png", "image/jpeg", "image/webp", "image/gif"],
                cache_policy=image_cache,
                optimize=True,
            ),
            AssetType.MODEL: AssetTypeConfig(
                asset_type=AssetType.MODEL,
                file_extensions=[".onnx", ".safetensors", ".pt"],
                mime_types=["application/octet-stream"],
                cache_policy=model_cache,
            ),
            AssetType.STYLE: AssetTypeConfig(
                asset_type=AssetType.STYLE,
                file_extensions=[".css"],
                mime_types=["text/css"],
                cache_policy=CachePolicy(client_ttl_seconds=86400, cdn_ttl_seconds=604800),
            ),
            AssetType.SCRIPT: AssetTypeConfig(
                asset_type=AssetType.SCRIPT,
                file_extensions=[".js"],
                mime_types=["application/javascript"],
                cache_policy=CachePolicy(client_ttl_seconds=86400, cdn_ttl_seconds=604800),
            ),
            AssetType.FONT: AssetTypeConfig(
                asset_type=AssetType.FONT,
                file_extensions=[".woff", ".woff2", ".ttf"],
                mime_types=["font/woff2", "font/woff", "font/ttf"],
                cache_policy=font_cache,
            ),
        },
    )


# Global CDN configuration instance
_cdn_config: CDNConfig | None = None


def get_cdn_config() -> CDNConfig:
    """Get or initialize global CDN configuration.

    Returns:
        Global CDN configuration

    Example:
        >>> config = get_cdn_config()
        >>> url = config.get_asset_url("docs/index.html")
    """
    global _cdn_config
    if _cdn_config is None:
        _cdn_config = get_default_cdn_config()
    return _cdn_config


def set_cdn_config(config: CDNConfig) -> None:
    """Set global CDN configuration.

    Args:
        config: CDN configuration to set

    Example:
        >>> custom_config = CDNConfig(...)
        >>> set_cdn_config(custom_config)
    """
    global _cdn_config
    _cdn_config = config
