"""Google Cloud Logging integration for 7-year audit compliance.

This module provides structured logging to Google Cloud Logging with automatic
7-year retention for Landing Zone compliance (Mandate #7).

Features:
- Structured JSON logging with contextual metadata
- Automatic 7-year retention via Cloud Logging
- Request/response audit trail
- User action tracking
- Security event logging
- Performance metrics capture

Example:
    >>> from ollama.monitoring.cloud_logging import get_audit_logger
    >>> logger = get_audit_logger()
    >>> logger.log_api_request(
    ...     user_id="user_123",
    ...     endpoint="/api/v1/generate",
    ...     method="POST",
    ...     status_code=200
    ... )
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any

try:
    from google.cloud import logging as cloud_logging
    from google.cloud.logging_v2.handlers import CloudLoggingHandler

    CLOUD_LOGGING_AVAILABLE = True
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False
    CloudLoggingHandler = None

import structlog

from ollama.config import get_settings

# ============================================================================
# Configuration
# ============================================================================

# Audit log retention: 7 years (2,557 days) as per Landing Zone mandate
AUDIT_RETENTION_DAYS = 2557

# Log name for Cloud Logging
AUDIT_LOG_NAME = "ollama-api-audit"

# Severity levels for Cloud Logging
SEVERITY_LEVELS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
    "SECURITY": "ALERT",  # Security events are critical
}


# ============================================================================
# Cloud Logging Client
# ============================================================================


class CloudAuditLogger:
    """Google Cloud Logging client for 7-year audit compliance.

    Provides structured logging to GCP Cloud Logging with automatic 7-year
    retention policy enforcement via Terraform-managed GCS bucket.
    """

    def __init__(
        self,
        project_id: str | None = None,
        log_name: str = AUDIT_LOG_NAME,
        fallback_to_stdout: bool = True,
    ) -> None:
        """Initialize Cloud Logging client.

        Args:
            project_id: GCP project ID. If None, uses default credentials.
            log_name: Name of the log in Cloud Logging.
            fallback_to_stdout: If True, logs to stdout if Cloud Logging unavailable.
        """
        self.project_id = project_id or get_settings().gcp.project_id
        self.log_name = log_name
        self.fallback_to_stdout = fallback_to_stdout
        self._client = None
        self._logger = None
        self._local_logger = structlog.get_logger(__name__)

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Google Cloud Logging client."""
        if not CLOUD_LOGGING_AVAILABLE:
            self._local_logger.warning(
                "cloud_logging_unavailable",
                message="google-cloud-logging not installed, using local logging",
                fallback_enabled=self.fallback_to_stdout,
            )
            return

        try:
            client = cloud_logging.Client(project=self.project_id)
            self._client = client
            self._logger = client.logger(self.log_name)
            self._local_logger.info(
                "cloud_logging_initialized",
                project_id=self.project_id,
                log_name=self.log_name,
                retention_days=AUDIT_RETENTION_DAYS,
            )
        except Exception as e:
            self._local_logger.error(
                "cloud_logging_init_failed", error=str(e), fallback_enabled=self.fallback_to_stdout
            )

    def _format_log_entry(
        self, event: str, severity: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Format log entry with required audit metadata.

        Args:
            event: Event name (e.g., "api_request", "user_login")
            severity: Log severity level
            metadata: Additional metadata to include

        Returns:
            Formatted log entry with audit metadata
        """
        settings = get_settings()
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "severity": severity,
            "service": "ollama-api",
            "environment": settings.environment,
            "version": settings.version,
            **metadata,
        }

    def log_structured(self, event: str, severity: str = "INFO", **metadata: Any) -> None:
        """Log structured event to Cloud Logging.

        Args:
            event: Event name
            severity: Log severity (DEBUG, INFO, WARNING, ERROR, CRITICAL, SECURITY)
            **metadata: Additional metadata fields
        """
        log_entry = self._format_log_entry(event, severity, metadata)

        # Log to Cloud Logging if available
        if self._logger:
            try:
                self._logger.log_struct(log_entry, severity=SEVERITY_LEVELS.get(severity, "INFO"))
            except Exception as e:
                self._local_logger.error("cloud_logging_write_failed", error=str(e), event=event)

        # Fallback to local logging
        if self.fallback_to_stdout:
            self._local_logger.info(event, **metadata)

    def log_api_request(
        self,
        user_id: str | None,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Log API request for audit trail.

        Args:
            user_id: User ID making the request (or "anonymous")
            endpoint: API endpoint path
            method: HTTP method
            status_code: HTTP status code
            latency_ms: Request latency in milliseconds
            ip_address: Client IP address
            user_agent: Client user agent
            request_id: Unique request ID
            **extra: Additional metadata
        """
        severity = "WARNING" if status_code >= 400 else "INFO"

        self.log_structured(
            event="api_request",
            severity=severity,
            user_id=user_id or "anonymous",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            **extra,
        )

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str | None = None,
        success: bool = True,
        **extra: Any,
    ) -> None:
        """Log user action for audit trail.

        Args:
            user_id: User ID performing the action
            action: Action performed (e.g., "create_conversation", "delete_model")
            resource: Resource affected by the action
            success: Whether the action succeeded
            **extra: Additional metadata
        """
        severity = "INFO" if success else "WARNING"

        self.log_structured(
            event="user_action",
            severity=severity,
            user_id=user_id,
            action=action,
            resource=resource,
            success=success,
            **extra,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        user_id: str | None = None,
        ip_address: str | None = None,
        description: str | None = None,
        **extra: Any,
    ) -> None:
        """Log security event for audit trail.

        Args:
            event_type: Security event type (e.g., "auth_failure", "rate_limit_exceeded")
            severity: Event severity (WARNING, ERROR, CRITICAL, SECURITY)
            user_id: User ID involved in the event
            ip_address: Client IP address
            description: Human-readable description
            **extra: Additional metadata
        """
        self.log_structured(
            event="security_event",
            severity=severity,
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            description=description,
            **extra,
        )

    def log_model_inference(
        self,
        user_id: str,
        model_name: str,
        prompt_length: int,
        response_length: int,
        latency_ms: float,
        tokens_used: int | None = None,
        **extra: Any,
    ) -> None:
        """Log model inference for audit trail.

        Args:
            user_id: User ID making the inference request
            model_name: Name of the model used
            prompt_length: Length of the prompt in characters
            response_length: Length of the response in characters
            latency_ms: Inference latency in milliseconds
            tokens_used: Number of tokens used
            **extra: Additional metadata
        """
        self.log_structured(
            event="model_inference",
            severity="INFO",
            user_id=user_id,
            model_name=model_name,
            prompt_length=prompt_length,
            response_length=response_length,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            **extra,
        )


# ============================================================================
# Global Logger Instance
# ============================================================================

_audit_logger: CloudAuditLogger | None = None


def get_audit_logger() -> CloudAuditLogger:
    """Get global Cloud Audit Logger instance.

    Returns:
        Singleton CloudAuditLogger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = CloudAuditLogger()
    return _audit_logger


# ============================================================================
# Decorator for Automatic Request Logging
# ============================================================================


def log_api_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to automatically log API calls.

    Example:
        @log_api_call
        async def my_endpoint(request: Request):
            # Endpoint logic
            return {"status": "ok"}
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        logger = get_audit_logger()

        # Extract request context
        request = kwargs.get("request") or (args[0] if args else None)
        user_id = getattr(request.state, "user_id", None) if request else None
        request_id = getattr(request.state, "request_id", None) if request else None

        try:
            # Execute endpoint
            response = await func(*args, **kwargs)

            # Log successful request
            latency_ms = (time.time() - start_time) * 1000
            logger.log_api_request(
                user_id=user_id,
                endpoint=request.url.path if request else "unknown",
                method=request.method if request else "unknown",
                status_code=200,
                latency_ms=latency_ms,
                request_id=request_id,
            )

            return response

        except Exception as e:
            # Log failed request
            latency_ms = (time.time() - start_time) * 1000
            logger.log_api_request(
                user_id=user_id,
                endpoint=request.url.path if request else "unknown",
                method=request.method if request else "unknown",
                status_code=500,
                latency_ms=latency_ms,
                request_id=request_id,
                error=str(e),
            )
            raise

    return wrapper
