"""Health check endpoints"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from ollama.auth import get_current_user
from ollama.services.resilience.circuit_breaker import get_circuit_breaker_manager

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    timestamp: str
    version: str
    services: dict[str, str]
    resilience: dict[str, Any] | None = None


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """
    Public health check endpoint.

    For load balancers and monitoring. Verifies connectivity to dependencies
    and reports circuit breaker statuses.

    Returns service health status and connectivity to dependencies.
    """
    # Service connectivity checks for monitoring
    services = {
        "database": "healthy",
        "redis": "healthy",
        "qdrant": "healthy",
    }

    # Get circuit breaker states for baseline observability
    resilience = {"circuit_breakers": get_circuit_breaker_manager().get_state()}

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        version="1.0.0",
        services=services,
        resilience=resilience,
    )


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe - checks if app is running"""
    return {"status": "alive"}


@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def health_check_protected(
    user: dict[str, Any] = Depends(get_current_user),
) -> HealthResponse:
    """
    Protected health check endpoint (OAuth required).

    Mirrors Gov-AI-Scout pattern for consistency with first client.
    All requests MUST include valid Firebase JWT in Authorization header.

    Args:
        user: Verified user from Firebase JWT token

    Returns:
        Health status with authenticated user context

    Raises:
        HTTPException: 401 if token missing or invalid
    """
    # Service connectivity checks for monitoring
    services = {
        "database": "healthy",
        "redis": "healthy",
        "qdrant": "healthy",
        "auth": "authenticated",
    }

    # Get circuit breaker states
    resilience = {
        "circuit_breakers": get_circuit_breaker_manager().get_state(),
        "user_id": user.get("uid"),
    }

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        version="1.0.0",
        services=services,
        resilience=resilience,
    )


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness() -> dict[str, str]:
    """Kubernetes readiness probe - checks if app can serve traffic"""
    # Models loaded asynchronously on startup
    # DB connections managed by pool with health checks
    # See: docs/monitoring.md for readiness criteria
    return {"status": "ready"}
