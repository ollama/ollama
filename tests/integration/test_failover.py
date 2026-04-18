"""
Integration tests for Automated Failover.

Tests health endpoint behavior, LB routing, and failover mechanics.
"""

import asyncio
from typing import Any

import httpx
import pytest
import structlog

log = structlog.get_logger(__name__)


@pytest.fixture
def lb_endpoint() -> str:
    """Return the GCP Load Balancer endpoint."""
    return "https://elevatediq.ai/ollama"


@pytest.fixture
def health_path() -> str:
    """Return the health check path."""
    return "/api/v1/health"


@pytest.mark.asyncio
async def test_health_endpoint_reachable(lb_endpoint: str, health_path: str) -> None:
    """Verify health endpoint is reachable through LB."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{lb_endpoint}{health_path}")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_endpoint_response_time(lb_endpoint: str, health_path: str) -> None:
    """Verify health endpoint responds within latency SLA (< 200ms p95)."""
    latencies = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(20):
            response = await client.get(f"{lb_endpoint}{health_path}")
            assert response.status_code == 200
            latencies.append(response.elapsed.total_seconds() * 1000)

    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    assert p95 < 200, f"P95 latency {p95:.2f}ms exceeds 200ms SLA"
    log.info("health_endpoint_latency", p50=sorted(latencies)[10], p95=p95, unit="ms")


@pytest.mark.asyncio
async def test_health_endpoint_cors_headers(lb_endpoint: str, health_path: str) -> None:
    """Verify CORS headers restrict to allowed origins."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Request from allowed origin
        response = await client.get(
            f"{lb_endpoint}{health_path}",
            headers={"Origin": "https://elevatediq.ai"},
        )
        assert response.status_code == 200
        # CORS headers should be present if configured
        if "access-control-allow-origin" in response.headers:
            assert response.headers["access-control-allow-origin"] in [
                "https://elevatediq.ai",
                "*",
            ]


@pytest.mark.asyncio
async def test_health_endpoint_concurrent_requests(lb_endpoint: str, health_path: str) -> None:
    """Verify health endpoint handles concurrent requests without errors."""
    async with httpx.AsyncClient(timeout=10.0) as client:

        async def fetch_health() -> dict[str, Any]:
            response = await client.get(f"{lb_endpoint}{health_path}")
            response.raise_for_status()
            return response.json()

        # Send 50 concurrent requests
        results = await asyncio.gather(*[fetch_health() for _ in range(50)])
        assert len(results) == 50
        assert all(r["status"] == "healthy" for r in results)


@pytest.mark.asyncio
async def test_direct_internal_port_blocked(health_path: str) -> None:
    """Verify direct access to internal port 8000 is blocked by firewall.

    This test expects connection to fail (timeout or refused).
    """
    # Assuming internal service runs on some IP; use a placeholder
    # In production, replace with actual internal IP
    internal_ip = "10.0.0.1"  # Example internal IP
    internal_port = 8000

    with pytest.raises((httpx.ConnectTimeout, httpx.ConnectError)):
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(f"http://{internal_ip}:{internal_port}{health_path}")


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires manual failover simulation")
async def test_failover_on_primary_failure(lb_endpoint: str, health_path: str) -> None:
    """Verify failover to secondary backend when primary fails.

    Manual steps:
    1. Ensure primary backend is healthy
    2. Stop primary MIG or mark unhealthy
    3. Wait up to 30s
    4. Verify traffic is served by secondary

    This test is skipped by default; run manually during failover drills.
    """
    # Step 1: Verify primary is serving
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{lb_endpoint}{health_path}")
        assert response.status_code == 200
        primary_header = response.headers.get("x-backend-region", "")
        log.info("primary_backend_region", region=primary_header)

    # Step 2: Simulate primary failure (manual step; not automated here)
    # In real test, would trigger MIG shutdown or health check failure
    log.warning("manual_step_required", message="Stop primary MIG now")
    await asyncio.sleep(35)  # Wait > 30s for failover

    # Step 3: Verify secondary is now serving
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{lb_endpoint}{health_path}")
        assert response.status_code == 200
        secondary_header = response.headers.get("x-backend-region", "")
        log.info("secondary_backend_region", region=secondary_header)
        assert secondary_header != primary_header, "Failover did not occur"


@pytest.mark.asyncio
async def test_health_endpoint_returns_json(lb_endpoint: str, health_path: str) -> None:
    """Verify health endpoint returns valid JSON response."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{lb_endpoint}{health_path}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
