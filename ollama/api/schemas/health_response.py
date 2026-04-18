"""Schema: HealthResponse for server health endpoint."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Server health status."""

    status: str
    version: str
    environment: str
    public_url: str | None = None
