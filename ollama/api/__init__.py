"""API layer module.

Organized into functional containers:
- routes/: HTTP endpoint handlers
- schemas/: Pydantic request/response models
- dependencies/: FastAPI dependency injection

All client requests are routed through the API layer, which validates input,
delegates to services, and returns formatted responses.
"""

__all__ = ["routes", "schemas", "dependencies"]
