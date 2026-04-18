"""Resource manager dependency."""

from typing import Annotated

from fastapi import Depends

from ollama.services.resources.manager import ResourceManager


async def get_resource_manager() -> ResourceManager | None:
    """Dependency for ResourceManager.

    Returns the global ResourceManager instance initialized at startup.
    """
    from ollama.main import get_resource_manager as main_get_resource_manager

    try:
        return await main_get_resource_manager()
    except Exception:
        return None


ResourceDependency = Annotated[ResourceManager | None, Depends(get_resource_manager)]
