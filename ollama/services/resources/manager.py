"""Resource manager implementation.

Coordinates GPU and memory allocation between inference and training workloads.
"""

import asyncio

import structlog

from ollama.services.resources.types import WorkloadType

log = structlog.get_logger(__name__)


class ResourceManager:
    """Arbitrates access to local hardware resources.

    Ensures that high-priority or resource-intensive tasks do not collide.
    In local dev, this usually means exclusive access to the GPU for training.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_workload: WorkloadType | None = None

    async def acquire(self, workload: WorkloadType, timeout: float = 30.0) -> bool:
        """Attempt to acquire exclusive resource access for a workload.

        Args:
            workload: The type of workload requesting access.
            timeout: Maximum time to wait for resources.

        Returns:
            True if access granted, False otherwise.
        """
        log.info("resource_request", workload=workload)

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            self._active_workload = workload
            log.info("resource_acquired", workload=workload)
            return True
        except asyncio.TimeoutError:
            log.warning("resource_acquisition_timeout", workload=workload)
            return False

    def release(self) -> None:
        """Release currently held resources."""
        if self._lock.locked():
            log.info("resource_released", workload=self._active_workload)
            self._active_workload = None
            self._lock.release()

    @property
    def is_locked(self) -> bool:
        """Check if resources are currently locked."""
        return self._lock.locked()

    @property
    def current_workload(self) -> WorkloadType | None:
        """Get the type of the currently active workload."""
        return self._active_workload
