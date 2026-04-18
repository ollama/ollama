"""Security dependency helpers."""

import logging
from typing import Annotated

from fastapi import Depends

from ollama.config import get_settings
from ollama.services.security.dlp_redactor import DLPRedactor

log = logging.getLogger(__name__)

# Global DLP redactor instance
_dlp_redactor: DLPRedactor | None = None


async def get_dlp_redactor() -> DLPRedactor:
    """FastAPI dependency that yields the DLP redactor service."""
    global _dlp_redactor

    if _dlp_redactor is None:
        settings = get_settings()
        _dlp_redactor = DLPRedactor(project_id=settings.gcp.project_id)
        log.info("DLP redactor dependency initialized")

    return _dlp_redactor


DLPDependency = Annotated[DLPRedactor, Depends(get_dlp_redactor)]
