"""FastAPI endpoints to control the ProfilerManager in development.

Endpoints (mounted under `/monitoring`):
 - POST `/profiler/start`  -> start profiler
 - POST `/profiler/stop`   -> stop profiler and return stats
 - GET  `/profiler/status` -> return whether profiler is running

These endpoints are intended for local development and diagnostics only.
Do NOT expose them in production without proper auth and access controls.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException
from ollama.monitoring.profiling import ProfilerManager
from pydantic import BaseModel

router = APIRouter()

# singleton manager used by endpoints
_mgr = ProfilerManager()


class StartRequest(BaseModel):
    enable_tracemalloc: bool = True


@router.post("/profiler/start")
async def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Verify developer API key for profiler endpoints.

    Uses `DEV_PROFILER_API_KEY` env var. If not set, endpoints are disabled.
    """
    expected = os.getenv("DEV_PROFILER_API_KEY")
    if expected is None:
        raise HTTPException(status_code=403, detail="Profiler endpoints are disabled")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@router.post("/profiler/start")
async def start_profiler(req: StartRequest, _=Depends(_verify_api_key)) -> dict:
    if _mgr.is_running():
        raise HTTPException(status_code=409, detail="Profiler already running")
    _mgr.start(enable_tracemalloc=req.enable_tracemalloc)
    return {"status": "started"}


@router.post("/profiler/stop")
async def stop_profiler(_=Depends(_verify_api_key)) -> dict:
    if not _mgr.is_running():
        raise HTTPException(status_code=409, detail="Profiler not running")
    stats = _mgr.stop_and_get_stats()
    return {"status": "stopped", "stats": stats}


@router.get("/profiler/status")
async def profiler_status(_=Depends(_verify_api_key)) -> dict:
    return {"running": _mgr.is_running()}


__all__ = ["router"]
