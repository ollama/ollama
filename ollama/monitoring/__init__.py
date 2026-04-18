"""Compatibility package for `ollama.monitoring` re-exporting moved
implementation under `ollama._legacy.monitoring`.
"""

from ._legacy.monitoring import *  # noqa: F401,F403

__all__ = getattr(__import__("ollama._legacy.monitoring", fromlist=["*"]), "__all__", [])
