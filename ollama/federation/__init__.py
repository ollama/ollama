"""Compatibility package for `ollama.federation` re-exporting moved
implementation under `ollama._legacy.federation`.
"""

from ._legacy.federation import *  # noqa: F401,F403

__all__ = getattr(__import__("ollama._legacy.federation", fromlist=["*"]), "__all__", [])
