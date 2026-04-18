"""Compatibility package for `ollama.cost` that re-exports the moved
implementation under `ollama._legacy.cost`.

This preserves existing import paths (`from ollama.cost import ...`) while
keeping the package layout compliant with Landing Zone rules.
"""

from ._legacy.cost import *  # noqa: F401,F403

__all__ = getattr(__import__("ollama._legacy.cost", fromlist=["*"]), "__all__", [])
