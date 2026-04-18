import os
from pathlib import Path

"""Compatibility shim for `ollama.agents`.

This package makes the legacy agents directory importable as
`ollama.agents.*` by adding the legacy agents folder to the package
`__path__` so submodule imports resolve correctly.
"""

# Compute the absolute path to `ollama/_legacy/group_a/agents`
_HERE = Path(__file__).resolve().parent
_legacy_agents = (_HERE.parent / "_legacy" / "group_a" / "agents").resolve()

if _legacy_agents.exists():
    # Prepend to __path__ so standard submodule imports find the legacy modules
    __path__.insert(0, str(_legacy_agents))
else:
    # Fall back to normal behavior; import errors will surface for missing files
    pass

__all__ = []
