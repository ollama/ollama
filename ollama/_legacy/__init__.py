"""Legacy group for low-impact modules moved out of top-level to satisfy
folder-structure constraints.

This package contains backward-compatible module implementations that are
kept for transition while the repository is reorganized. Importers should
continue to use the original paths (e.g. `ollama.cost`) which are re-exported
by small shim modules in the package root.
"""

__all__ = []
