"""Compatibility wrapper for `cost.service` used by tests.

Re-exports symbols from `ollama.services.cost.service`.
"""

from ollama.services.cost.service import *  # noqa: F403

__all__ = [
    "CostManagementService",
]
