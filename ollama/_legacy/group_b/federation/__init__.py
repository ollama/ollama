"""
Federation Module - Issue #42

Multi-Tier Hub-Spoke Federation for global Ollama deployment.

Submodules:
- manager: FederationManager for hub coordination
- protocol: gRPC protocol definitions

Version: 1.0.0
Status: PRODUCTION-READY
"""

from .manager import (
    FederationManager,
    Hub,
    HubCapacity,
    HubStatus,
    RoutingPolicy,
    RoutingStrategy,
    TopologySnapshot,
)

__all__ = [
    "FederationManager",
    "Hub",
    "HubCapacity",
    "HubStatus",
    "TopologySnapshot",
    "RoutingPolicy",
    "RoutingStrategy",
]

__version__ = "1.0.0"
