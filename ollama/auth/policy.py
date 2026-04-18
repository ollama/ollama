"""Policy engine abstractions and adapters for ZeroTrustManager.

Provides a pluggable `PolicyEngine` interface, a simple in-process
`SimplePolicyEngine` suitable for unit tests and examples, and an optional
`OPAAdapter` that calls a remote OPA server (HTTP) when `requests` is
available.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import requests

try:
    import requests
except Exception:
    requests = None  # type: ignore


class PolicyEngine:
    """Abstract policy engine interface."""

    def evaluate(self, identity: dict[str, Any], resource: str, action: str) -> bool:
        raise NotImplementedError()


class SimplePolicyEngine(PolicyEngine):
    """Minimal in-process policy engine for unit tests.

    Rules:
    - `admin` role: allow everything
    - `service` role: allow `read` actions
    - otherwise: deny
    """

    def evaluate(self, identity: dict[str, Any], resource: str, action: str) -> bool:
        roles = identity.get("roles", [])
        if "admin" in roles:
            return True
        if action == "read" and "service" in roles:
            return True
        return False


class OPAAdapter(PolicyEngine):
    """Adapter that queries a running OPA server using the REST API.

    This adapter is optional and requires the `requests` package. It's
    suitable for production integrations where policies live in OPA.
    """

    def __init__(self, opa_url: str, policy_path: str = "allow") -> None:
        if requests is None:
            raise RuntimeError("requests is required for OPAAdapter")
        self.opa_url = opa_url.rstrip("/")
        self.policy_path = policy_path

    def evaluate(self, identity: dict[str, Any], resource: str, action: str) -> bool:
        url = f"{self.opa_url}/v1/data/{self.policy_path}"
        payload = {"input": {"identity": identity, "resource": resource, "action": action}}
        resp = requests.post(url, json=payload, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        # Expect OPA result under 'result' key
        return bool(data.get("result", False))
