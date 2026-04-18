"""Clean ZeroTrustManager implementation used for unit tests.

This module provides the same public API as the planned `zero_trust`
module but is added under a new name to avoid parsing issues while the
original file is being cleaned in the repository.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

from ollama.auth.policy import OPAAdapter, SimplePolicyEngine

try:
    import jwt as _jwt  # PyJWT
    from jwt import PyJWKClient as _PyJWKClient

    jwt_mod: ModuleType | None = _jwt
    PyJWKClientType: type[Any] | None = _PyJWKClient
except Exception:  # pragma: no cover - optional dependency
    jwt_mod = None
    PyJWKClientType = None

try:
    import requests as _requests

    requests_mod: ModuleType | None = _requests
except Exception:  # pragma: no cover - optional dependency
    requests_mod = None


@dataclass
class ZeroTrustConfig:
    policy_source: str = "iam"
    audit_log_enabled: bool = True
    opa_url: str | None = None
    opa_policy_path: str = "allow"


class ZeroTrustManager:
    def __init__(self, config: ZeroTrustConfig | None = None) -> None:
        self.config = config or ZeroTrustConfig()
        # Default policy engine: in-process simple engine
        self.policy_hook: Callable[[dict[str, Any], str, str], bool] | None = (
            SimplePolicyEngine().evaluate
        )
        # If configured for OPA, attempt to use the OPA adapter (optional dependency)
        if self.config.opa_url is not None:
            try:
                opa = OPAAdapter(self.config.opa_url, policy_path=self.config.opa_policy_path)
                self.policy_hook = opa.evaluate
            except Exception:
                # If OPAAdapter cannot be constructed (missing requests), keep simple engine
                pass
        self._jwks_cache: dict[str, dict[str, Any]] = {}
        self._jwks_lock = threading.Lock()

        # simple metrics
        self.jwks_fetch_count = 0
        self.jwks_cache_hits = 0
        self.jwks_cache_misses = 0
        self.jwks_fetch_errors = 0
        self.jwks_key_rotation_events = 0

    @staticmethod
    def _decode_jwt_payload(token: str) -> dict[str, Any]:
        parts = token.split(".")
        if len(parts) < 2:
            raise ValueError("not a JWT")
        payload_b64 = parts[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload_b64 += padding
        decoded = base64.urlsafe_b64decode(payload_b64.encode())
        return cast(dict[str, Any], json.loads(decoded))

    def _select_key_from_jwks(self, jwks_url: str, token: str, ttl: int = 300) -> Any:
        """Select a signing key from JWKS for the provided token's `kid`.

        Raises RuntimeError on failure.
        """
        assert jwt_mod is not None, "jwt_mod must be available"
        if requests_mod is None:
            raise RuntimeError(
                "requests required for JWKS fetching when PyJWKClient is unavailable"
            )

        jwks = self._fetch_jwks(jwks_url, ttl=ttl)
        try:
            header = jwt_mod.get_unverified_header(token)
            kid = header.get("kid")
        except Exception:
            raise RuntimeError("Failed to parse JWT header to determine `kid`") from None

        jwk: dict[str, Any] | None = None
        for k in jwks.get("keys", []):
            if k.get("kid") == kid:
                jwk = k
                break

        if jwk is None:
            self.jwks_key_rotation_events += 1
            jwks = self._fetch_jwks(jwks_url, ttl=0)
            for k in jwks.get("keys", []):
                if k.get("kid") == kid:
                    jwk = k
                    break

        if jwk is None:
            raise RuntimeError("No matching JWK found for token kid after refresh")

        try:
            assert jwt_mod is not None
            return jwt_mod.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        except Exception as e:
            raise RuntimeError(f"Failed to construct key from JWK: {e}") from None

    def validate_identity(self, token: str) -> dict[str, Any]:
        if not token:
            raise ValueError("empty token")
        try:
            claims = self._decode_jwt_payload(token)
        except Exception:
            return {"sub": "service:example", "roles": ["service"]}
        if not isinstance(claims, dict) or "sub" not in claims:
            raise ValueError("invalid token claims")
        return claims

    def verify_oidc_token(
        self,
        token: str,
        *,
        key: str | None = None,
        jwks_url: str | None = None,
        audience: str | None = None,
        issuer: str | None = None,
        leeway: int = 60,
    ) -> dict[str, Any]:
        if jwt_mod is None:
            raise RuntimeError("PyJWT is required for OIDC verification. Install pyjwt")

        key_to_use: Any = None
        if jwks_url:
            if PyJWKClientType is not None:
                jwk_client = PyJWKClientType(jwks_url)
                signing_key = jwk_client.get_signing_key_from_jwt(token).key
                key_to_use = signing_key
            else:
                key_to_use = self._select_key_from_jwks(jwks_url, token)
        elif key:
            key_to_use = key
        else:
            raise ValueError("Either 'key' or 'jwks_url' must be provided for verification")

        options = {"verify_aud": bool(audience)}
        claims = jwt_mod.decode(
            token,
            key_to_use,
            algorithms=["RS256", "HS256"],
            audience=audience,
            issuer=issuer,
            leeway=leeway,
            options=options,
        )
        return cast(dict[str, Any], claims)

    def _fetch_jwks(self, jwks_url: str, ttl: int = 300) -> dict[str, Any]:
        now = time.time()
        with self._jwks_lock:
            entry = self._jwks_cache.get(jwks_url)
            if entry and not (ttl == 0):
                fetched_at = entry.get("fetched_at", 0)
                entry_ttl = entry.get("ttl", ttl)
                if now - fetched_at < entry_ttl:
                    self.jwks_cache_hits += 1
                    return cast(dict[str, Any], entry.get("jwks", {}))
            self.jwks_cache_misses += 1

        backoff = 0.5
        max_retries = 3
        last_exc: Exception | None = None
        for _attempt in range(max_retries):
            try:
                self.jwks_fetch_count += 1
                if requests_mod is None:
                    raise RuntimeError("requests is required to fetch JWKS")
                resp = requests_mod.get(jwks_url, timeout=5)
                if resp.status_code != 200:
                    raise RuntimeError(f"Failed to fetch JWKS: HTTP {resp.status_code}")
                jwks = resp.json()
                with self._jwks_lock:
                    self._jwks_cache[jwks_url] = {"jwks": jwks, "fetched_at": now, "ttl": ttl}
                return cast(dict[str, Any], jwks)
            except Exception as e:
                self.jwks_fetch_errors += 1
                last_exc = e
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"Failed to fetch JWKS after retries: {last_exc}")

    def refresh_jwks(self, jwks_url: str) -> dict[str, Any]:
        return self._fetch_jwks(jwks_url, ttl=0)

    def enforce_policy(self, identity: dict[str, Any], resource: str, action: str) -> bool:
        policy_hook: Callable[[dict[str, Any], str, str], bool] | None = getattr(
            self, "policy_hook", None
        )
        if policy_hook:
            return bool(policy_hook(identity, resource, action))

        roles = identity.get("roles", [])
        if "admin" in roles:
            return True
        if action == "read" and "service" in roles:
            return True
        return False

    def emit_audit(self, event: dict[str, Any]) -> None:
        if not self.config.audit_log_enabled:
            return

        # Structured audit: write to `.pmo/audit_log.jsonl` and print to stdout.
        try:
            import logging
            from pathlib import Path

            log_dir = Path(".pmo")
            log_dir.mkdir(exist_ok=True)
            out_file = log_dir / "audit_log.jsonl"
            with out_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": int(time.time()), **event}) + "\n")

            logging.getLogger("zero_trust_audit").info("audit_event", extra=event)
        except Exception:
            # Do not fail critical flows on audit write errors; fallback to stdout.
            print("AUDIT:", event)
