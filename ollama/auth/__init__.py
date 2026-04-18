"""Authentication and Authorization domain.

Top-level imports are attempted but optional dependencies (e.g., FastAPI)
may be unavailable in lightweight test environments. We import lazily and
fall back to a minimal export set when those dependencies are missing so
unit tests that import submodules (like ``zero_trust``) can run without
installing full runtime dependencies.
"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import FastAPI types only for static type checking; avoid runtime dependency.
    from fastapi import Request
else:
    Request = Any

try:
    from .impl.firebase_auth import (
        get_current_user,
        get_or_create_user,
        init_firebase,
        require_role,
        require_root_admin,
        revoke_user_tokens,
    )
    from .impl.manager import AuthManager
    from .impl.middleware import require_auth, verify_token_optional
    from .policy import OPAAdapter, PolicyEngine, SimplePolicyEngine
    from .zero_trust import ZeroTrustManager

    __all__ = [
        "AuthManager",
        "OPAAdapter",
        "PolicyEngine",
        "SimplePolicyEngine",
        "ZeroTrustManager",
        "get_current_user",
        "get_or_create_user",
        "init_firebase",
        "require_auth",
        "require_role",
        "require_root_admin",
        "revoke_user_tokens",
        "verify_token_optional",
    ]
    _auth_available = True
except Exception:  # pragma: no cover - defensive for test environments
    # Provide minimal stubs so unit tests importing top-level symbols do not fail
    def init_firebase(credentials_path: str | None = None) -> None:
        raise RuntimeError("firebase auth not available in this environment")

    async def get_current_user(request: Request, require_auth: bool = True) -> dict[str, Any]:
        raise RuntimeError("get_current_user is not available in lightweight test env")

    def get_or_create_user(email: str, display_name: str | None = None) -> dict[str, Any]:
        raise RuntimeError("get_or_create_user is not available in lightweight test env")

    def require_role(allowed_roles: list[str]) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        async def _stub(user: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("require_role is not available in lightweight test env")

        return _stub

    def require_root_admin(root_admin_email: str) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        async def _stub(user: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("require_root_admin is not available in lightweight test env")

        return _stub

    def revoke_user_tokens(uid: str) -> None:
        raise RuntimeError("revoke_user_tokens is not available in lightweight test env")

    def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("require_auth is not available in lightweight test env")

        return _wrapper

    async def verify_token_optional(request: Request) -> dict[str, Any]:
        raise RuntimeError("verify_token_optional is not available in lightweight test env")

    class AuthManagerStub:  # minimal stub with different name
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AuthManager is not available in lightweight test env")

    AuthManager = AuthManagerStub  # type: ignore[misc,assignment]

    class ZeroTrustManagerStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("ZeroTrustManager is not available in lightweight test env")

    ZeroTrustManager = ZeroTrustManagerStub  # type: ignore[misc,assignment]

    class PolicyEngineStub:
        def evaluate(self, *args: Any, **kwargs: Any) -> bool:
            raise RuntimeError("PolicyEngine is not available in lightweight test env")

    PolicyEngine = PolicyEngineStub # type: ignore
    SimplePolicyEngine = PolicyEngineStub # type: ignore
    OPAAdapter = PolicyEngineStub # type: ignore

    __all__ = [
        "AuthManager",
        "OPAAdapter",
        "PolicyEngine",
        "SimplePolicyEngine",
        "ZeroTrustManager",
        "get_current_user",
        "get_or_create_user",
        "init_firebase",
        "require_auth",
        "require_role",
        "require_root_admin",
        "revoke_user_tokens",
        "verify_token_optional",
    ]
    _auth_available = False
