"""Authentication implementation containers."""

from .firebase_auth import (
    get_current_user,
    get_or_create_user,
    init_firebase,
    require_role,
    require_root_admin,
    revoke_user_tokens,
)
from .manager import AuthManager
from .middleware import require_auth, verify_token_optional

__all__ = [
    "init_firebase",
    "get_current_user",
    "get_or_create_user",
    "require_role",
    "require_root_admin",
    "revoke_user_tokens",
    "require_auth",
    "verify_token_optional",
    "verify_token_optional",
    "AuthManager",
]
