import pytest

jwt = pytest.importorskip("jwt")

from ollama.auth.zero_trust import ZeroTrustManager


def test_verify_oidc_token_with_hs256():
    mgr = ZeroTrustManager()
    secret = "test-secret"
    payload = {"sub": "test-sub", "roles": ["service"]}
    # PyJWT encode may return bytes or str depending on version
    token = jwt.encode(payload, secret, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode()

    claims = mgr.verify_oidc_token(token, key=secret)
    assert claims["sub"] == "test-sub"
    assert "roles" in claims
