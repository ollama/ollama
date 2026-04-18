from ollama.auth.zero_trust import ZeroTrustManager


def test_zero_trust_manager_validate_identity():
    mgr = ZeroTrustManager()
    claims = mgr.validate_identity("dummy-token")
    assert isinstance(claims, dict)
    assert "sub" in claims


def test_enforce_policy_defaults():
    mgr = ZeroTrustManager()
    claims = {"sub": "svc", "roles": ["service"]}
    assert mgr.enforce_policy(claims, resource="/api/resource", action="read") is True
    assert mgr.enforce_policy(claims, resource="/api/resource", action="write") is False
