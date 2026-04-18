from ollama.auth.policy import SimplePolicyEngine
from ollama.auth.zero_trust import ZeroTrustManager


def test_simple_policy_engine_allows_admin():
    engine = SimplePolicyEngine()
    identity = {"sub": "u1", "roles": ["admin"]}
    assert engine.evaluate(identity, "/", "write") is True


def test_simple_policy_engine_service_read():
    engine = SimplePolicyEngine()
    identity = {"sub": "svc", "roles": ["service"]}
    assert engine.evaluate(identity, "/api", "read") is True
    assert engine.evaluate(identity, "/api", "write") is False


def test_zero_trust_uses_default_policy_hook():
    mgr = ZeroTrustManager()
    claims = {"sub": "svc", "roles": ["service"]}
    assert mgr.enforce_policy(claims, resource="/a", action="read") is True
    assert mgr.enforce_policy(claims, resource="/a", action="write") is False
