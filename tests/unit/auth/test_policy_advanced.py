import pytest

from ollama.auth.policy import OPAAdapter, PolicyEngine, SimplePolicyEngine


def test_base_policy_engine():
    pe = PolicyEngine()
    with pytest.raises(NotImplementedError):
        pe.evaluate({}, "res", "act")


def test_simple_policy_engine():
    engine = SimplePolicyEngine()
    assert engine.evaluate({"roles": ["admin"]}, "any", "any") is True
    assert engine.evaluate({"roles": ["service"]}, "any", "read") is True
    assert engine.evaluate({"roles": ["service"]}, "any", "write") is False
    assert engine.evaluate({}, "any", "any") is False


def test_opa_adapter_init_fail(monkeypatch):
    import ollama.auth.policy as policy

    monkeypatch.setattr(policy, "requests", None)
    with pytest.raises(RuntimeError, match="requests is required"):
        OPAAdapter("http://localhost:8181")


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_opa_adapter_evaluate(monkeypatch):
    import ollama.auth.policy as policy

    class MockRequests:
        def post(self, url, json, timeout):
            assert "identity" in json["input"]
            if json["input"]["identity"].get("allow"):
                return DummyResp({"result": True})
            return DummyResp({"result": False})

    monkeypatch.setattr(policy, "requests", MockRequests())

    adapter = OPAAdapter("http://opa:8181")
    assert adapter.evaluate({"allow": True}, "res", "act") is True
    assert adapter.evaluate({"allow": False}, "res", "act") is False


def test_opa_adapter_no_result(monkeypatch):
    import ollama.auth.policy as policy

    class MockRequests:
        def post(self, url, json, timeout):
            return DummyResp({})  # No result key

    monkeypatch.setattr(policy, "requests", MockRequests())

    adapter = OPAAdapter("http://opa:8181")
    assert adapter.evaluate({}, "res", "act") is False
