from ollama.auth.policy import OPAAdapter, SimplePolicyEngine


def test_simple_policy_engine_allows_admin():
    engine = SimplePolicyEngine()
    identity = {"sub": "user:1", "roles": ["admin"]}
    assert engine.evaluate(identity, "any", "delete") is True


def test_simple_policy_engine_service_read():
    engine = SimplePolicyEngine()
    identity = {"sub": "svc:1", "roles": ["service"]}
    assert engine.evaluate(identity, "resource", "read") is True
    assert engine.evaluate(identity, "resource", "write") is False


def test_opa_adapter_calls_requests(monkeypatch):
    calls = []

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return DummyResp({"result": True})

    monkeypatch.setattr("ollama.auth.policy.requests", type("M", (), {"post": fake_post}))

    adapter = OPAAdapter("http://opa:8181", policy_path="example/allow")
    ok = adapter.evaluate({"sub": "u"}, "res", "act")
    assert ok is True
    assert len(calls) == 1
