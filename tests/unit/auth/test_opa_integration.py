from ollama.auth.zero_trust import ZeroTrustConfig, ZeroTrustManager


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeRequests:
    def __init__(self, result):
        self._result = result

    def post(self, url, json=None, timeout=2):
        return _FakeResp({"result": self._result})


def test_opa_adapter_allows_when_opa_returns_true(monkeypatch):
    fake_requests = _FakeRequests(True)
    # Patch the requests object used by the OPAAdapter (in policy module)
    import ollama.auth.policy as policy_mod

    monkeypatch.setattr(policy_mod, "requests", fake_requests)

    cfg = ZeroTrustConfig(opa_url="http://opa.local", opa_policy_path="allow")
    manager = ZeroTrustManager(config=cfg)

    result = manager.enforce_policy({"roles": ["user"]}, "resource:x", "write")
    assert result is True


def test_opa_adapter_denies_when_opa_returns_false(monkeypatch):
    fake_requests = _FakeRequests(False)
    import ollama.auth.policy as policy_mod

    monkeypatch.setattr(policy_mod, "requests", fake_requests)

    cfg = ZeroTrustConfig(opa_url="http://opa.local", opa_policy_path="allow")
    manager = ZeroTrustManager(config=cfg)

    result = manager.enforce_policy({"roles": ["user"]}, "resource:x", "write")
    assert result is False
