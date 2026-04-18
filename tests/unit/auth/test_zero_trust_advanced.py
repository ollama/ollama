import json

import pytest

from ollama.auth.zero_trust import ZeroTrustConfig, ZeroTrustManager


class DummyResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_emit_audit_persistence(tmp_path, monkeypatch):
    # Change working directory to tmp_path for isolated audit log testing
    monkeypatch.chdir(tmp_path)

    config = ZeroTrustConfig(audit_log_enabled=True)
    manager = ZeroTrustManager(config=config)

    event = {"action": "login", "user": "test-user"}
    manager.emit_audit(event)

    audit_file = tmp_path / ".pmo" / "audit_log.jsonl"
    assert audit_file.exists()

    content = audit_file.read_text()
    data = json.loads(content)
    assert data["action"] == "login"
    assert data["user"] == "test-user"
    assert "ts" in data


def test_enforce_policy_roles():
    manager = ZeroTrustManager()

    # Admin role should allow everything
    assert manager.enforce_policy({"roles": ["admin"]}, "any", "any") is True

    # Service role should allow read
    assert manager.enforce_policy({"roles": ["service"]}, "any", "read") is True
    assert manager.enforce_policy({"roles": ["service"]}, "any", "write") is False

    # No roles should deny
    assert manager.enforce_policy({}, "any", "read") is False


def test_select_key_from_jwks_success(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/jwks"
    token = "header.payload.signature"  # Fake JWT

    # Mock jwt_mod.get_unverified_header
    class MockJWT:
        def get_unverified_header(self, t):
            return {"kid": "key1"}

        class algorithms:
            class RSAAlgorithm:
                @staticmethod
                def from_jwk(j):
                    return "mock-key-object"

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())

    # Mock requests_mod.get
    def mock_get(url, timeout=5):
        return DummyResp(200, {"keys": [{"kid": "key1", "kty": "RSA"}]})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)

    key = manager._select_key_from_jwks(jwks_url, token)
    assert key == "mock-key-object"


def test_select_key_from_jwks_no_match_retry(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/jwks"
    token = "header.payload.signature"

    # Mock jwt_mod.get_unverified_header
    class MockJWT:
        def get_unverified_header(self, t):
            return {"kid": "new-key"}

        class algorithms:
            class RSAAlgorithm:
                @staticmethod
                def from_jwk(j):
                    return "new-key-object"

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())

    calls = []

    def mock_get(url, timeout=5):
        calls.append(url)
        # First call doesn't have the key, second (refresh) does
        if len(calls) == 1:
            return DummyResp(200, {"keys": [{"kid": "old-key"}]})
        return DummyResp(200, {"keys": [{"kid": "new-key"}]})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)

    key = manager._select_key_from_jwks(jwks_url, token)
    assert key == "new-key-object"
    assert manager.jwks_key_rotation_events == 1


def test_verify_oidc_token_via_jwks(monkeypatch):
    manager = ZeroTrustManager()
    token = "header.payload.signature"
    jwks_url = "https://example.com/jwks"

    class MockKey:
        def __init__(self):
            self.key = "mock-signing-key"

    class MockJWKClient:
        def __init__(self, url):
            self.url = url

        def get_signing_key_from_jwt(self, t):
            return MockKey()

    class MockJWT:
        def decode(self, t, key, **kwargs):
            assert key == "mock-signing-key"
            return {"sub": "user1"}

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())
    monkeypatch.setattr("ollama.auth.zero_trust.PyJWKClientType", MockJWKClient)

    claims = manager.verify_oidc_token(token, jwks_url=jwks_url)
    assert claims["sub"] == "user1"


def test_fetch_jwks_retries_and_failure(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/fail"

    def mock_get(url, timeout=5):
        raise RuntimeError("network error")

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)
    monkeypatch.setattr("time.sleep", lambda x: None)  # Fast sleep

    with pytest.raises(RuntimeError, match="after retries"):
        manager._fetch_jwks(jwks_url)
    assert manager.jwks_fetch_count == 3
    assert manager.jwks_fetch_errors == 3


def test_decode_jwt_payload_error():
    with pytest.raises(ValueError, match="not a JWT"):
        ZeroTrustManager._decode_jwt_payload("not-a-jwt")


def test_select_key_from_jwks_header_fail(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/jwks"

    # Mock requests to return keys
    def mock_get(url, timeout=5):
        return DummyResp(200, {"keys": []})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)

    # Mock header fail
    class MockJWT:
        def get_unverified_header(self, t):
            raise Exception("parse error")

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())

    with pytest.raises(RuntimeError, match="Failed to parse JWT header"):
        manager._select_key_from_jwks(jwks_url, "token")


def test_select_key_from_jwks_no_match_after_refresh(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/jwks"
    token = "header.payload.signature"

    class MockJWT:
        def get_unverified_header(self, t):
            return {"kid": "unknown"}

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())

    def mock_get(url, timeout=5):
        return DummyResp(200, {"keys": [{"kid": "some-other-key"}]})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)

    with pytest.raises(RuntimeError, match="No matching JWK found"):
        manager._select_key_from_jwks(jwks_url, token)


def test_select_key_from_jwks_forced_path(monkeypatch):
    """Test verification when PyJWKClientType is None, forcing use of _select_key_from_jwks."""
    manager = ZeroTrustManager()
    token = "header.payload.signature"
    jwks_url = "https://example.com/jwks"

    monkeypatch.setattr("ollama.auth.zero_trust.PyJWKClientType", None)

    def mock_select(url, t):
        return "fallback-key"

    monkeypatch.setattr(manager, "_select_key_from_jwks", mock_select)

    class MockJWT:
        def decode(self, t, key, **kwargs):
            return {"sub": "user2"}

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())

    claims = manager.verify_oidc_token(token, jwks_url=jwks_url)
    assert claims["sub"] == "user2"


def test_verify_oidc_token_value_errors():
    manager = ZeroTrustManager()
    with pytest.raises(ValueError, match="must be provided"):
        manager.verify_oidc_token("token")


def test_validate_identity_edge_cases():
    manager = ZeroTrustManager()
    # Invalid claims
    with pytest.raises(ValueError, match="invalid token claims"):
        manager.validate_identity(
            "header.eyAibm9fc3ViIjogMSB9.signature"
        )  # Base64 for {"no_sub": 1}


def test_opa_adapter_init_failure(monkeypatch):
    # If OPA is configured but adapter fails (e.g. requests missing)
    config = ZeroTrustConfig(opa_url="http://opa:8181")

    # Temporarily set requests_mod to None to trigger exception in OPAAdapter if we can
    # But OPAAdapter is in policy.py. Let's mock it to raise exception.
    def mock_opa_init(*args, **kwargs):
        raise RuntimeError("OPA missing")

    # We need to mock the OPAAdapter CLASS in ollama.auth.zero_trust
    monkeypatch.setattr("ollama.auth.zero_trust.OPAAdapter", mock_opa_init)

    manager = ZeroTrustManager(config=config)
    # Should fallback to SimplePolicyEngine
    assert manager.policy_hook is not None


def test_select_key_from_jwks_construct_fail(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/jwks"
    token = "header.payload.signature"

    class MockRSA:
        @staticmethod
        def from_jwk(j):
            raise Exception("bad jwk")

    class MockJWT:
        def get_unverified_header(self, t):
            return {"kid": "key1"}

        class algorithms:
            RSAAlgorithm = MockRSA

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())

    def mock_get(url, timeout=5):
        return DummyResp(200, {"keys": [{"kid": "key1"}]})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)

    with pytest.raises(RuntimeError, match="Failed to construct key"):
        manager._select_key_from_jwks(jwks_url, token)


def test_validate_identity_token_empty():
    manager = ZeroTrustManager()
    with pytest.raises(ValueError, match="empty token"):
        manager.validate_identity("")


def test_validate_identity_decode_fallback():
    manager = ZeroTrustManager()
    # Decoding "not-a-jwt" should trigger the except block
    claims = manager.validate_identity("not-a-jwt")
    assert claims["sub"] == "service:example"


def test_verify_oidc_token_with_key(monkeypatch):
    manager = ZeroTrustManager()

    class MockJWT:
        def decode(self, t, key, **kwargs):
            assert key == "my-secret"
            return {"sub": "user3"}

    monkeypatch.setattr("ollama.auth.zero_trust.jwt_mod", MockJWT())

    claims = manager.verify_oidc_token("token", key="my-secret")
    assert claims["sub"] == "user3"


def test_fetch_jwks_cache_hit(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/jwks"

    def mock_get(url, timeout=5):
        return DummyResp(200, {"keys": []})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)

    # First fetch (miss)
    manager._fetch_jwks(jwks_url)
    assert manager.jwks_cache_misses == 1

    # Second fetch (hit)
    manager._fetch_jwks(jwks_url)
    assert manager.jwks_cache_hits == 1


def test_fetch_jwks_requests_missing(monkeypatch):
    manager = ZeroTrustManager()
    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod", None)
    with pytest.raises(RuntimeError, match="requests is required"):
        manager._fetch_jwks("url")


def test_fetch_jwks_bad_status(monkeypatch):
    manager = ZeroTrustManager()

    def mock_get(url, timeout=5):
        return DummyResp(500, {})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)
    monkeypatch.setattr("time.sleep", lambda x: None)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        manager._fetch_jwks("url")


def test_refresh_jwks(monkeypatch):
    manager = ZeroTrustManager()
    calls = []

    def mock_get(url, timeout=5):
        calls.append(url)
        return DummyResp(200, {"keys": []})

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", mock_get)

    manager.refresh_jwks("url")
    assert len(calls) == 1


def test_enforce_policy_no_hook():
    manager = ZeroTrustManager()
    manager.policy_hook = None

    # Service role read
    assert manager.enforce_policy({"roles": ["service"]}, "res", "read") is True
    # Service role write (deny)
    assert manager.enforce_policy({"roles": ["service"]}, "res", "write") is False
    # None
    assert manager.enforce_policy({}, "res", "read") is False


def test_emit_audit_disabled():
    config = ZeroTrustConfig(audit_log_enabled=False)
    manager = ZeroTrustManager(config=config)
    # Should return early
    manager.emit_audit({"event": "none"})


def test_emit_audit_exception_fallback(monkeypatch, capsys):
    manager = ZeroTrustManager()

    # Mock Path.mkdir to raise exception
    def mock_mkdir(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("pathlib.Path.mkdir", mock_mkdir)

    manager.emit_audit({"action": "test"})
    captured = capsys.readouterr()
    assert "AUDIT: {'action': 'test'}" in captured.out
