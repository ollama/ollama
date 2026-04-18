from ollama.auth.zero_trust import ZeroTrustManager


class DummyResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_fetch_jwks_cache_and_hits(monkeypatch):
    manager = ZeroTrustManager()

    jwks_url = "https://example.com/.well-known/jwks.json"
    sample = {"keys": [{"kid": "k1", "kty": "RSA"}]}

    # First call: requests.get returns success
    calls = {"count": 0}

    def fake_get(url, timeout=5):
        calls["count"] += 1
        return DummyResp(200, sample)

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", fake_get)

    # First fetch should miss cache and fetch
    jwks = manager._fetch_jwks(jwks_url, ttl=10)
    assert jwks == sample
    assert manager.jwks_cache_misses >= 1
    assert manager.jwks_fetch_count >= 1

    # Second fetch should hit cache
    jwks2 = manager._fetch_jwks(jwks_url, ttl=10)
    assert jwks2 == sample
    assert manager.jwks_cache_hits >= 1


def test_fetch_jwks_retries_then_success(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/rotating/jwks.json"

    sample = {"keys": [{"kid": "kx", "kty": "RSA"}]}
    seq = {"calls": 0}

    def flaky_get(url, timeout=5):
        seq["calls"] += 1
        if seq["calls"] == 1:
            return DummyResp(500, {})
        return DummyResp(200, sample)

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", flaky_get)

    jwks = manager._fetch_jwks(jwks_url, ttl=10)
    assert jwks == sample
    # One error recorded then success
    assert manager.jwks_fetch_errors >= 1
    assert manager.jwks_fetch_count >= 2


def test_refresh_forces_fetch(monkeypatch):
    manager = ZeroTrustManager()
    jwks_url = "https://example.com/force/jwks.json"
    sample1 = {"keys": [{"kid": "a", "kty": "RSA"}]}
    sample2 = {"keys": [{"kid": "b", "kty": "RSA"}]}

    # First returns sample1
    def get_first(url, timeout=5):
        return DummyResp(200, sample1)

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", get_first)
    jwks1 = manager._fetch_jwks(jwks_url, ttl=100)
    assert jwks1 == sample1

    # Now change implementation to return sample2
    def get_second(url, timeout=5):
        return DummyResp(200, sample2)

    monkeypatch.setattr("ollama.auth.zero_trust.requests_mod.get", get_second)

    # Normal fetch should return cached sample1
    jwks_cached = manager._fetch_jwks(jwks_url, ttl=100)
    assert jwks_cached == sample1

    # refresh_jwks should force new fetch and return sample2
    jwks_refreshed = manager.refresh_jwks(jwks_url)
    assert jwks_refreshed == sample2
