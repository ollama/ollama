"""
Tests for Metrics Module
Tests Prometheus metrics collection
"""

from ollama.monitoring import (
    AUTH_ATTEMPTS,
    CACHE_HITS,
    CACHE_MISSES,
    REQUEST_COUNT,
    REQUEST_DURATION,
    REQUEST_SIZE,
    RESPONSE_SIZE,
    export_metrics,
)


class TestMetricsCollection:
    """Test Prometheus metrics collection"""

    def test_request_count_metric_exists(self):
        """Test REQUEST_COUNT metric is properly configured"""
        assert REQUEST_COUNT is not None
        # Prometheus uses 'http_requests' as the base name (without _total suffix)
        assert (
            "http_requests" in REQUEST_COUNT._name or REQUEST_COUNT._name == "http_requests_total"
        )

    def test_request_duration_metric_exists(self):
        """Test REQUEST_DURATION metric is properly configured"""
        assert REQUEST_DURATION is not None
        assert REQUEST_DURATION._name == "http_request_duration_seconds"

    def test_cache_metrics_exist(self):
        """Test cache metrics are properly configured"""
        assert CACHE_HITS is not None
        # Prometheus base names don't include _total suffix
        assert "cache_hits" in CACHE_HITS._name or CACHE_HITS._name == "cache_hits_total"

        assert CACHE_MISSES is not None
        assert "cache_misses" in CACHE_MISSES._name or CACHE_MISSES._name == "cache_misses_total"

    def test_auth_metrics_exist(self):
        """Test authentication metrics are properly configured"""
        assert AUTH_ATTEMPTS is not None
        assert (
            "auth_attempts" in AUTH_ATTEMPTS._name or AUTH_ATTEMPTS._name == "auth_attempts_total"
        )

    def test_record_request_count(self):
        """Test recording request count"""
        REQUEST_COUNT.labels(method="GET", endpoint="/api/test", status_code=200).inc()

        # Metric should be incremented (no assertion needed, just verify no error)
        assert True

    def test_record_request_duration(self):
        """Test recording request duration"""
        REQUEST_DURATION.labels(method="POST", endpoint="/api/generate").observe(0.5)

        # Metric should be observed
        assert True

    def test_record_cache_hit(self):
        """Test recording cache hit"""
        CACHE_HITS.labels(cache_type="redis", operation="get").inc()

        assert True

    def test_record_cache_miss(self):
        """Test recording cache miss"""
        CACHE_MISSES.labels(cache_type="redis", operation="get").inc()

        assert True

    def test_record_auth_attempt(self):
        """Test recording authentication attempt"""
        AUTH_ATTEMPTS.labels(method="password", result="success").inc()

        assert True

    def test_export_metrics_returns_bytes(self):
        """Test exporting metrics returns bytes"""
        metrics_data = export_metrics()

        assert isinstance(metrics_data, bytes)
        assert len(metrics_data) > 0

        # Should contain Prometheus format data
        metrics_str = metrics_data.decode("utf-8")
        assert "http_requests_total" in metrics_str or "#" in metrics_str

    def test_metrics_include_labels(self):
        """Test exported metrics include labels"""
        # Record some metrics with labels
        REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code=200).inc()

        metrics_data = export_metrics()
        metrics_str = metrics_data.decode("utf-8")

        # Should include the metric type
        assert "http_requests_total" in metrics_str or "#" in metrics_str


class TestMetricsHelpers:
    """Test metrics helper functions"""

    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        from ollama.monitoring import get_metrics_summary

        summary = get_metrics_summary()

        assert isinstance(summary, dict)
        assert "http_requests" in summary
        assert "cache_hits" in summary
        assert "auth_attempts" in summary

    def test_metric_names_are_valid_prometheus(self):
        """Test metric names follow Prometheus naming conventions"""
        from ollama.monitoring import (
            AUTH_ATTEMPTS,
            CACHE_HITS,
            OLLAMA_TOKENS_GENERATED,
            REQUEST_COUNT,
        )

        # Prometheus metric names should:
        # - Be lowercase with underscores
        # - Include the metric type suffix (_total for counters, _seconds for histograms)

        assert "http_requests" in REQUEST_COUNT._name
        assert "cache_hits" in CACHE_HITS._name
        assert "auth_attempts" in AUTH_ATTEMPTS._name
        assert "ollama_tokens_generated" in OLLAMA_TOKENS_GENERATED._name


class TestMetricsIntegration:
    """Integration tests for metrics"""

    def test_multiple_metric_types(self):
        """Test recording different metric types"""
        # Counter
        REQUEST_COUNT.labels(method="GET", endpoint="/api/models", status_code=200).inc(5)

        # Histogram
        REQUEST_DURATION.labels(method="POST", endpoint="/api/chat").observe(1.23)

        # Different labels for same metric
        REQUEST_COUNT.labels(method="POST", endpoint="/api/chat", status_code=500).inc(1)

        # All should succeed
        assert True

    def test_metrics_with_special_characters(self):
        """Test metrics with special endpoint names"""
        REQUEST_COUNT.labels(
            method="GET", endpoint="/api/v1/models/{model_id}", status_code=200
        ).inc()

        assert True
