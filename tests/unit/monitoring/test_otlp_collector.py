from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider

from ollama.monitoring.otlp_collector import OTLPCollectorManager, setup_otlp_tracing


def test_otlp_collector_initialization():
    """Test that OTLPCollectorManager initializes correctly."""
    manager = OTLPCollectorManager(
        service_name="test-service", endpoint="http://localhost:4317", sample_rate=1.0
    )
    with (
        patch(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
        ) as mock_exporter,
        patch("opentelemetry.trace.set_tracer_provider") as mock_set_provider,
        patch("opentelemetry.sdk.trace.TracerProvider.add_span_processor") as mock_add_processor,
    ):

        provider = manager.initialize()

        assert isinstance(provider, TracerProvider)
        assert manager.service_name == "test-service"
        assert manager.endpoint == "http://localhost:4317"
        mock_set_provider.assert_called_once_with(provider)
        mock_add_processor.assert_called_once()


def test_setup_otlp_tracing_singleton():
    """Test that setup_otlp_tracing returns a singleton instance."""
    # Reset singleton first
    import ollama.monitoring.otlp_collector as otlp_module

    otlp_module._manager = None

    with patch("ollama.monitoring.otlp_collector.OTLPCollectorManager.initialize"):
        m1 = setup_otlp_tracing(service_name="s1")
        m2 = setup_otlp_tracing(service_name="s2")

        assert m1 is m2
        assert m1.service_name == "s1"  # Initial service name preserved


def test_get_tracer():
    """Test getting a tracer from the manager."""
    manager = OTLPCollectorManager(service_name="test-service")

    with patch("ollama.monitoring.otlp_collector.OTLPCollectorManager.initialize") as mock_init:
        # Use a real TracerProvider for the mock
        manager._tracer_provider = TracerProvider()
        tracer = manager.get_tracer("test-tracer")

        assert tracer is not None
        mock_init.assert_not_called()  # Already initialized
