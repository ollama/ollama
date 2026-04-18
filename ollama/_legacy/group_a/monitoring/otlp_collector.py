"""OpenTelemetry OTLP Collector Integration.

Configures and manages the OTLP exporter for distributed tracing,
routing traces to the centralized OTLP collector which then forwards
to Jaeger and Grafana Tempo.

Implements Elite Execution Protocol Section: "Observability & Tracing"
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

# Setup logging
logger = logging.getLogger(__name__)


class OTLPCollectorManager:
    """Manages the lifecycle of the OpenTelemetry OTLP collector exporter."""

    def __init__(
        self,
        service_name: str = "ollama-api",
        endpoint: str | None = None,
        insecure: bool = True,
        sample_rate: float = 1.0,
    ) -> None:
        """Initialize OTLP Collector Manager.

        Args:
            service_name: Name of the service for tracing.
            endpoint: OTLP collector endpoint (e.g., "http://otel-collector:4317").
            insecure: Whether to use insecure connection.
            sample_rate: Trace sampling rate (0.0 to 1.0).
        """
        self.service_name = service_name
        self.endpoint = endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
        )
        self.insecure = insecure
        self.sample_rate = sample_rate
        self._tracer_provider: TracerProvider | None = None

    def initialize(self) -> TracerProvider:
        """Initialize the OTLP exporter and global tracer provider.

        Returns:
            Configured TracerProvider instance.
        """
        try:
            # Define resource attributes (GCP Landing Zone compliance)
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "deployment.environment": os.getenv("ENVIRONMENT", "development"),
                    "application": "ollama",
                    "team": "elevatediq-engineering",
                }
            )

            # Create OTLP Span Exporter (gRPC)
            exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                insecure=self.insecure,
            )

            # Setup Sampling (Probabilistic)
            sampler = ParentBased(root=TraceIdRatioBased(self.sample_rate))

            # Create Tracer Provider
            provider = TracerProvider(resource=resource, sampler=sampler)

            # Add Batch Span Processor for high-performance ingestion
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )
            provider.add_span_processor(processor)

            # Set global tracer provider
            trace.set_tracer_provider(provider)
            self._tracer_provider = provider

            logger.info(
                "✅ OTLP Collector initialized",
                extra={
                    "endpoint": self.endpoint,
                    "service_name": self.service_name,
                    "sample_rate": self.sample_rate,
                },
            )
            return provider

        except Exception as e:
            logger.error(f"❌ Failed to initialize OTLP Collector: {e}", exc_info=True)
            raise

    def get_tracer(self, name: str) -> trace.Tracer:
        """Get a tracer instance for the given name.

        Args:
            name: Name of the tracer (usually __name__).

        Returns:
            OpenTelemetry Tracer instance.
        """
        if not self._tracer_provider:
            self.initialize()
        return trace.get_tracer(name)

    def shutdown(self) -> None:
        """Shut down the tracer provider and flush spans."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
            logger.info("🛑 OTLP Tracer Provider shut down")


# Singleton instance
_manager: OTLPCollectorManager | None = None


def setup_otlp_tracing(
    service_name: str = "ollama-api",
    endpoint: str | None = None,
    sample_rate: float = 1.0,
) -> OTLPCollectorManager:
    """Convenience function to set up OTLP tracing.

    Args:
        service_name: Name of the service.
        endpoint: Collector endpoint.
        sample_rate: Trace sampling rate.

    Returns:
        OTLPCollectorManager instance.
    """
    global _manager
    if _manager is None:
        _manager = OTLPCollectorManager(
            service_name=service_name,
            endpoint=endpoint,
            sample_rate=sample_rate,
        )
        _manager.initialize()
    return _manager


def get_otlp_manager() -> OTLPCollectorManager | None:
    """Get the global OTLP manager instance."""
    return _manager
