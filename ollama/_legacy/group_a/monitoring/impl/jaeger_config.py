"""
Jaeger Tracing Configuration and Integration
Distributes tracing context across microservices
"""

import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:  # pragma: no cover - optional dependency
    FastAPIInstrumentor: Any = None  # type: ignore[no-redef]

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except ImportError:  # pragma: no cover - optional dependency
    HTTPXClientInstrumentor: Any = None  # type: ignore[no-redef]

try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor
except ImportError:  # pragma: no cover - optional dependency
    RedisInstrumentor: Any = None  # type: ignore[no-redef]

try:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
except ImportError:  # pragma: no cover - optional dependency
    SQLAlchemyInstrumentor: Any = None  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class JaegerConfig:
    """Jaeger tracing configuration"""

    def __init__(
        self,
        service_name: str = "ollama-api",
        jaeger_host: str = "jaeger",
        jaeger_port: int = 6831,
        jaeger_udp_port: int = 6831,
        trace_sample_rate: float = 0.1,
    ):
        """
        Initialize Jaeger configuration

        Args:
            service_name: Service name for traces
            jaeger_host: Jaeger agent host
            jaeger_port: Jaeger agent port
            jaeger_udp_port: Jaeger UDP port (for Thrift protocol)
            trace_sample_rate: Fraction of traces to sample (0.0-1.0)
        """
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.jaeger_udp_port = jaeger_udp_port
        self.trace_sample_rate = trace_sample_rate
        self._tracer_provider: TracerProvider | None = None

    def initialize_tracer(self) -> TracerProvider:
        """
        Initialize OTLP tracer provider (migrated from Jaeger)

        Returns:
            Configured TracerProvider
        """
        try:
            # Create OTLP exporter pointing to the collector
            # Default OTLP endpoint is usually http://otel-collector:4317
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://{self.jaeger_host}:4317")

            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True,
            )

            # Create resource
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                }
            )

            # Create tracer provider with sampler
            sampler = ParentBased(root=TraceIdRatioBased(self.sample_rate))
            trace_provider = TracerProvider(resource=resource, sampler=sampler)

            # Add OTLP exporter
            trace_provider.add_span_processor(BatchSpanProcessor(exporter))

            # Set global tracer provider
            trace.set_tracer_provider(trace_provider)

            self._tracer_provider = trace_provider
            logger.info(f"✅ Tracer initialized (OTLP -> {endpoint})")

            return trace_provider

        except Exception as e:
            logger.error(f"❌ Failed to initialize tracer: {e}")
            raise

    def instrument_fastapi(self, app: Any) -> None:
        """
        Instrument FastAPI app for tracing

        Args:
            app: FastAPI application instance
        """
        if FastAPIInstrumentor is None:
            logger.warning("⚠️  FastAPI instrumentation not available")
            return
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("✅ FastAPI instrumented for tracing")
        except Exception as e:
            logger.error(f"⚠️  Failed to instrument FastAPI: {e}")

    def instrument_sqlalchemy(self, engine: Any) -> None:
        """
        Instrument SQLAlchemy for tracing

        Args:
            engine: SQLAlchemy engine
        """
        if SQLAlchemyInstrumentor is None:
            logger.warning("⚠️  SQLAlchemy instrumentation not available")
            return
        try:
            SQLAlchemyInstrumentor().instrument(engine=engine, service=self.service_name)
            logger.info("✅ SQLAlchemy instrumented for tracing")
        except Exception as e:
            logger.error(f"⚠️  Failed to instrument SQLAlchemy: {e}")

    def instrument_httpx(self) -> None:
        """
        Instrument httpx for tracing
        """
        if HTTPXClientInstrumentor is None:
            logger.warning("⚠️  httpx instrumentation not available")
            return
        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("✅ httpx instrumented for tracing")
        except Exception as e:
            logger.error(f"⚠️  Failed to instrument httpx: {e}")

    def instrument_redis(self) -> None:
        """
        Instrument Redis for tracing
        """
        if RedisInstrumentor is None:
            logger.warning("⚠️  Redis instrumentation not available")
            return
        try:
            RedisInstrumentor().instrument()
            logger.info("✅ Redis instrumented for tracing")
        except Exception as e:
            logger.error(f"⚠️  Failed to instrument Redis: {e}")

    def get_tracer(self, name: str) -> trace.Tracer:
        """
        Get tracer instance

        Args:
            name: Tracer name (usually __name__)

        Returns:
            Tracer instance
        """
        if self._tracer_provider is None:
            self.initialize_tracer()

        return trace.get_tracer(name)


# Global Jaeger config instance
_jaeger_config: JaegerConfig | None = None


def init_jaeger(
    service_name: str = "ollama-api",
    jaeger_host: str = "jaeger",
    jaeger_port: int = 6831,
    trace_sample_rate: float = 0.1,
) -> JaegerConfig:
    """
    Initialize Jaeger configuration

    Args:
        service_name: Service name
        jaeger_host: Jaeger host
        jaeger_port: Jaeger port
        trace_sample_rate: Sample rate

    Returns:
        JaegerConfig instance
    """
    global _jaeger_config
    _jaeger_config = JaegerConfig(
        service_name=service_name,
        jaeger_host=jaeger_host,
        jaeger_port=jaeger_port,
        trace_sample_rate=trace_sample_rate,
    )
    return _jaeger_config


def get_jaeger_config() -> JaegerConfig | None:
    """
    Get global Jaeger config instance

    Returns:
        JaegerConfig or None if not initialized
    """
    return _jaeger_config
