"""FastAPI and Library Instrumentation for OpenTelemetry.

Provides automated instrumentation for FastAPI, SQLAlchemy, Redis, and HTTP clients
to ensure end-to-end trace propagation across all system components.

Implements Elite Execution Protocol Section: "Automated Tracing & Observability"
"""

import logging
from typing import Any

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

logger = logging.getLogger(__name__)


class OTLPInstrumentor:
    """Handles automated instrumentation of various libraries."""

    @staticmethod
    def instrument_all(app: Any | None = None) -> None:
        """Instrument all supported libraries.

        Args:
            app: FastAPI application instance for FastAPI instrumentation.
        """
        OTLPInstrumentor.instrument_fastapi(app)
        OTLPInstrumentor.instrument_sqlalchemy()
        OTLPInstrumentor.instrument_redis()
        OTLPInstrumentor.instrument_httpx()

    @staticmethod
    def instrument_fastapi(app: Any | None) -> None:
        """Instrument FastAPI application.

        Args:
            app: FastAPI application instance.
        """
        if not app:
            logger.warning("⚠️ Skipping FastAPI instrumentation: No app instance provided")
            return

        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("✅ FastAPI instrumented for tracing")
        except Exception as e:
            logger.error(f"❌ Failed to instrument FastAPI: {e}")

    @staticmethod
    def instrument_sqlalchemy(engine: Any | None = None) -> None:
        """Instrument SQLAlchemy.

        Args:
            engine: Optional SQLAlchemy engine to instrument. If None, instruments globally.
        """
        try:
            if engine:
                SQLAlchemyInstrumentor().instrument(engine=engine)
            else:
                SQLAlchemyInstrumentor().instrument()
            logger.info("✅ SQLAlchemy instrumented for tracing")
        except Exception as e:
            logger.error(f"❌ Failed to instrument SQLAlchemy: {e}")

    @staticmethod
    def instrument_redis() -> None:
        """Instrument Redis client."""
        try:
            RedisInstrumentor().instrument()
            logger.info("✅ Redis instrumented for tracing")
        except Exception as e:
            logger.error(f"❌ Failed to instrument Redis: {e}")

    @staticmethod
    def instrument_httpx() -> None:
        """Instrument HTTPX client for outgoing requests."""
        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("✅ HTTPX instrumented for tracing")
        except Exception as e:
            logger.error(f"❌ Failed to instrument HTTPX: {e}")

    @staticmethod
    def uninstrument_all() -> None:
        """Remove all instrumentations."""
        try:
            FastAPIInstrumentor().uninstrument()
            SQLAlchemyInstrumentor().uninstrument()
            RedisInstrumentor().uninstrument()
            HTTPXClientInstrumentor().uninstrument()
            logger.info("🛑 All libraries uninstrumented")
        except Exception as e:
            logger.error(f"❌ Error during uninstrumentation: {e}")
