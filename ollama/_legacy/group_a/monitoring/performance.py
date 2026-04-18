"""Performance testing and benchmarking utilities.

Provides benchmarking decorators, SLO validation, and performance
metrics collection for load testing and performance analysis.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class PerformanceMetrics:
    """Performance metrics for a request.

    Attributes:
        duration_ms: Request duration in milliseconds
        start_time: Request start time (seconds since epoch)
        end_time: Request end time (seconds since epoch)
        success: Whether request succeeded
        error: Error message if failed
    """

    duration_ms: float
    start_time: float
    end_time: float
    success: bool
    error: str | None = None

    def exceeds_slo(self, slo_ms: float) -> bool:
        """Check if request exceeds SLO threshold.

        Args:
            slo_ms: SLO threshold in milliseconds

        Returns:
            True if duration exceeds SLO
        """
        return self.duration_ms > slo_ms


class SLOValidator:
    """Service Level Objective validator.

    Tracks request performance against SLOs and provides
    aggregated statistics.

    Attributes:
        name: Endpoint name
        slo_ms: SLO threshold in milliseconds
        metrics: List of recorded metrics
    """

    def __init__(self, name: str, slo_ms: float) -> None:
        """Initialize SLO validator.

        Args:
            name: Endpoint name
            slo_ms: SLO threshold in milliseconds
        """
        self.name = name
        self.slo_ms = slo_ms
        self.metrics: list[PerformanceMetrics] = []

    def add_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric.

        Args:
            metric: Performance metrics instance
        """
        self.metrics.append(metric)

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregated performance statistics.

        Returns:
            Dictionary with p50, p95, p99, success_rate, slo_compliance
        """
        if not self.metrics:
            return {}

        durations = [m.duration_ms for m in self.metrics]
        durations.sort()
        count = len(durations)

        successful = sum(1 for m in self.metrics if m.success)
        slo_compliant = sum(1 for m in self.metrics if not m.exceeds_slo(self.slo_ms))

        return {
            "endpoint": self.name,
            "slo_ms": self.slo_ms,
            "total_requests": count,
            "successful_requests": successful,
            "slo_compliant": slo_compliant,
            "success_rate": (successful / count) * 100 if count > 0 else 0,
            "slo_compliance": (slo_compliant / count) * 100 if count > 0 else 0,
            "p50_ms": durations[int(count * 0.50) - 1] if count > 0 else 0,
            "p95_ms": durations[int(count * 0.95) - 1] if count > 0 else 0,
            "p99_ms": durations[int(count * 0.99) - 1] if count > 1 else 0,
            "mean_ms": sum(durations) / count if count > 0 else 0,
            "min_ms": min(durations) if count > 0 else 0,
            "max_ms": max(durations) if count > 0 else 0,
        }

    def validate_slo(self) -> bool:
        """Validate if SLO is met.

        Returns:
            True if 95%+ of requests meet SLO
        """
        if not self.metrics:
            return True
        stats = self.get_statistics()
        slo_compliance = stats.get("slo_compliance", 0)
        return bool(isinstance(slo_compliance, (int, float)) and slo_compliance >= 95)


def benchmark_async(
    slo_ms: float | None = None,
) -> Callable[..., Any]:
    """Decorator for benchmarking async functions.

    Args:
        slo_ms: SLO threshold in milliseconds

    Returns:
        Decorated function

    Example:
        @benchmark_async(slo_ms=500)
        async def my_endpoint() -> dict:
            return {"status": "ok"}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            success = True
            error = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end = time.time()
                duration_ms = (end - start) * 1000

                # Store metric in app state if available
                if hasattr(func, "__self__"):
                    # Method - check for app context
                    pass

                # Log performance
                level = "warning" if (slo_ms and duration_ms > slo_ms) else "debug"
                import logging

                logger = logging.getLogger(__name__)
                logger.log(
                    getattr(logging, level.upper()),
                    f"{func.__name__} took {duration_ms:.1f}ms",
                    extra={
                        "function": func.__name__,
                        "duration_ms": duration_ms,
                        "slo_ms": slo_ms,
                        "success": success,
                        "error": error,
                    },
                )

        return wrapper

    return decorator


def benchmark(
    slo_ms: float | None = None,
) -> Callable[..., Any]:
    """Decorator for benchmarking sync functions.

    Args:
        slo_ms: SLO threshold in milliseconds

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            success = True
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end = time.time()
                duration_ms = (end - start) * 1000

                level = "warning" if (slo_ms and duration_ms > slo_ms) else "debug"
                import logging

                logger = logging.getLogger(__name__)
                logger.log(
                    getattr(logging, level.upper()),
                    f"{func.__name__} took {duration_ms:.1f}ms",
                    extra={
                        "function": func.__name__,
                        "duration_ms": duration_ms,
                        "slo_ms": slo_ms,
                        "success": success,
                        "error": error,
                    },
                )

        return wrapper

    return decorator
