"""Lightweight profiling utilities for Ollama.

Provides a simple `ProfilerManager` that uses `cProfile` and `tracemalloc`
to collect CPU and memory profiles on-demand. Designed for local/dev use and
to support generating profiling dumps that can be converted to flamegraphs.

This is a minimal, safe implementation that avoids adding heavy runtime
dependencies while still enabling programmatic profiling from tests or
administrative endpoints.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import threading
import tracemalloc

logger_name = "ollama.monitoring.profiling"


class ProfilerManager:
    """Manage a CPU profiler and optional memory snapshotting.

    Usage:
        mgr = ProfilerManager()
        mgr.start()
        # run workload
        stats_text = mgr.stop_and_get_stats()
    """

    def __init__(self) -> None:
        self._profiler: cProfile.Profile | None = None
        self._lock = threading.Lock()
        self._tracemalloc_enabled = False

    def start(self, enable_tracemalloc: bool = True) -> None:
        """Start CPU profiler and optionally tracemalloc."""
        with self._lock:
            if self._profiler is not None:
                return
            self._profiler = cProfile.Profile()
            self._profiler.enable()
            if enable_tracemalloc and not tracemalloc.is_tracing():
                tracemalloc.start()
                self._tracemalloc_enabled = True

    def stop_and_get_stats(self, sort_by: str = "cumulative", top_n: int = 50) -> str:
        """Stop the profiler and return a human-readable stats snapshot.

        Returns a string containing the top functions by `sort_by`.
        """
        with self._lock:
            if self._profiler is None:
                return "No profiler running"

            self._profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self._profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(top_n)
            self._profiler = None

            # Memory snapshot
            if self._tracemalloc_enabled:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")
                s.write("\nTop memory usage (tracemalloc):\n")
                for stat in top_stats[:10]:
                    s.write(str(stat) + "\n")
                # keep tracemalloc running for subsequent snapshots

            return s.getvalue()

    def is_running(self) -> bool:
        with self._lock:
            return self._profiler is not None


__all__ = ["ProfilerManager"]
