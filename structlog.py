"""Minimal fallback shim for `structlog` used during local test runs.

This provides a tiny subset of the structlog API (`get_logger`, `.bind()`, and
logging methods) so tests can run in environments where `structlog` isn't
installed. It's intentionally small and only for test/dev use; production
should install the real `structlog` package.
"""
from __future__ import annotations
import logging
from typing import Any

_root = logging.getLogger("ollama.structlog")
if not _root.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _root.addHandler(handler)
_root.setLevel(logging.INFO)


class SimpleLogger:
    def __init__(self, name: str | None = None) -> None:
        self._name = name or "structlog"
        self._logger = logging.getLogger(self._name)

    def bind(self, **kwargs: Any) -> "SimpleLogger":
        return self

    def info(self, *args: Any, **kwargs: Any) -> None:
        self._logger.info("%s", " ".join(map(str, args)))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self._logger.warning("%s", " ".join(map(str, args)))

    def error(self, *args: Any, **kwargs: Any) -> None:
        self._logger.error("%s", " ".join(map(str, args)))

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self._logger.debug("%s", " ".join(map(str, args)))


def get_logger(name: str | None = None) -> SimpleLogger:
    return SimpleLogger(name)
