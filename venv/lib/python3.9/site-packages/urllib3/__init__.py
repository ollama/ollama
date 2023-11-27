"""
Python HTTP library with thread-safe connection pooling, file post support, user friendly, and more
"""

from __future__ import annotations

# Set default logging handler to avoid "No handler found" warnings.
import logging
import typing
import warnings
from logging import NullHandler

from . import exceptions
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from ._version import __version__
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, connection_from_url
from .filepost import _TYPE_FIELDS, encode_multipart_formdata
from .poolmanager import PoolManager, ProxyManager, proxy_from_url
from .response import BaseHTTPResponse, HTTPResponse
from .util.request import make_headers
from .util.retry import Retry
from .util.timeout import Timeout

# Ensure that Python is compiled with OpenSSL 1.1.1+
# If the 'ssl' module isn't available at all that's
# fine, we only care if the module is available.
try:
    import ssl
except ImportError:
    pass
else:
    if not ssl.OPENSSL_VERSION.startswith("OpenSSL "):  # Defensive:
        warnings.warn(
            "urllib3 v2 only supports OpenSSL 1.1.1+, currently "
            f"the 'ssl' module is compiled with {ssl.OPENSSL_VERSION!r}. "
            "See: https://github.com/urllib3/urllib3/issues/3020",
            exceptions.NotOpenSSLWarning,
        )
    elif ssl.OPENSSL_VERSION_INFO < (1, 1, 1):  # Defensive:
        raise ImportError(
            "urllib3 v2 only supports OpenSSL 1.1.1+, currently "
            f"the 'ssl' module is compiled with {ssl.OPENSSL_VERSION!r}. "
            "See: https://github.com/urllib3/urllib3/issues/2168"
        )

__author__ = "Andrey Petrov (andrey.petrov@shazow.net)"
__license__ = "MIT"
__version__ = __version__

__all__ = (
    "HTTPConnectionPool",
    "HTTPHeaderDict",
    "HTTPSConnectionPool",
    "PoolManager",
    "ProxyManager",
    "HTTPResponse",
    "Retry",
    "Timeout",
    "add_stderr_logger",
    "connection_from_url",
    "disable_warnings",
    "encode_multipart_formdata",
    "make_headers",
    "proxy_from_url",
    "request",
    "BaseHTTPResponse",
)

logging.getLogger(__name__).addHandler(NullHandler())


def add_stderr_logger(
    level: int = logging.DEBUG,
) -> logging.StreamHandler[typing.TextIO]:
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging.

    Returns the handler after adding it.
    """
    # This method needs to be in this __init__.py to get the __name__ correct
    # even if urllib3 is vendored within another package.
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


# ... Clean up.
del NullHandler


# All warning filters *must* be appended unless you're really certain that they
# shouldn't be: otherwise, it's very hard for users to use most Python
# mechanisms to silence them.
# SecurityWarning's always go off by default.
warnings.simplefilter("always", exceptions.SecurityWarning, append=True)
# InsecurePlatformWarning's don't vary between requests, so we keep it default.
warnings.simplefilter("default", exceptions.InsecurePlatformWarning, append=True)


def disable_warnings(category: type[Warning] = exceptions.HTTPWarning) -> None:
    """
    Helper for quickly disabling all urllib3 warnings.
    """
    warnings.simplefilter("ignore", category)


_DEFAULT_POOL = PoolManager()


def request(
    method: str,
    url: str,
    *,
    body: _TYPE_BODY | None = None,
    fields: _TYPE_FIELDS | None = None,
    headers: typing.Mapping[str, str] | None = None,
    preload_content: bool | None = True,
    decode_content: bool | None = True,
    redirect: bool | None = True,
    retries: Retry | bool | int | None = None,
    timeout: Timeout | float | int | None = 3,
    json: typing.Any | None = None,
) -> BaseHTTPResponse:
    """
    A convenience, top-level request method. It uses a module-global ``PoolManager`` instance.
    Therefore, its side effects could be shared across dependencies relying on it.
    To avoid side effects create a new ``PoolManager`` instance and use it instead.
    The method does not accept low-level ``**urlopen_kw`` keyword arguments.
    """

    return _DEFAULT_POOL.request(
        method,
        url,
        body=body,
        fields=fields,
        headers=headers,
        preload_content=preload_content,
        decode_content=decode_content,
        redirect=redirect,
        retries=retries,
        timeout=timeout,
        json=json,
    )
