import sys

try:
    # Our match_hostname function is the same as 3.10's, so we only want to
    # import the match_hostname function if it's at least that good.
    # We also fallback on Python 3.10+ because our code doesn't emit
    # deprecation warnings and is the same as Python 3.10 otherwise.
    if sys.version_info < (3, 5) or sys.version_info >= (3, 10):
        raise ImportError("Fallback to vendored code")

    from ssl import CertificateError, match_hostname
except ImportError:
    try:
        # Backport of the function from a pypi module
        from backports.ssl_match_hostname import (  # type: ignore
            CertificateError,
            match_hostname,
        )
    except ImportError:
        # Our vendored copy
        from ._implementation import CertificateError, match_hostname  # type: ignore

# Not needed, but documenting what we provide.
__all__ = ("CertificateError", "match_hostname")
