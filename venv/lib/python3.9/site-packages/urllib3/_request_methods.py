from __future__ import annotations

import json as _json
import typing
from urllib.parse import urlencode

from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .filepost import _TYPE_FIELDS, encode_multipart_formdata
from .response import BaseHTTPResponse

__all__ = ["RequestMethods"]

_TYPE_ENCODE_URL_FIELDS = typing.Union[
    typing.Sequence[typing.Tuple[str, typing.Union[str, bytes]]],
    typing.Mapping[str, typing.Union[str, bytes]],
]


class RequestMethods:
    """
    Convenience mixin for classes who implement a :meth:`urlopen` method, such
    as :class:`urllib3.HTTPConnectionPool` and
    :class:`urllib3.PoolManager`.

    Provides behavior for making common types of HTTP request methods and
    decides which type of request field encoding to use.

    Specifically,

    :meth:`.request_encode_url` is for sending requests whose fields are
    encoded in the URL (such as GET, HEAD, DELETE).

    :meth:`.request_encode_body` is for sending requests whose fields are
    encoded in the *body* of the request using multipart or www-form-urlencoded
    (such as for POST, PUT, PATCH).

    :meth:`.request` is for making any kind of request, it will look up the
    appropriate encoding format and use one of the above two methods to make
    the request.

    Initializer parameters:

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.
    """

    _encode_url_methods = {"DELETE", "GET", "HEAD", "OPTIONS"}

    def __init__(self, headers: typing.Mapping[str, str] | None = None) -> None:
        self.headers = headers or {}

    def urlopen(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        encode_multipart: bool = True,
        multipart_boundary: str | None = None,
        **kw: typing.Any,
    ) -> BaseHTTPResponse:  # Abstract
        raise NotImplementedError(
            "Classes extending RequestMethods must implement "
            "their own ``urlopen`` method."
        )

    def request(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        fields: _TYPE_FIELDS | None = None,
        headers: typing.Mapping[str, str] | None = None,
        json: typing.Any | None = None,
        **urlopen_kw: typing.Any,
    ) -> BaseHTTPResponse:
        """
        Make a request using :meth:`urlopen` with the appropriate encoding of
        ``fields`` based on the ``method`` used.

        This is a convenience method that requires the least amount of manual
        effort. It can be used in most situations, while still having the
        option to drop down to more specific methods when necessary, such as
        :meth:`request_encode_url`, :meth:`request_encode_body`,
        or even the lowest level :meth:`urlopen`.
        """
        method = method.upper()

        if json is not None and body is not None:
            raise TypeError(
                "request got values for both 'body' and 'json' parameters which are mutually exclusive"
            )

        if json is not None:
            if headers is None:
                headers = self.headers.copy()  # type: ignore
            if not ("content-type" in map(str.lower, headers.keys())):
                headers["Content-Type"] = "application/json"  # type: ignore

            body = _json.dumps(json, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )

        if body is not None:
            urlopen_kw["body"] = body

        if method in self._encode_url_methods:
            return self.request_encode_url(
                method,
                url,
                fields=fields,  # type: ignore[arg-type]
                headers=headers,
                **urlopen_kw,
            )
        else:
            return self.request_encode_body(
                method, url, fields=fields, headers=headers, **urlopen_kw
            )

    def request_encode_url(
        self,
        method: str,
        url: str,
        fields: _TYPE_ENCODE_URL_FIELDS | None = None,
        headers: typing.Mapping[str, str] | None = None,
        **urlopen_kw: str,
    ) -> BaseHTTPResponse:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the url. This is useful for request methods like GET, HEAD, DELETE, etc.
        """
        if headers is None:
            headers = self.headers

        extra_kw: dict[str, typing.Any] = {"headers": headers}
        extra_kw.update(urlopen_kw)

        if fields:
            url += "?" + urlencode(fields)

        return self.urlopen(method, url, **extra_kw)

    def request_encode_body(
        self,
        method: str,
        url: str,
        fields: _TYPE_FIELDS | None = None,
        headers: typing.Mapping[str, str] | None = None,
        encode_multipart: bool = True,
        multipart_boundary: str | None = None,
        **urlopen_kw: str,
    ) -> BaseHTTPResponse:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the body. This is useful for request methods like POST, PUT, PATCH, etc.

        When ``encode_multipart=True`` (default), then
        :func:`urllib3.encode_multipart_formdata` is used to encode
        the payload with the appropriate content type. Otherwise
        :func:`urllib.parse.urlencode` is used with the
        'application/x-www-form-urlencoded' content type.

        Multipart encoding must be used when posting files, and it's reasonably
        safe to use it in other times too. However, it may break request
        signing, such as with OAuth.

        Supports an optional ``fields`` parameter of key/value strings AND
        key/filetuple. A filetuple is a (filename, data, MIME type) tuple where
        the MIME type is optional. For example::

            fields = {
                'foo': 'bar',
                'fakefile': ('foofile.txt', 'contents of foofile'),
                'realfile': ('barfile.txt', open('realfile').read()),
                'typedfile': ('bazfile.bin', open('bazfile').read(),
                              'image/jpeg'),
                'nonamefile': 'contents of nonamefile field',
            }

        When uploading a file, providing a filename (the first parameter of the
        tuple) is optional but recommended to best mimic behavior of browsers.

        Note that if ``headers`` are supplied, the 'Content-Type' header will
        be overwritten because it depends on the dynamic random boundary string
        which is used to compose the body of the request. The random boundary
        string can be explicitly set with the ``multipart_boundary`` parameter.
        """
        if headers is None:
            headers = self.headers

        extra_kw: dict[str, typing.Any] = {"headers": HTTPHeaderDict(headers)}
        body: bytes | str

        if fields:
            if "body" in urlopen_kw:
                raise TypeError(
                    "request got values for both 'fields' and 'body', can only specify one."
                )

            if encode_multipart:
                body, content_type = encode_multipart_formdata(
                    fields, boundary=multipart_boundary
                )
            else:
                body, content_type = (
                    urlencode(fields),  # type: ignore[arg-type]
                    "application/x-www-form-urlencoded",
                )

            extra_kw["body"] = body
            extra_kw["headers"].setdefault("Content-Type", content_type)

        extra_kw.update(urlopen_kw)

        return self.urlopen(method, url, **extra_kw)
