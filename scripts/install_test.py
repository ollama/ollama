#!/usr/bin/env python3

import contextlib
import http.server
import os
import socket
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = ROOT / "scripts" / "install.sh"
SHELL = os.environ.get("SHELL_UNDER_TEST", "sh")
PAYLOAD = bytes(range(256)) * 1024
FIRST_CHUNK = len(PAYLOAD) // 4


class QuietHTTPServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass


class InterruptedDownloadHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "resume"
    requests = []
    payload_requests = 0

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        range_header = self.headers.get("Range")
        type(self).requests.append((self.path, range_header))

        if self.mode == "redirect-resume" and self.path == "/archive":
            self.send_response(http.HTTPStatus.FOUND)
            self.send_header("Content-Length", "0")
            self.send_header("Location", "/payload")
            self.end_headers()
            return

        type(self).payload_requests += 1

        if self.mode == "ignore-range" and range_header:
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            return

        start = 0
        if range_header:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            self.send_response(http.HTTPStatus.PARTIAL_CONTENT)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
            )
        else:
            self.send_response(http.HTTPStatus.OK)

        remaining = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(remaining)))
        self.end_headers()

        should_interrupt = self.mode == "always-fail" or self.payload_requests == 1
        if should_interrupt:
            chunk_size = min(FIRST_CHUNK, max(1, len(remaining) // 2))
            self.wfile.write(remaining[:chunk_size])
            self.wfile.flush()
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return

        self.wfile.write(remaining)


@contextlib.contextmanager
def download_server(mode):
    handler = type(
        "Handler",
        (InterruptedDownloadHandler,),
        {"mode": mode, "requests": [], "payload_requests": 0},
    )
    server = QuietHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def download_wrapper():
    source = INSTALL_SCRIPT.read_text()
    prefix, entrypoint = source.rsplit("\nmain", 1)
    if entrypoint.strip():
        raise AssertionError("scripts/install.sh must end with a main call")
    return (
        prefix
        + '\nset -eu\nwarning() { echo ">>> $*" >&2; }\ndownload "$1" "$2"\n'
    )


class InstallerDownloadTest(unittest.TestCase):
    def run_download(self, mode):
        with tempfile.TemporaryDirectory() as temp_dir, download_server(mode) as (
            server,
            handler,
        ):
            temp_path = Path(temp_dir)
            wrapper = temp_path / "download-test.sh"
            output = temp_path / "archive"
            wrapper.write_text(download_wrapper())
            result = subprocess.run(
                [
                    SHELL,
                    str(wrapper),
                    f"http://127.0.0.1:{server.server_port}/archive",
                    str(output),
                ],
                text=True,
                capture_output=True,
                timeout=20,
                env={**os.environ, "LC_ALL": "C"},
            )
            data = output.read_bytes() if output.exists() else b""
            return result, data, handler.requests

    def test_resumes_interrupted_download(self):
        result, data, requests = self.run_download("resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data, PAYLOAD)
        self.assertEqual(
            requests,
            [("/archive", None), ("/archive", f"bytes={FIRST_CHUNK}-")],
        )

    def test_resumes_interrupted_download_after_redirect(self):
        result, data, requests = self.run_download("redirect-resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data, PAYLOAD)
        self.assertEqual(
            requests,
            [
                ("/archive", None),
                ("/payload", None),
                ("/archive", f"bytes={FIRST_CHUNK}-"),
                ("/payload", f"bytes={FIRST_CHUNK}-"),
            ],
        )

    def test_restarts_when_server_ignores_range(self):
        result, data, requests = self.run_download("ignore-range")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data, PAYLOAD)
        self.assertEqual(requests[0], ("/archive", None))
        self.assertEqual(requests[1], ("/archive", f"bytes={FIRST_CHUNK}-"))
        self.assertEqual(requests[2], ("/archive", None))

    def test_fails_after_bounded_attempts(self):
        result, data, requests = self.run_download("always-fail")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(data, PAYLOAD)
        self.assertEqual(len(requests), 3)


if __name__ == "__main__":
    unittest.main()
