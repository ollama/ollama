#!/usr/bin/env python3
"""Serve a loopback Ollama app update endpoint for manual upgrade testing.

The server only models the public update surface: an update JSON response with
an artifact URL and optional version, plus files available under /download/.
It does not model ollama.com rollout, account, or policy logic.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import mimetypes
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse


APP_ARTIFACTS = {
    "windows": "OllamaSetup.exe",
    "darwin": "Ollama-darwin.zip",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def file_etag(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return '"' + hashlib.sha256(payload).hexdigest()[:32] + '"'


def content_etag(data: bytes) -> str:
    return '"' + hashlib.sha256(data).hexdigest()[:32] + '"'


def sidecar_etag(path: Path) -> Optional[str]:
    sidecar = path.with_name(path.name + ".etag")
    if not sidecar.exists():
        return None
    etag = sidecar.read_text(encoding="utf-8").strip()
    return etag or None


class ServedFile:
    def __init__(self, path: Path, etag: Optional[str], content: Optional[bytes] = None):
        self.path = path
        self.etag = etag
        self.content = content


class UpdateHandler(http.server.BaseHTTPRequestHandler):
    server_version = "OllamaUpdateTest/1.0"

    def do_HEAD(self) -> None:
        self.handle_request(send_body=False)

    def do_GET(self) -> None:
        self.handle_request(send_body=True)

    def handle_request(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/api/update", "/update.json"}:
            self.serve_update_json(parsed.query, send_body)
            return

        if path.startswith("/download/"):
            name = unquote(path.removeprefix("/download/"))
            if "/" not in name and "\\" not in name and name in self.server.files:
                self.serve_file(self.server.files[name], send_body)
                return

        self.send_error(404, "not found")

    def serve_update_json(self, query: str, send_body: bool) -> None:
        artifact_name = self.server.update_artifact_name(query)
        if not artifact_name:
            self.send_response(204)
            self.end_headers()
            return

        body = {
            "url": f"{self.server.base_url}/download/{quote(artifact_name)}",
        }
        if self.server.version:
            body["version"] = self.server.version

        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def serve_file(self, served: ServedFile, send_body: bool) -> None:
        if not served.path.exists():
            self.send_error(404, f"missing {served.path}")
            return

        content_type = mimetypes.guess_type(str(served.path))[0] or "application/octet-stream"
        content_length = len(served.content) if served.content is not None else served.path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        if served.etag is not None:
            self.send_header("ETag", served.etag)
        self.end_headers()

        if send_body:
            if served.content is not None:
                self.wfile.write(served.content)
            else:
                with served.path.open("rb") as fp:
                    while True:
                        chunk = fp.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))


class UpdateServer(http.server.ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        handler,
        *,
        base_url: str,
        files: dict[str, ServedFile],
        version: str,
    ):
        super().__init__(address, handler)
        self.base_url = base_url
        self.files = files
        self.version = version

    def update_artifact_name(self, query: str) -> str:
        os_name = parse_qs(query).get("os", [""])[0].lower()
        if os_name in APP_ARTIFACTS:
            name = APP_ARTIFACTS[os_name]
            return name if name in self.files else ""

        for name in APP_ARTIFACTS.values():
            if name in self.files:
                return name
        return ""


def patched_install_script(path: Path, download_base_url: str) -> bytes:
    content = path.read_text(encoding="utf-8-sig")
    old = '$DownloadBaseURL = "https://ollama.com/download"'
    new = f'$DownloadBaseURL = "{download_base_url.rstrip("/")}"'
    if content.count(old) != 1:
        raise ValueError(
            f"{path} does not contain the expected DownloadBaseURL assignment; "
            "refusing to serve an install.ps1 that may download from production"
        )
    return content.replace(old, new, 1).encode("utf-8")


def discover_files(root: Path, dist: Path, omit_etags: bool, download_base_url: str) -> dict[str, ServedFile]:
    files: dict[str, ServedFile] = {}
    for path in dist.iterdir() if dist.exists() else []:
        if not path.is_file():
            continue
        if path.name.endswith(".etag"):
            continue
        if path.name == "install.ps1":
            content = patched_install_script(path, download_base_url)
            etag = None if omit_etags else (sidecar_etag(path) or content_etag(content))
            files[path.name] = ServedFile(path, etag, content)
            continue
        etag = None if omit_etags else (sidecar_etag(path) or file_etag(path))
        files[path.name] = ServedFile(path, etag)

    install_script = root / "scripts" / "install.ps1"
    if "install.ps1" not in files and install_script.exists():
        content = patched_install_script(install_script, download_base_url)
        etag = None if omit_etags else content_etag(content)
        files["install.ps1"] = ServedFile(install_script, etag, content)
    return files


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Loopback bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--dist", type=Path, default=root / "dist", help="Directory of artifacts to serve")
    parser.add_argument("--version", default="", help="Optional update response version")
    parser.add_argument("--omit-etags", action="store_true", help="Serve files without ETag headers")
    parser.add_argument("--prepare-only", action="store_true", help="Print resolved files without starting the server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    dist = args.dist.resolve()
    base_url = f"http://{args.host}:{args.port}"
    try:
        files = discover_files(root, dist, args.omit_etags, f"{base_url}/download")
    except ValueError as err:
        raise SystemExit(str(err)) from err

    if not files:
        raise SystemExit(f"no files found in dist directory: {dist}")

    print(f"Update endpoint: {base_url}/api/update")
    print(f"Dist:            {dist}")
    print(f"Version field:   {args.version or '(omitted)'}")
    for name, served in sorted(files.items()):
        print(f"/download/{quote(name)}")
        print(f"  file: {served.path}")
        print(f"  etag: {served.etag or '(omitted)'}")

    if args.prepare_only:
        return

    server = UpdateServer(
        (args.host, args.port),
        UpdateHandler,
        base_url=base_url,
        files=files,
        version=args.version,
    )
    print("Serving until interrupted.")
    server.serve_forever()


if __name__ == "__main__":
    main()
