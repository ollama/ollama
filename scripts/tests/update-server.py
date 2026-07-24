#!/usr/bin/env python3
"""Serve a loopback Ollama app update endpoint for manual upgrade testing."""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import mimetypes
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_install_script(root: Path) -> Path:
    dist_script = root / "dist" / "install.ps1"
    if dist_script.exists():
        return dist_script
    return root / "scripts" / "install.ps1"


def file_etag(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return '"' + hashlib.sha256(payload).hexdigest()[:32] + '"'


def patch_install_script(source: Path, output: Path, download_base_url: str) -> Path:
    content = source.read_text(encoding="utf-8-sig")
    old = '$DownloadBaseURL = "https://ollama.com/download"'
    new = f'$DownloadBaseURL = "{download_base_url.rstrip("/")}"'
    if old not in content:
        raise SystemExit(f"{source} does not contain expected DownloadBaseURL assignment")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content.replace(old, new), encoding="utf-8")
    return output


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
            self.serve_update_json(send_body)
            return
        if path == "/download/install.ps1":
            self.serve_file(self.server.install_script, self.server.script_etag, send_body)
            return
        if path == "/download/OllamaSetup.exe":
            self.serve_file(self.server.installer, self.server.installer_etag, send_body)
            return

        self.send_error(404, "not found")

    def serve_update_json(self, send_body: bool) -> None:
        body = {
            "url": f"{self.server.base_url}/download/OllamaSetup.exe",
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

    def serve_file(self, path: Path, etag: Optional[str], send_body: bool) -> None:
        if not path.exists():
            self.send_error(404, f"missing {path}")
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        if etag is not None:
            self.send_header("ETag", etag)
        self.end_headers()

        if send_body:
            with path.open("rb") as fp:
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
        installer: Path,
        install_script: Path,
        version: str,
        installer_etag: Optional[str],
        script_etag: Optional[str],
    ):
        super().__init__(address, handler)
        self.base_url = base_url
        self.installer = installer
        self.install_script = install_script
        self.version = version
        self.installer_etag = installer_etag
        self.script_etag = script_etag


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Loopback bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--version", default="", help="Optional update response version")
    parser.add_argument("--installer-etag", default="", help="Override the installer ETag")
    parser.add_argument("--script-etag", default="", help="Override the install.ps1 ETag")
    parser.add_argument("--omit-installer-etag", action="store_true", help="Serve OllamaSetup.exe without an ETag")
    parser.add_argument("--omit-script-etag", action="store_true", help="Serve install.ps1 without an ETag")
    parser.add_argument("--installer", type=Path, default=root / "dist" / "OllamaSetup.exe", help="OllamaSetup.exe to serve")
    parser.add_argument("--install-script", type=Path, default=default_install_script(root), help="install.ps1 to serve")
    parser.add_argument(
        "--patch-install-script",
        action="store_true",
        help="Generate a test install.ps1 with DownloadBaseURL pointed at this server",
    )
    parser.add_argument(
        "--output-install-script",
        type=Path,
        default=root / ".cache" / "update-server" / "install.ps1",
        help="Path for the generated patched install.ps1",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate files and print paths without starting the server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    installer = args.installer.resolve()
    install_script = args.install_script.resolve()
    base_url = f"http://{args.host}:{args.port}"

    if not installer.exists():
        raise SystemExit(f"installer not found: {installer}")
    if not install_script.exists():
        raise SystemExit(f"install script not found: {install_script}")

    if args.patch_install_script:
        install_script = patch_install_script(
            install_script,
            args.output_install_script.resolve(),
            f"{base_url}/download",
        )

    installer_etag = None if args.omit_installer_etag else (args.installer_etag or file_etag(installer))
    script_etag = None if args.omit_script_etag else (args.script_etag or file_etag(install_script))

    print(f"Update endpoint: {base_url}/api/update")
    print(f"Installer URL:   {base_url}/download/OllamaSetup.exe")
    print(f"install.ps1:     {install_script}")
    print(f"Installer:       {installer}")
    print(f"Version field:   {args.version or '(omitted)'}")
    print(f"Installer ETag:  {installer_etag or '(omitted)'}")
    print(f"install.ps1 ETag:{script_etag or '(omitted)'}")

    if args.prepare_only:
        return

    server = UpdateServer(
        (args.host, args.port),
        UpdateHandler,
        base_url=base_url,
        installer=installer,
        install_script=install_script,
        version=args.version,
        installer_etag=installer_etag,
        script_etag=script_etag,
    )
    print("Serving until interrupted.")
    server.serve_forever()


if __name__ == "__main__":
    main()
