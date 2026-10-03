import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    with ThreadingHTTPServer((args.host, args.port), SimpleHTTPRequestHandler) as server:
        args.ready_file.write_text(str(server.server_address[1]), encoding="ascii")
        server.serve_forever()


if __name__ == "__main__":
    main()
