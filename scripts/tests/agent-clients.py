#!/usr/bin/env python3
"""Exercise an Ollama model through real coding-agent tool workflows."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


CLIENT_ARGS = {
    "muse": (
        "exec", "--trust-workspace", "--disable-web-tools",
        "--no-foreign-personal-context", "--user-input-auto-resolve",
        "--max-model-steps", "20", "{prompt}",
    ),
    "claude": (
        "--print", "--dangerously-skip-permissions", "--no-session-persistence",
        "--disable-slash-commands", "--allowed-tools",
        "Read,Write,Edit,Bash,Glob", "--output-format", "json", "{prompt}",
    ),
    "pi": (
        "--offline", "--no-session", "--no-extensions", "--no-skills",
        "--no-prompt-templates", "--no-themes", "--no-context-files",
        "--tools", "read,bash,edit,write,ls", "--approve", "--mode", "json",
        "--print", "{prompt}",
    ),
    "opencode": (
        "run", "--pure", "--format", "json", "--dangerously-skip-permissions",
        "{prompt}",
    ),
    "codex": (
        "exec", "--ephemeral", "--ignore-rules", "--skip-git-repo-check",
        "--approve-for-me", "--json", "-C", "{workspace}", "{prompt}",
    ),
}
SCENARIOS = {
    "workspace-edit": "list, read, shell, write, edit, and verify",
    "missing-file-recovery": "recover from an intentional read error, then complete the workspace edit",
}
FIXTURE_FILES = {
    "go.mod": "module example.com/agent-clients\n\ngo 1.24.0\n",
    "pkg/alpha.go": 'package pkg\n\nfunc Alpha() string { return "alpha" }\n',
    "pkg/beta.go": 'package pkg\n\nfunc Beta() string { return "beta" }\n',
}
SCENARIO_PROMPTS = {
    "workspace-edit": """Work only inside this disposable workspace. Complete this exact tool sequence:

1. List the inputs directory. Its one filename is intentionally not provided.
2. Use your file-reading tool to read the discovered file and obtain its token.
3. Use your file-reading tool to read go.mod and determine the module path.
4. Use the shell to count all .go files below pkg.
5. Use your file-writing tool to write scratch/result.txt with exactly these three lines, substituting only values learned from tools:
module=<module path>
go_files=<count>
token=<token>
6. Use your file-editing tool to append this exact fourth line without rewriting the file:
status=verified
7. Use your file-reading tool to read scratch/result.txt again and verify it.

After the tools finish, include AGENT_CLIENTS_OK in your final response.
""",
    "missing-file-recovery": """Work only inside this disposable workspace. Complete this exact recovery sequence:

1. List the inputs directory. Its one filename is intentionally not provided.
2. Use your file-reading tool to read {{MISSING_FILE}}. The file intentionally does not exist; observe the real tool error and recover from it. Do not use the shell for this step.
3. Use your file-reading tool to read the discovered file in inputs and obtain its token.
4. Use your file-reading tool to read go.mod and determine the module path.
5. Use the shell to count all .go files below pkg.
6. Use your file-writing tool to write scratch/result.txt with exactly these three lines, substituting only values learned from tools:
module=<module path>
go_files=<count>
token=<token>
7. Use your file-editing tool to append this exact fourth line without rewriting the file:
status=verified
8. Use your file-reading tool to read scratch/result.txt again and verify it.

After the tools finish, include AGENT_CLIENTS_OK in your final response.
""",
}
FINAL_MARKER = "AGENT_CLIENTS_OK"
FAILED_POST = re.compile(r"\|\s*[45]\d\d\s*\|.*\|\s*POST(?:\s|$)")


def parse_list(parser: argparse.ArgumentParser, raw: str, allowed: tuple[str, ...], kind: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [value for value in values if value not in allowed]
    if not values or unknown or len(values) != len(set(values)):
        parser.error(f"invalid {kind} list {raw!r}; choose unique values from {','.join(allowed)}")
    return values


def resolve_executable(parser: argparse.ArgumentParser, value: str) -> Path:
    if os.sep in value:
        path = Path(value).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    else:
        found = shutil.which(value)
        if found:
            return Path(found).resolve()
    parser.error(f"executable not found: {value}")


def client_env(home: Path, host: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "OLLAMA_HOST": host,
        "MUSE_NO_AUTO_UPDATE": "1",
        "PI_OFFLINE": "1",
        "NO_COLOR": "1",
    })
    return env


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_bounded(
    command: list[str], workspace: Path, env: dict[str, str],
    stdout_path: Path, stderr_path: Path, timeout: int,
    server_log_path: Path | None,
) -> tuple[int, bool, str | None]:
    log_offset = server_log_path.stat().st_size if server_log_path else 0
    last_server_error = None

    def server_error() -> str | None:
        nonlocal last_server_error, log_offset
        if not server_log_path:
            return None
        with server_log_path.open("r", encoding="utf-8", errors="replace") as log:
            log.seek(log_offset)
            lines = log.readlines()
            log_offset = log.tell()
        for line in lines:
            if " level=ERROR " in line:
                last_server_error = line.strip()
            if FAILED_POST.search(line):
                if last_server_error and " error=" in last_server_error:
                    return "error=" + last_server_error.split(" error=", 1)[1]
                return line.strip()
        return None

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, cwd=workspace, env=env, stdout=stdout, stderr=stderr,
            start_new_session=True, text=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_process(process)
                    return 124, True, None
                try:
                    returncode = process.wait(timeout=min(0.5, remaining))
                    return returncode, False, server_error()
                except subprocess.TimeoutExpired:
                    error = server_error()
                    if error:
                        stop_process(process)
                        return 125, False, error
                    if time.monotonic() >= deadline:
                        stop_process(process)
                        return 124, True, None
        except KeyboardInterrupt:
            stop_process(process)
            raise


def normalize_host(host: str) -> str:
    return host if "://" in host else f"http://{host}"


def start_server(ollama: Path, run_root: Path) -> tuple[subprocess.Popen, object, str, Path]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    address = f"127.0.0.1:{port}"
    host = f"http://{address}"
    temp_dir = run_root / "tmp"
    temp_dir.mkdir()
    log = (run_root / "server.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "OLLAMA_HOST": address,
        "OLLAMA_DEBUG": "2",
        "OLLAMA_DEBUG_LOG_REQUESTS": "1",
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_NOPRUNE": "1",
        "TMPDIR": str(temp_dir),
        "TMP": str(temp_dir),
        "TEMP": str(temp_dir),
        "GIN_MODE": "release",
    })
    process = subprocess.Popen(
        [str(ollama), "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited during startup ({process.returncode})")
            try:
                with urllib.request.urlopen(host + "/api/version", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not become ready within 60 seconds")

        request_dirs = list(temp_dir.glob("ollama-request-logs-*"))
        if len(request_dirs) != 1:
            raise RuntimeError("server did not create its raw request-log directory")
        return process, log, host, request_dirs[0]
    except Exception:
        stop_process(process)
        log.close()
        raise


def preflight_model(ollama: Path, host: str, model: str, env: dict[str, str]) -> None:
    try:
        result = subprocess.run(
            [str(ollama), "show", model], env=env, capture_output=True,
            text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"model preflight failed: {error}") from error
    if result.returncode:
        raise RuntimeError(f"model preflight failed: {(result.stderr or result.stdout).strip()}")


def run_case(
    client: str, scenario: str, ollama: Path, model: str, host: str,
    timeout: int, test_number: int, run_root: Path, request_logs: Path,
    server_log_path: Path | None,
) -> bool:
    name = f"{client} / {scenario}"
    case_root = run_root / client / scenario
    workspace = case_root / "workspace"
    home = case_root / "home"
    case_root.mkdir(parents=True)
    home.mkdir()
    workspace.mkdir()
    for relative_path, contents in FIXTURE_FILES.items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    (workspace / "scratch").mkdir()
    (workspace / "inputs").mkdir()

    token = secrets.token_hex(8)
    token_file = workspace / "inputs" / f"discovery-{token[:8]}.txt"
    token_file.write_text(f"token={token}\n", encoding="utf-8")
    expected = "\n".join((
        "module=example.com/agent-clients", "go_files=2",
        f"token={token}", "status=verified",
    ))
    prompt = SCENARIO_PROMPTS[scenario]
    prompt = prompt.replace("{{MISSING_FILE}}", f"missing-{token}.txt")
    (case_root / "prompt.txt").write_text(prompt, encoding="utf-8")
    (case_root / "expected-result.txt").write_text(expected + "\n", encoding="utf-8")

    extra = [value.format(prompt=prompt, workspace=workspace) for value in CLIENT_ARGS[client]]
    command = [str(ollama), "launch", client, "--model", model, "--yes", "--", *extra]
    print(f"# {name}: running", flush=True)
    returncode, timed_out, server_error = run_bounded(
        command, workspace, client_env(home, host), case_root / "stdout.log",
        case_root / "stderr.log", timeout, server_log_path,
    )

    failures = []
    if server_error:
        failures.append(f"server request failed: {server_error}")
    elif timed_out:
        failures.append("timed out")
    elif returncode:
        failures.append(f"client exited {returncode}")
    if not failures:
        stdout = (case_root / "stdout.log").read_text(encoding="utf-8", errors="replace")
        if FINAL_MARKER not in stdout:
            failures.append(f"final response omitted {FINAL_MARKER}")
        result_path = workspace / "scratch" / "result.txt"
        if not result_path.is_file():
            failures.append("scratch/result.txt was not created")
        elif "\n".join(result_path.read_text(encoding="utf-8").splitlines()) != expected:
            failures.append("scratch/result.txt has unexpected content")
        if not any(
            token in path.read_text(encoding="utf-8", errors="replace")
            for path in request_logs.glob("*_body.json")
        ):
            failures.append("discovery token absent from raw request logs")

    (case_root / "case.json").write_text(json.dumps({
        "client": client,
        "scenario": scenario,
        "returncode": returncode,
        "timed_out": timed_out,
        "server_error": server_error,
        "token": token,
        "token_file": token_file.name,
        "command": [*command[:-1], "<see prompt.txt>"],
        "failures": failures,
    }, indent=2) + "\n", encoding="utf-8")
    if not failures:
        shutil.rmtree(home)

    print(f"{'not ok' if failures else 'ok'} {test_number} - {name}", flush=True)
    for failure in failures:
        print(f"#   {failure}", flush=True)
    if failures:
        print(f"#   stdout: {case_root / 'stdout.log'}", flush=True)
        print(f"#   stderr: {case_root / 'stderr.log'}", flush=True)
    return not failures


def argument_parser(repo_root: Path) -> argparse.ArgumentParser:
    default_ollama = repo_root / "ollama"
    scenario_help = "\n".join(f"  {name}: {description}" for name, description in SCENARIOS.items())
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="scenarios:\n" + scenario_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", help="existing server; otherwise spawn and manage one")
    parser.add_argument(
        "--ollama", default=str(default_ollama if default_ollama.is_file() else "ollama")
    )
    parser.add_argument(
        "--clients", default=",".join(CLIENT_ARGS),
        help="comma-separated: " + ", ".join(CLIENT_ARGS),
    )
    parser.add_argument(
        "--scenarios", default=",".join(SCENARIOS),
        help="comma-separated scenario names",
    )
    parser.add_argument("--timeout", type=int, default=600, help="seconds per case")
    parser.add_argument("--server-log", type=Path, help="required with --host")
    parser.add_argument(
        "--request-logs", type=Path,
        help="OLLAMA_DEBUG_LOG_REQUESTS directory; required with --host",
    )
    return parser


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser = argument_parser(repo_root)
    args = parser.parse_args()
    ollama = resolve_executable(parser, args.ollama)
    clients = parse_list(parser, args.clients, tuple(CLIENT_ARGS), "client")
    scenarios = parse_list(parser, args.scenarios, tuple(SCENARIOS), "scenario")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.host and (not args.server_log or not args.request_logs):
        parser.error("--host requires --server-log and --request-logs")
    if not args.host and (args.server_log or args.request_logs):
        parser.error("--server-log and --request-logs require --host")

    if args.server_log:
        args.server_log = args.server_log.expanduser().resolve()
        if not args.server_log.is_file():
            parser.error(f"server log not found: {args.server_log}")
    if args.request_logs:
        args.request_logs = args.request_logs.expanduser().resolve()
        if not args.request_logs.is_dir():
            parser.error(f"request-log directory not found: {args.request_logs}")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    run_root = repo_root / ".cache" / "agent-client-tests" / stamp
    run_root.mkdir(parents=True)
    print("TAP version 13", flush=True)
    print(f"1..{len(clients) * len(scenarios)}", flush=True)
    print(f"# artifacts: {run_root}", flush=True)
    print(f"# clients: {', '.join(clients)}", flush=True)
    print("# scenarios:", flush=True)
    for scenario in scenarios:
        print(f"#   {scenario}: {SCENARIOS[scenario]}", flush=True)

    passed = failed = skipped = 0
    run_metadata = {
        "model": args.model,
        "server_mode": "existing" if args.host else "managed-per-client",
        "ollama": str(ollama),
        "timeout_seconds": args.timeout,
        "clients": clients,
        "scenarios": scenarios,
    }
    if args.host:
        run_metadata.update({
            "host": normalize_host(args.host),
            "server_log": str(args.server_log),
            "request_logs": str(args.request_logs),
        })
    (run_root / "run.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8",
    )

    if args.host:
        print(f"# server: using {run_metadata['host']}", flush=True)
        print(f"# server log: {args.server_log}", flush=True)
        print(f"# request logs: {args.request_logs}", flush=True)

    try:
        for client_index, client in enumerate(clients):
            client_root = run_root / client
            client_root.mkdir()
            server = None
            server_log = None
            try:
                if args.host:
                    host = run_metadata["host"]
                    request_logs = args.request_logs
                    log_path = args.server_log
                else:
                    print(f"# {client}: starting server", flush=True)
                    server, server_log, host, request_logs = start_server(ollama, client_root)
                    log_path = client_root / "server.log"
                    print(f"# {client}: server ready at {host} (log: {log_path})", flush=True)

                (client_root / "server.json").write_text(json.dumps({
                    "host": host,
                    "server_log": str(log_path),
                    "request_logs": str(request_logs),
                }, indent=2) + "\n", encoding="utf-8")

                metadata_home = client_root / "preflight-home"
                metadata_home.mkdir()
                env = client_env(metadata_home, host)
                print(f"# {client}: checking model {args.model}", flush=True)
                preflight_model(ollama, host, args.model, env)
                print(f"# {client}: model ready", flush=True)

                for index, scenario in enumerate(scenarios):
                    test_number = client_index * len(scenarios) + index + 1
                    ok = run_case(
                        client, scenario, ollama, args.model, host, args.timeout,
                        test_number, run_root, request_logs,
                        None if args.host else log_path,
                    )
                    passed += int(ok)
                    failed += int(not ok)
                    if not ok:
                        for remaining_index, remaining in enumerate(scenarios[index + 1:], index + 1):
                            remaining_number = client_index * len(scenarios) + remaining_index + 1
                            print(
                                f"ok {remaining_number} - {client} / {remaining} "
                                "# SKIP prior scenario failed",
                                flush=True,
                            )
                            skipped += 1
                        break
            finally:
                if server is not None:
                    print(f"# {client}: stopping server", flush=True)
                    stop_process(server)
                    print(f"# {client}: server stopped", flush=True)
                if server_log is not None:
                    server_log.close()
    except KeyboardInterrupt:
        print("Bail out! interrupted", flush=True)
        return 130
    except RuntimeError as error:
        print(f"Bail out! {error}", flush=True)
        return 2
    print(f"# summary: {passed} passed, {failed} failed, {skipped} skipped")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
