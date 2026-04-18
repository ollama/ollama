#!/usr/bin/env python3

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple


def github_token_from_git_credentials() -> str:
    req = "protocol=https\nhost=github.com\n\n"
    out = subprocess.run(
        ["git", "credential", "fill"],
        input=req,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("No GitHub token available from git credential helper")


def api_call(
    method: str,
    url: str,
    headers: Dict[str, str],
    data: Optional[Dict] = None,
    timeout: int = 20,
) -> Tuple[int, Dict]:
    payload = None
    req_headers = dict(headers)
    if data is not None:
        req_headers["Content-Type"] = "application/json"
        payload = json.dumps(data).encode()
    req = urllib.request.Request(url, method=method, headers=req_headers, data=payload)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
            parsed = {"raw": body}
        return e.code, parsed


def ensure_label(base: str, headers: Dict[str, str]) -> None:
    status, _ = api_call(
        "POST",
        f"{base}/labels",
        headers,
        {
            "name": "agent-ready",
            "color": "1d76db",
            "description": "Ready for autonomous agent implementation",
        },
    )
    if status not in (201, 422):
        raise RuntimeError(f"Unable to ensure label agent-ready: status={status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync agent-ready label to next-wave issues")
    parser.add_argument("--repo", default="kushin77/ollama")
    parser.add_argument("--queue", default=".github/next_wave_execution_queue.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-issues", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    with open(args.queue, "r", encoding="utf-8") as handle:
        queue = json.load(handle)

    token = github_token_from_git_credentials()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "copilot-codex",
    }
    base = f"https://api.github.com/repos/{args.repo}"
    ensure_label(base, headers)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": args.repo,
        "queue": args.queue,
        "processed": 0,
        "labels_added": 0,
        "already_labeled": 0,
        "failures": [],
        "issues": [],
    }

    queue_issues = queue.get("next_wave_issues", [])
    if args.max_issues > 0:
        queue_issues = queue_issues[:args.max_issues]

    for issue in queue_issues:
        issue_number = int(issue["number"])
        report["processed"] += 1
        status, response = api_call(
            "POST",
            f"{base}/issues/{issue_number}/labels",
            headers,
            {"labels": ["agent-ready"]},
            timeout=args.timeout,
        )
        if status == 200:
            names = {label.get("name") for label in response if isinstance(label, dict)}
            if "agent-ready" in names:
                had_agent_ready = any(label == "agent-ready" for label in issue.get("labels", []))
                if had_agent_ready:
                    report["already_labeled"] += 1
                else:
                    report["labels_added"] += 1
            report["issues"].append({"issue": issue_number, "status": "ok"})
            continue
        report["failures"].append({"issue": issue_number, "status": status, "response": response})
        report["issues"].append({"issue": issue_number, "status": f"failed:{status}"})

    out_path = args.output or f".github/agent_ready_sync_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": out_path,
        "processed": report["processed"],
        "labels_added": report["labels_added"],
        "already_labeled": report["already_labeled"],
        "failures": len(report["failures"]),
    }, indent=2))


if __name__ == "__main__":
    main()
