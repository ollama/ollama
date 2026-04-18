#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ISSUE_ONLY_LABELS = {"agent-ready"}


def github_token_from_git_credentials() -> str:
    request = "protocol=https\nhost=github.com\n\n"
    output = subprocess.run(
        ["git", "credential", "fill"],
        input=request,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    for line in output.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("No GitHub token available from git credential helper")


def api_call(method: str, url: str, headers: dict[str, str], data: dict[str, Any] | None = None) -> tuple[int, Any]:
    request_headers = dict(headers)
    payload = None
    if data is not None:
        request_headers["Content-Type"] = "application/json"
        payload = json.dumps(data).encode("utf-8")

    request = urllib.request.Request(url, method=method, headers=request_headers, data=payload)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
            parsed = {"raw": body}
        return exc.code, parsed


def fetch_open_items(repo: str, headers: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    pull_requests: list[dict[str, Any]] = []
    page = 1
    while True:
        status, batch = api_call(
            "GET",
            f"https://api.github.com/repos/{repo}/issues?state=open&per_page=100&page={page}",
            headers,
        )
        if status != 200:
            raise RuntimeError(f"Unable to fetch open items: status={status} payload={batch}")
        if not batch:
            break
        for item in batch:
            if "pull_request" in item:
                pull_requests.append(item)
            else:
                issues.append(item)
        page += 1
    return issues, pull_requests


def labels_for(item: dict[str, Any]) -> list[str]:
    return [label.get("name", "") for label in item.get("labels", []) if isinstance(label, dict)]


def is_issue_only_label(label: str) -> bool:
    return label in ISSUE_ONLY_LABELS or label.startswith("shard/")


def cleanup_pr_labels(repo: str, headers: dict[str, str], pull_requests: list[dict[str, Any]]) -> dict[str, Any]:
    report = {
        "processed_pull_requests": len(pull_requests),
        "updated_pull_requests": 0,
        "removed_labels": [],
        "failures": [],
    }

    for pull_request in pull_requests:
        number = int(pull_request["number"])
        labels = labels_for(pull_request)
        removable = [label for label in labels if is_issue_only_label(label)]
        if not removable:
            continue

        removed_for_pr = []
        for label in removable:
            status, payload = api_call(
                "DELETE",
                f"https://api.github.com/repos/{repo}/issues/{number}/labels/{urllib.parse.quote(label, safe='')}",
                headers,
            )
            if status not in (200, 404):
                report["failures"].append(
                    {
                        "number": number,
                        "label": label,
                        "status": status,
                        "message": payload.get("message", ""),
                    }
                )
                continue
            removed_for_pr.append(label)

        if removed_for_pr:
            report["updated_pull_requests"] += 1
            report["removed_labels"].append({"number": number, "labels": removed_for_pr})

    return report


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    output = result.stdout.strip()
    return json.loads(output) if output.startswith("{") else {"stdout": output}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile agent-ready issue state and exclude pull requests")
    parser.add_argument("--repo", default="kushin77/ollama")
    parser.add_argument("--queue-output", default=".github/agent_ready_queue.json")
    parser.add_argument("--shards-output", default=".github/agent_ready_shards.json")
    parser.add_argument("--report-output", default="")
    args = parser.parse_args()

    token = github_token_from_git_credentials()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "copilot-agent-ready-reconciler",
    }

    issues, pull_requests = fetch_open_items(args.repo, headers)
    pr_cleanup = cleanup_pr_labels(args.repo, headers, pull_requests)

    queue_result = run([
        "python3",
        "scripts/build_agent_ready_queue.py",
        "--repo",
        args.repo,
        "--output",
        args.queue_output,
    ])
    shard_result = run([
        "python3",
        "scripts/generate_agent_ready_shards.py",
        "--input",
        args.queue_output,
        "--output",
        args.shards_output,
        "--shards",
        "4",
    ])
    apply_result = run([
        "python3",
        "scripts/apply_agent_ready_shards.py",
        "--repo",
        args.repo,
        "--manifest",
        args.shards_output,
    ])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": args.repo,
        "real_open_issue_count": len(issues),
        "open_pull_request_count": len(pull_requests),
        "pull_request_label_cleanup": pr_cleanup,
        "queue_result": queue_result,
        "shard_result": shard_result,
        "apply_result": apply_result,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = Path(args.report_output or f".github/agent_ready_reconciliation_report_{stamp}.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "reconciliation_report": str(report_path),
                "apply_report": apply_result.get("report"),
                "processed": apply_result.get("processed", 0),
                "updated": apply_result.get("updated", 0),
                "already_correct": apply_result.get("already_correct", 0),
                "failed": apply_result.get("failed", 0),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
