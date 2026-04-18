#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def fetch_open_issues(repo: str, token: str) -> list[dict[str, Any]]:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "copilot-agent-ready-queue-builder",
    }
    issues: list[dict[str, Any]] = []
    page = 1
    while True:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/issues?state=open&per_page=100&page={page}",
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        for item in batch:
            if "pull_request" in item:
                continue
            issues.append(item)
        page += 1
    return issues


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": int(issue["number"]),
        "title": issue.get("title", ""),
        "labels": [label.get("name", "") for label in issue.get("labels", [])],
        "url": issue.get("html_url", ""),
        "updated_at": issue.get("updated_at"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical agent-ready queue excluding pull requests")
    parser.add_argument("--repo", default="kushin77/ollama")
    parser.add_argument("--output", default=".github/agent_ready_queue.json")
    args = parser.parse_args()

    token = github_token_from_git_credentials()
    issues = fetch_open_issues(args.repo, token)

    agent_ready = []
    for issue in issues:
        labels = {label.get("name", "") for label in issue.get("labels", [])}
        if "agent-ready" not in labels:
            continue
        agent_ready.append(normalize_issue(issue))

    agent_ready.sort(key=lambda issue: issue["number"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(agent_ready, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "repository": args.repo,
                "open_issue_count": len(issues),
                "agent_ready_count": len(agent_ready),
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
