#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
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


def priority_rank(labels: set[str]) -> int:
    if "priority/high" in labels:
        return 0
    if "priority/medium" in labels:
        return 1
    if "priority/low" in labels:
        return 2
    return 3


def wave_rank(labels: set[str]) -> int:
    values: list[int] = []
    for label in labels:
        if label.startswith("wave/"):
            try:
                values.append(int(label.split("/", 1)[1]))
            except ValueError:
                continue
    return min(values) if values else 999


def is_p0(title: str, labels: set[str]) -> bool:
    return "priority-p0" in labels or bool(re.search(r"\[p0\]|\bp0\b", title, re.IGNORECASE))


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    names = sorted(label["name"] for label in issue.get("labels", []))
    return {
        "number": issue["number"],
        "title": issue["title"],
        "labels": names,
        "url": issue["url"],
        "updated_at": issue["updatedAt"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic non-P0 execution queue")
    parser.add_argument("--repo", default="kushin77/ollama")
    parser.add_argument("--next-wave-size", type=int, default=50)
    parser.add_argument("--queue-output", default=".github/next_wave_execution_queue.json")
    parser.add_argument("--full-output", default=".github/non_p0_execution_queue.json")
    parser.add_argument("--report-output", default="")
    args = parser.parse_args()

    token = github_token_from_git_credentials()
    env = {**os.environ, "GH_TOKEN": token}

    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            args.repo,
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number,title,url,labels,updatedAt",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    issues = json.loads(result.stdout)

    non_p0_raw: list[dict[str, Any]] = []
    p0_count = 0
    for issue in issues:
        names = {label["name"].lower() for label in issue.get("labels", [])}
        if is_p0(issue["title"], names):
            p0_count += 1
            continue
        non_p0_raw.append(issue)

    non_p0_raw.sort(
        key=lambda issue: (
            priority_rank({label["name"].lower() for label in issue.get("labels", [])}),
            wave_rank({label["name"].lower() for label in issue.get("labels", [])}),
            issue["updatedAt"],
            issue["number"],
        )
    )

    non_p0 = [normalize_issue(issue) for issue in non_p0_raw]
    next_wave = non_p0[: max(0, args.next_wave_size)]

    priority_counts: Counter[str] = Counter()
    wave_counts: Counter[str] = Counter()
    missing_agent_ready = 0
    for issue in non_p0:
        labels = {label.lower() for label in issue["labels"]}
        if "agent-ready" not in labels:
            missing_agent_ready += 1
        for label in labels:
            if label.startswith("priority/"):
                priority_counts[label] += 1
            if label.startswith("wave/"):
                wave_counts[label] += 1

    generated_at = datetime.now(timezone.utc).isoformat()
    queue_payload = {
        "generated_at_utc": generated_at,
        "repository": args.repo,
        "selection": "open non-P0 issues sorted by priority, wave, updated_at",
        "next_wave_size": args.next_wave_size,
        "non_p0_count": len(non_p0),
        "next_wave_issues": next_wave,
    }
    full_payload = {
        "generated_at_utc": generated_at,
        "repository": args.repo,
        "open_issue_total": len(issues),
        "p0_count": p0_count,
        "non_p0_count": len(non_p0),
        "missing_agent_ready": missing_agent_ready,
        "priority_counts": dict(sorted(priority_counts.items())),
        "wave_counts": dict(sorted(wave_counts.items())),
        "issues": non_p0,
    }
    report_payload = {
        "generated_at_utc": generated_at,
        "repository": args.repo,
        "queue_output": args.queue_output,
        "full_output": args.full_output,
        "open_issue_total": len(issues),
        "p0_count": p0_count,
        "non_p0_count": len(non_p0),
        "next_wave_size": len(next_wave),
        "missing_agent_ready": missing_agent_ready,
        "priority_counts": dict(sorted(priority_counts.items())),
        "top_waves": wave_counts.most_common(20),
    }

    queue_path = Path(args.queue_output)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue_payload, indent=2) + "\n", encoding="utf-8")

    full_path = Path(args.full_output)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(full_payload, indent=2) + "\n", encoding="utf-8")

    if args.report_output:
        report_path = Path(args.report_output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = Path(f".github/non_p0_execution_queue_report_{stamp}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "queue": str(queue_path),
        "full": str(full_path),
        "report": str(report_path),
        "open_issue_total": len(issues),
        "p0_count": p0_count,
        "non_p0_count": len(non_p0),
        "next_wave_size": len(next_wave),
        "missing_agent_ready": missing_agent_ready,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
