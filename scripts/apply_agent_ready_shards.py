#!/usr/bin/env python3

import argparse
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


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


def api_call(method: str, url: str, headers: Dict[str, str], data: Optional[Dict] = None) -> Tuple[int, Dict]:
    request_headers = dict(headers)
    body = None
    if data is not None:
        request_headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()

    req = urllib.request.Request(url, method=method, headers=request_headers, data=body)
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read().decode()
            return resp.status, (json.loads(payload) if payload else {})
    except urllib.error.HTTPError as e:
        payload = e.read().decode()
        try:
            parsed = json.loads(payload) if payload else {}
        except Exception:
            parsed = {"raw": payload}
        return e.code, parsed


def ensure_label(base: str, headers: Dict[str, str], name: str, color: str, description: str) -> None:
    status, _ = api_call(
        "POST",
        f"{base}/labels",
        headers,
        {
            "name": name,
            "color": color,
            "description": description,
        },
    )
    if status not in (201, 422):
        raise RuntimeError(f"Unable to ensure label {name}: status={status}")


def issue_label_set(base: str, headers: Dict[str, str], issue_number: int) -> Set[str]:
    status, payload = api_call(
        "GET",
        f"{base}/issues/{issue_number}",
        headers,
    )
    if status != 200:
        return set()
    labels = payload.get("labels", [])
    names: Set[str] = set()
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def build_issue_shard_map(payload: Dict) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for shard in payload.get("shards", []):
        shard_num = int(shard["shard"])
        for issue in shard.get("issues", []):
            mapping[int(issue["number"])] = shard_num
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic shard labels to open agent-ready issues")
    parser.add_argument("--repo", default="kushin77/ollama")
    parser.add_argument("--manifest", default=".github/agent_ready_shards.json")
    parser.add_argument("--max-issues", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    token = github_token_from_git_credentials()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "copilot-codex",
    }
    base = f"https://api.github.com/repos/{args.repo}"

    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    issue_to_shard = build_issue_shard_map(manifest)
    issue_numbers = sorted(issue_to_shard.keys())
    if args.max_issues > 0:
        issue_numbers = issue_numbers[:args.max_issues]

    max_shard = max(issue_to_shard.values()) if issue_to_shard else 0
    for shard_num in range(1, max_shard + 1):
        ensure_label(
            base,
            headers,
            f"shard/{shard_num}",
            "0e8a16",
            f"Deterministic autonomous execution shard {shard_num}",
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": args.repo,
        "manifest": args.manifest,
        "processed": 0,
        "updated": 0,
        "already_correct": 0,
        "failed": 0,
        "failures": [],
    }

    for issue_number in issue_numbers:
        target = f"shard/{issue_to_shard[issue_number]}"
        report["processed"] += 1

        current = issue_label_set(base, headers, issue_number)
        if target in current and not any(name.startswith("shard/") and name != target for name in current):
            report["already_correct"] += 1
            continue

        remove = [name for name in current if name.startswith("shard/") and name != target]
        for old_label in remove:
            status_del, payload_del = api_call(
                "DELETE",
                f"{base}/issues/{issue_number}/labels/{urllib.parse.quote(old_label, safe='')}",
                headers,
            )
            if status_del not in (200, 404):
                report["failed"] += 1
                report["failures"].append(
                    {
                        "issue": issue_number,
                        "step": "remove",
                        "label": old_label,
                        "status": status_del,
                        "message": payload_del.get("message", ""),
                    }
                )

        status_add, payload_add = api_call(
            "POST",
            f"{base}/issues/{issue_number}/labels",
            headers,
            {"labels": [target]},
        )
        if status_add == 200:
            report["updated"] += 1
            continue

        report["failed"] += 1
        report["failures"].append(
            {
                "issue": issue_number,
                "step": "add",
                "label": target,
                "status": status_add,
                "message": payload_add.get("message", ""),
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output or f".github/agent_ready_shard_apply_report_{ts}.json"
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report": out_path,
                "processed": report["processed"],
                "updated": report["updated"],
                "already_correct": report["already_correct"],
                "failed": report["failed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()