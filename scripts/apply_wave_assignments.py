#!/usr/bin/env python3

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    h = dict(headers)
    body = None
    if data is not None:
        h["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    req = urllib.request.Request(url, method=method, headers=h, data=body)
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
    # 201 created, 422 already exists.
    if status not in (201, 422):
        raise RuntimeError(f"Unable to ensure label {name}: status={status}")


def get_issue_comments(base: str, headers: Dict[str, str], issue_number: int) -> List[Dict]:
    status, comments = api_call(
        "GET",
        f"{base}/issues/{issue_number}/comments?per_page=100",
        headers,
    )
    if status == 200 and isinstance(comments, list):
        return comments
    return []


def has_marker(comments: List[Dict], marker: str) -> bool:
    for comment in comments:
        body = comment.get("body") or ""
        if marker in body:
            return True
    return False


def build_issue_wave_map(manifest: Dict) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for wave in manifest.get("waves", []):
        wave_num = wave["wave"]
        for issue in wave.get("issues", []):
            mapping[int(issue["number"])] = int(wave_num)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic wave labels/comments to GitHub issues")
    parser.add_argument("--repo", default="kushin77/ollama")
    parser.add_argument("--manifest", default=".github/autonomous_execution_batches.json")
    parser.add_argument("--previous-report", default="")
    parser.add_argument("--comment-budget", type=int, default=25)
    parser.add_argument("--comments-only", action="store_true")
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
    marker = "<!-- wave-assignment-v1 -->"

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    issue_to_wave = build_issue_wave_map(manifest)
    issue_numbers = sorted(issue_to_wave.keys())

    deferred_filter: Optional[set] = None
    if args.previous_report:
        with open(args.previous_report, "r", encoding="utf-8") as f:
            prev = json.load(f)
        deferred_filter = {int(item["issue"]) for item in prev.get("deferred_comment_issues", [])}

    if deferred_filter is not None:
        issue_numbers = [issue for issue in issue_numbers if issue in deferred_filter]

    if args.max_issues > 0:
        issue_numbers = issue_numbers[:args.max_issues]

    # Ensure label set exists before assignment only when labels are being mutated.
    if not args.comments_only:
        max_wave = max(issue_to_wave.values()) if issue_to_wave else 0
        for wave_num in range(1, max_wave + 1):
            ensure_label(
                base,
                headers,
                f"wave/{wave_num}",
                "5319e7",
                f"Autonomous execution batch {wave_num}",
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": args.repo,
        "manifest": args.manifest,
        "processed": 0,
        "labels_added": 0,
        "comments_added": 0,
        "already_marked": 0,
        "comment_deferred": 0,
        "rate_limit_blocked": False,
        "failures": [],
        "deferred_comment_issues": [],
    }

    comment_budget = args.comment_budget

    for issue_number in issue_numbers:
        wave_num = issue_to_wave[issue_number]
        report["processed"] += 1

        # Add deterministic wave label.
        label_name = f"wave/{wave_num}"
        if not args.comments_only:
            s_label, _ = api_call(
                "POST",
                f"{base}/issues/{issue_number}/labels",
                headers,
                {"labels": [label_name]},
            )
            if s_label == 200:
                report["labels_added"] += 1
            elif s_label != 422:
                report["failures"].append({
                    "issue": issue_number,
                    "wave": wave_num,
                    "step": "label",
                    "status": s_label,
                })

        comments = get_issue_comments(base, headers, issue_number)
        if has_marker(comments, marker):
            report["already_marked"] += 1
            continue

        if deferred_filter is not None and issue_number not in deferred_filter:
            continue

        if comment_budget <= 0 or report["rate_limit_blocked"]:
            report["comment_deferred"] += 1
            report["deferred_comment_issues"].append(
                {"issue": issue_number, "wave": wave_num, "reason": "budget_or_rate_limit"}
            )
            continue

        body = (
            f"{marker}\n"
            f"Autonomous assignment: this issue is in execution wave {wave_num}.\n\n"
            "Closure policy:\n"
            "- Close only after linked implementation commit(s).\n"
            "- Include verification evidence in issue comments.\n"
            "- Re-runs are idempotent via marker checks and committed reports.\n"
        )

        s_comment, r_comment = api_call(
            "POST",
            f"{base}/issues/{issue_number}/comments",
            headers,
            {"body": body},
        )

        if s_comment == 201:
            report["comments_added"] += 1
            comment_budget -= 1
            continue

        message = str(r_comment.get("message", "")).lower()
        if s_comment == 403 and "secondary rate limit" in message:
            report["rate_limit_blocked"] = True
            report["comment_deferred"] += 1
            report["deferred_comment_issues"].append(
                {"issue": issue_number, "wave": wave_num, "reason": "secondary_rate_limit"}
            )
            continue

        report["comment_deferred"] += 1
        report["deferred_comment_issues"].append(
            {"issue": issue_number, "wave": wave_num, "reason": f"comment_status_{s_comment}"}
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output or f".github/wave_assignment_report_{timestamp}.json"
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": out_path,
        "processed": report["processed"],
        "labels_added": report["labels_added"],
        "comments_added": report["comments_added"],
        "already_marked": report["already_marked"],
        "comment_deferred": report["comment_deferred"],
        "rate_limit_blocked": report["rate_limit_blocked"],
        "failures": len(report["failures"]),
    }, indent=2))
if __name__ == "__main__":
    main()
