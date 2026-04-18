#!/usr/bin/env python3
"""Generate a live, immutable triage manifest for GitHub issues.

This script is idempotent: it reads live GitHub issue state, cross-references
local git history for evidence, and writes a deterministic JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMMIT_RECORD_SEPARATOR = "\x1e"
COMMIT_FIELD_SEPARATOR = "\x1f"
MERGE_PULL_REQUEST_RE = re.compile(r"^Merge pull request #\d+\b", re.IGNORECASE)
CLOSURE_VERBS = (
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
    "implement",
    "implements",
    "implemented",
    "address",
    "addresses",
    "addressed",
)
STOPWORDS = {
    "add",
    "adds",
    "agent",
    "and",
    "are",
    "bug",
    "feat",
    "feature",
    "fix",
    "for",
    "from",
    "github",
    "http",
    "into",
    "issue",
    "migrate",
    "migration",
    "not",
    "pmo",
    "pull",
    "repo",
    "repository",
    "replace",
    "that",
    "the",
    "this",
    "with",
}


def resolve_token() -> str:
    for key in ("OLLAMA_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        value = os.getenv(key, "").strip()
        if value:
            return value

    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()

    raise RuntimeError("no GitHub token found in env or git credential helper")


class GitHubIssueClient:
    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{repo}"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ollama-live-triage-manifest",
        }

    def request(self, path: str) -> Any:
        request = urllib.request.Request(f"{self.base_url}{path}", headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {exc.code} for {path}: {body}") from exc

    def fetch_open_issues(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = self.request(f"/issues?state=open&per_page=100&page={page}")
            if not batch:
                break
            for item in batch:
                if "pull_request" not in item:
                    issues.append(item)
            if len(batch) < 100:
                break
            page += 1
        return issues


def issue_reference_pattern(issue_number: int) -> re.Pattern[str]:
    return re.compile(rf"(?<!\d)#{issue_number}(?!\d)", re.IGNORECASE)


def issue_url_pattern(issue_number: int) -> re.Pattern[str]:
    return re.compile(rf"/issues/{issue_number}(?!\d)", re.IGNORECASE)


def extract_issue_keywords(title: str) -> list[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", title)
    keywords: list[str] = []
    for token in re.findall(r"[a-z0-9]+", normalized.lower()):
        if len(token) < 4 or token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords


def is_merge_pull_request(subject: str) -> bool:
    return bool(MERGE_PULL_REQUEST_RE.match(subject.strip()))


def parse_git_log(output: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for record in output.split(COMMIT_RECORD_SEPARATOR):
        if not record.strip():
            continue
        parts = record.split(COMMIT_FIELD_SEPARATOR)
        if len(parts) != 3:
            continue
        commit, subject, body = parts
        entries.append(
            {
                "commit": commit.strip(),
                "subject": subject.strip(),
                "body": body.strip(),
            }
        )
    return entries


def classify_commit_evidence(issue_number: int, issue_title: str, commit: dict[str, str]) -> dict[str, Any] | None:
    subject = commit["subject"]
    body = commit["body"]
    combined = f"{subject}\n{body}"

    if is_merge_pull_request(subject):
        return None

    has_reference = bool(issue_reference_pattern(issue_number).search(combined)) or bool(
        issue_url_pattern(issue_number).search(combined)
    )
    if not has_reference:
        return None

    keyword_matches = [
        keyword
        for keyword in extract_issue_keywords(issue_title)
        if re.search(rf"\b{re.escape(keyword)}\b", combined, re.IGNORECASE)
    ]
    has_closure_verb = any(
        re.search(rf"\b{verb}\b", combined, re.IGNORECASE) for verb in CLOSURE_VERBS
    )

    if has_closure_verb and len(keyword_matches) >= 2:
        confidence = "high"
    elif len(keyword_matches) >= 3:
        confidence = "medium"
    else:
        return None

    return {
        "commit": commit["commit"],
        "subject": subject,
        "matched_keywords": keyword_matches,
        "has_closure_verb": has_closure_verb,
        "confidence": confidence,
    }


def git_evidence(issue_number: int, issue_title: str) -> list[dict[str, Any]]:
    reference_pattern = issue_reference_pattern(issue_number).pattern
    url_pattern = issue_url_pattern(issue_number).pattern
    proc = subprocess.run(
        [
            "git",
            "log",
            f"--format=%H{COMMIT_FIELD_SEPARATOR}%s{COMMIT_FIELD_SEPARATOR}%b{COMMIT_RECORD_SEPARATOR}",
            "--all",
            "--perl-regexp",
            f"--grep={reference_pattern}",
            f"--grep={url_pattern}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    results: list[dict[str, Any]] = []
    for commit in parse_git_log(proc.stdout):
        evidence = classify_commit_evidence(issue_number, issue_title, commit)
        if evidence is not None:
            results.append(evidence)
    return results


def labels(issue: dict[str, Any]) -> set[str]:
    return {label["name"] for label in issue.get("labels", [])}


def summarize(issues: list[dict[str, Any]]) -> dict[str, Any]:
    wave_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    agent_ready: list[dict[str, Any]] = []
    needs_evidence: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for issue in issues:
        issue_labels = labels(issue)
        for label in issue_labels:
            label_counts[label] += 1
            if label.startswith("wave/"):
                wave_counts[label] += 1

        item = {
            "number": issue["number"],
            "title": issue["title"],
            "url": issue["html_url"],
            "labels": sorted(issue_labels),
            "updated_at": issue["updated_at"],
        }

        if "needs-evidence" in issue_labels:
            item["evidence_commits"] = git_evidence(issue["number"], issue["title"])
            needs_evidence.append(item)

        if "status/triaged" in issue_labels and "status/planned-wave" in issue_labels and "needs-evidence" not in issue_labels:
            agent_ready.append(item)

        if "blocked" in issue_labels or "status/blocked" in issue_labels:
            blocked.append(item)

    close_candidates = [issue for issue in needs_evidence if issue.get("evidence_commits")]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "open_issue_count": len(issues),
        "agent_ready_count": len(agent_ready),
        "needs_evidence_count": len(needs_evidence),
        "close_candidate_count": len(close_candidates),
        "blocked_count": len(blocked),
        "wave_counts": dict(sorted(wave_counts.items())),
        "top_labels": label_counts.most_common(25),
        "close_candidates": close_candidates,
        "needs_evidence": needs_evidence,
        "agent_ready": agent_ready,
        "blocked": blocked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate live GitHub issue triage manifest")
    parser.add_argument("--repo", default="kushin77/ollama", help="GitHub repository owner/name")
    parser.add_argument("--output", default=".github/live_triage_manifest.json", help="Output JSON file path")
    args = parser.parse_args()

    token = resolve_token()
    client = GitHubIssueClient(args.repo, token)
    issues = client.fetch_open_issues()
    manifest = summarize(issues)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"open issues: {manifest['open_issue_count']}")
    print(f"agent ready: {manifest['agent_ready_count']}")
    print(f"needs evidence: {manifest['needs_evidence_count']}")
    print(f"close candidates: {manifest['close_candidate_count']}")
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
