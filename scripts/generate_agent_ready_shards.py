#!/usr/bin/env python3

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def is_pull_request(issue):
    if issue.get("pull_request") or issue.get("is_pull_request"):
        return True

    for key in ("url", "html_url"):
        value = issue.get(key)
        if isinstance(value, str) and "/pull/" in value:
            return True

    return False


def severity_from_labels(labels):
    names = [label.lower() for label in labels]
    if any("critical" in n or "priority-p0" in n for n in names):
        return "critical"
    if any(k in n for n in names for k in ("priority/high", "security", "bug", "high")):
        return "high"
    if any(k in n for n in names for k in ("priority/medium", "performance", "medium")):
        return "medium"
    return "low"


def load_issues(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        issues = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("open_issues"), list):
            issues = raw["open_issues"]
        elif isinstance(raw.get("issues"), list):
            issues = raw["issues"]
        elif isinstance(raw.get("agent_ready"), list) or isinstance(raw.get("needs_evidence"), list):
            issues = []
            if isinstance(raw.get("agent_ready"), list):
                issues.extend(raw["agent_ready"])
            if isinstance(raw.get("needs_evidence"), list):
                issues.extend(raw["needs_evidence"])
        else:
            issues = []
    else:
        issues = []

    normalized = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue

        if is_pull_request(issue):
            continue

        raw_labels = issue.get("labels", [])
        labels = []
        for label in raw_labels:
            if isinstance(label, dict):
                labels.append(label.get("name", ""))
            elif isinstance(label, str):
                labels.append(label)

        number = issue.get("number")
        if number is None:
            continue

        normalized.append(
            {
                "number": int(number),
                "title": issue.get("title", ""),
                "labels": labels,
                "url": issue.get("url") or issue.get("html_url"),
                "updated_at": issue.get("updatedAt") or issue.get("updated_at"),
                "severity": severity_from_labels(labels),
            }
        )
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic shards for open agent-ready issues")
    parser.add_argument("--input", default=".github/agent_ready_queue.json")
    parser.add_argument("--output", default=".github/agent_ready_shards.json")
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    issues = load_issues(args.input)
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    issues.sort(key=lambda i: (order.get(i["severity"], 99), i["number"]))

    shards = [{"shard": i + 1, "issues": []} for i in range(args.shards)]
    for idx, issue in enumerate(issues):
        shards[idx % args.shards]["issues"].append(issue)

    for shard in shards:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in shard["issues"]:
            counts[issue["severity"]] += 1
        shard["size"] = len(shard["issues"])
        shard["severity_counts"] = counts
        shard["issue_numbers"] = [issue["number"] for issue in shard["issues"]]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.input,
        "open_agent_ready_count": len(issues),
        "shard_count": args.shards,
        "assignment_policy": "severity_then_issue_number_round_robin",
        "shards": shards,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": args.output,
        "open_agent_ready_count": len(issues),
        "shard_count": args.shards,
        "shard_sizes": [s["size"] for s in shards],
    }, indent=2))


if __name__ == "__main__":
    main()
