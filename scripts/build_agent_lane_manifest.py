#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def classifications_by_issue(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for item in payload.get("classifications", []):
        if not item.get("success"):
            continue
        issue_number = item.get("issue")
        if issue_number is None:
            continue
        mapping[int(issue_number)] = item.get("classification", {})
    return mapping


def lane_summary(issues: list[dict[str, Any]]) -> dict[str, Any]:
    severity_counts: Counter[str] = Counter()
    ai_priority_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    complexity_counts: Counter[str] = Counter()
    duplicate_candidates = 0
    p0_count = 0

    for issue in issues:
        severity_counts[issue.get("severity", "unknown")] += 1
        ai = issue.get("ai_classification", {})
        ai_priority_counts[ai.get("priority", "unknown")] += 1
        category_counts[ai.get("category", "unknown")] += 1
        complexity_counts[ai.get("complexity", "unknown")] += 1
        if ai.get("is_duplicate"):
            duplicate_candidates += 1
        if "priority-p0" in set(issue.get("labels", [])):
            p0_count += 1

    return {
        "issue_count": len(issues),
        "p0_count": p0_count,
        "severity_counts": dict(sorted(severity_counts.items())),
        "ai_priority_counts": dict(sorted(ai_priority_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "complexity_counts": dict(sorted(complexity_counts.items())),
        "duplicate_candidates": duplicate_candidates,
    }


def recommended_model_for_lane(summary: dict[str, Any]) -> str:
    if summary.get("p0_count", 0) > 0 or summary.get("severity_counts", {}).get("critical", 0) > 0:
        return "llama3:8b"
    if summary.get("duplicate_candidates", 0) > 5:
        return "llama3:8b"
    if summary.get("complexity_counts", {}).get("complex", 0) > 10:
        return "mistral:7b"
    return "phi3-fast:latest"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build enriched agent execution lane manifest")
    parser.add_argument("--shards", default=".github/agent_ready_shards.json")
    parser.add_argument("--classifications", default=".github/ollama_classification_report.json")
    parser.add_argument("--output", default=".github/agent_execution_lanes.json")
    args = parser.parse_args()

    shards_payload = load_json(Path(args.shards))
    classification_payload = load_json(Path(args.classifications))
    classification_map = classifications_by_issue(classification_payload)

    lanes: list[dict[str, Any]] = []
    for shard in shards_payload.get("shards", []):
        enriched_issues = []
        for issue in shard.get("issues", []):
            issue_number = int(issue["number"])
            enriched_issues.append(
                {
                    **issue,
                    "ai_classification": classification_map.get(issue_number, {}),
                }
            )

        summary = lane_summary(enriched_issues)
        lanes.append(
            {
                "lane": f"shard/{shard['shard']}",
                "shard": shard["shard"],
                "recommended_model": recommended_model_for_lane(summary),
                "assignment_policy": "severity_then_issue_number_round_robin",
                "summary": summary,
                "issue_numbers": [issue["number"] for issue in enriched_issues],
                "issues": enriched_issues,
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_shards": args.shards,
        "source_classifications": args.classifications,
        "open_issue_count": shards_payload.get("open_agent_ready_count", 0),
        "lane_count": len(lanes),
        "execution_contract": {
            "real_issues_only": True,
            "pull_requests_excluded": True,
            "idempotent": True,
            "requires_commit_evidence": True,
            "close_only_after_verification": True,
        },
        "recommended_start_order": [lane["lane"] for lane in sorted(lanes, key=lambda item: item["shard"])],
        "lanes": lanes,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "open_issue_count": payload["open_issue_count"],
                "lane_count": payload["lane_count"],
                "recommended_start_order": payload["recommended_start_order"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
