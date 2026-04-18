#!/usr/bin/env python3

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def infer_severity(labels):
    lowered = [label.lower() for label in labels]
    if any("critical" in label or "priority-p0" in label for label in lowered):
        return "critical"
    if any(
        keyword in label
        for label in lowered
        for keyword in ("high", "security", "bug", "critical-path")
    ):
        return "high"
    if any(keyword in label for label in lowered for keyword in ("medium", "performance")):
        return "medium"
    return "low"


def chunk(items, chunk_size):
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic autonomous execution batches")
    parser.add_argument(
        "--snapshot",
        default=".github/open_issues_snapshot.json",
        help="Path to committed open issues snapshot JSON",
    )
    parser.add_argument(
        "--input",
        help="Backward-compatible alias for --snapshot",
    )
    parser.add_argument(
        "--output",
        default=".github/autonomous_execution_batches.json",
        help="Path for generated batches manifest",
    )
    parser.add_argument(
        "--wave-size",
        type=int,
        default=25,
        help="Number of issues per execution wave",
    )
    args = parser.parse_args()

    snapshot_arg = args.input or args.snapshot
    snapshot_path = Path(snapshot_arg)
    output_path = Path(args.output)

    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)

    if isinstance(snapshot, list):
        issues = snapshot
        repository = None
    else:
        issues = snapshot.get("open_issues") or snapshot.get("issues") or []
        repository = snapshot.get("repository")

    normalized = []
    for issue in issues:
        if not isinstance(issue, dict):
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

        severity = issue.get("severity") or infer_severity(labels)
        normalized.append(
            {
                "number": number,
                "title": issue.get("title", ""),
                "labels": labels,
                "updated_at": issue.get("updated_at"),
                "url": issue.get("url") or issue.get("html_url"),
                "severity": severity,
            }
        )

    normalized.sort(key=lambda issue: (SEVERITY_ORDER.get(issue["severity"], 99), issue["number"]))

    waves = []
    for wave_index, issues_in_wave in enumerate(chunk(normalized, args.wave_size), start=1):
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in issues_in_wave:
            severity_counts[issue["severity"]] += 1

        waves.append(
            {
                "wave": wave_index,
                "size": len(issues_in_wave),
                "severity_counts": severity_counts,
                "issue_numbers": [issue["number"] for issue in issues_in_wave],
                "issues": issues_in_wave,
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "source_snapshot": str(snapshot_path),
        "open_issue_count": len(normalized),
        "wave_size": args.wave_size,
        "wave_count": len(waves),
        "selection_policy": "severity_then_issue_number",
        "execution_rules": [
            "implement with code/docs/tests as appropriate",
            "post commit evidence before closure",
            "close only after verification evidence",
            "commit generated reports and state artifacts",
        ],
        "waves": waves,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Generated {len(waves)} waves covering {len(normalized)} issues -> {output_path}")


if __name__ == "__main__":
    main()
