#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_ollama_models(host: str) -> list[str]:
    req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
    with urllib.request.urlopen(req, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [model.get("name", "") for model in payload.get("models", [])]


def chunk(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def choose_model(recommended: str, available: list[str]) -> str:
    if recommended in available:
        return recommended
    if available:
        return available[0]
    return recommended


def worker_profile_for_lane(lane: str, summary: dict[str, Any]) -> str:
    if lane == "shard/1" or summary.get("p0_count", 0) > 0:
        return "claudeclaw-critical"
    if summary.get("duplicate_candidates", 0) > 0:
        return "hermes-review"
    return "hermes-delivery"


def build_batch_record(lane: dict[str, Any], batch_idx: int, issues: list[dict[str, Any]]) -> dict[str, Any]:
    issue_numbers = [issue["number"] for issue in issues]
    priorities = sorted({issue.get("ai_classification", {}).get("priority", "normal") for issue in issues})
    categories = sorted({issue.get("ai_classification", {}).get("category", "chore") for issue in issues})
    return {
        "batch_id": f"{lane['lane'].replace('/', '_')}_batch_{batch_idx:02d}",
        "lane": lane["lane"],
        "batch_index": batch_idx,
        "issue_count": len(issues),
        "issue_numbers": issue_numbers,
        "ai_priority_mix": priorities,
        "ai_category_mix": categories,
        "execution_requirements": {
            "commit_evidence_required": True,
            "close_after_verification_only": True,
            "idempotent_updates_only": True,
        },
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic per-lane autonomous work batches")
    parser.add_argument("--input", default=".github/agent_execution_lanes.json")
    parser.add_argument("--output", default=".github/agent_autonomous_dispatch.json")
    parser.add_argument("--workpack-dir", default=".github/lane_workpacks")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    args = parser.parse_args()

    lanes_payload = load_json(Path(args.input))
    available_models = fetch_ollama_models(args.ollama_host)

    workpack_dir = Path(args.workpack_dir)
    workpack_dir.mkdir(parents=True, exist_ok=True)

    dispatch_lanes: list[dict[str, Any]] = []
    for lane in lanes_payload.get("lanes", []):
        issues = lane.get("issues", [])
        issue_batches = chunk(issues, args.batch_size)
        worker_profile = worker_profile_for_lane(lane.get("lane", ""), lane.get("summary", {}))
        selected_model = choose_model(lane.get("recommended_model", ""), available_models)

        batches = [
            build_batch_record(lane, batch_idx=index, issues=batch)
            for index, batch in enumerate(issue_batches, start=1)
        ]

        lane_workpack = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lane": lane["lane"],
            "worker_profile": worker_profile,
            "model": {
                "recommended": lane.get("recommended_model"),
                "selected": selected_model,
                "available": available_models,
            },
            "summary": lane.get("summary", {}),
            "batch_size": args.batch_size,
            "batch_count": len(batches),
            "batches": batches,
            "global_rules": lanes_payload.get("execution_contract", {}),
        }

        workpack_file = workpack_dir / f"{lane['lane'].replace('/', '_')}_workpack.json"
        workpack_file.write_text(json.dumps(lane_workpack, indent=2) + "\n", encoding="utf-8")

        dispatch_lanes.append(
            {
                "lane": lane["lane"],
                "worker_profile": worker_profile,
                "selected_model": selected_model,
                "issue_count": lane.get("summary", {}).get("issue_count", 0),
                "batch_count": len(batches),
                "workpack": str(workpack_file),
            }
        )

    dispatch_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.input,
        "open_issue_count": lanes_payload.get("open_issue_count", 0),
        "lane_count": lanes_payload.get("lane_count", 0),
        "batch_size": args.batch_size,
        "ollama_host": args.ollama_host,
        "available_models": available_models,
        "recommended_start_order": lanes_payload.get("recommended_start_order", []),
        "dispatch_lanes": dispatch_lanes,
        "global_rules": lanes_payload.get("execution_contract", {}),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dispatch_manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "workpack_dir": str(workpack_dir),
                "lane_count": len(dispatch_lanes),
                "total_batches": sum(lane["batch_count"] for lane in dispatch_lanes),
                "batch_size": args.batch_size,
                "available_models": available_models,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
