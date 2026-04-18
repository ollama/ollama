#!/usr/bin/env python3
"""Auto remediation helper (analysis only).

Runs `pip-audit --format=json` if available, and emits a prioritized
remediation plan into `remediation/dep_audit_<ts>.json` and
`remediation/plan_<ts>.md`.

This script does NOT modify dependency files or create PRs. It's a
safe analysis tool to speed manual remediation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "remediation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pip_audit() -> list[dict[str, Any]]:
    """Run pip-audit and return parsed JSON vulnerabilities list.

    Requires `pip-audit` on PATH. If unavailable, informs the user.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("pip-audit is not installed. Install it with: pip install pip-audit")
        return []
    except subprocess.CalledProcessError as e:
        print("pip-audit failed:", e.stderr)
        return []

    try:
        data = json.loads(result.stdout)
    except Exception as e:
        print("Failed to parse pip-audit output:", e)
        return []
    # pip-audit JSON shape may vary; we normalize to a list of vulns
    if isinstance(data, dict) and "vulns" in data:
        return data.get("vulns", [])
    if isinstance(data, list):
        return data
    return []


def prioritize(vulns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}

    def key(v: dict[str, Any]):
        sev = v.get("severity", "unknown")
        return (severity_order.get(sev, 4), v.get("package", ""))

    return sorted(vulns, key=key)


def build_plan(vulns: list[dict[str, Any]]) -> dict[str, Any]:
    plan = {"generated_at": datetime.utcnow().isoformat() + "Z", "vulnerabilities": []}
    for v in vulns:
        pkg = v.get("package") or v.get("name")
        installed = v.get("installed_version") or v.get("version")
        advisory = v.get("advisory", {})
        fix_versions = v.get("fix_versions") or advisory.get("fix_versions") or []
        plan["vulnerabilities"].append(
            {
                "package": pkg,
                "installed_version": installed,
                "severity": v.get("severity"),
                "description": (
                    advisory.get("summary") if isinstance(advisory, dict) else v.get("description")
                ),
                "fix_versions": fix_versions,
            }
        )
    return plan


def write_outputs(plan: dict[str, Any]) -> None:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = OUT_DIR / f"dep_audit_{ts}.json"
    md_path = OUT_DIR / f"plan_{ts}.md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Dependency Remediation Plan ({plan.get('generated_at')})\n\n")
        fh.write(
            "Prioritize critical/high issues first. Review suggested fix versions and test locally.\n\n"
        )
        for v in plan.get("vulnerabilities", []):
            fh.write(
                f"- **{v.get('package')}**: installed {v.get('installed_version')} — severity: {v.get('severity')}\n"
            )
            if v.get("fix_versions"):
                fh.write(f"  - Suggested fixes: {', '.join(v.get('fix_versions'))}\n")
            if v.get("description"):
                fh.write(f"  - {v.get('description')}\n")
            fh.write("\n")

    print("Wrote:", json_path, md_path)


def main(dry_run: bool = True):
    vulns = run_pip_audit()
    if not vulns:
        print("No vulnerabilities found or pip-audit unavailable.")
        return
    prioritized = prioritize(vulns)
    plan = build_plan(prioritized)
    write_outputs(plan)
    if dry_run:
        print(
            "Dry run complete. Review remediation files under remediation/ and open PRs as needed."
        )
    else:
        print("Plan generated. Proceed to create branches/PRs for safe upgrades.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    main(dry_run=args.dry_run)
