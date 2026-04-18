#!/usr/bin/env python3

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def refresh_open_issue_snapshot(output_path: Path):
    script = r'''
import json, subprocess, urllib.request
from datetime import datetime, timezone
from pathlib import Path
owner, repo='kushin77','ollama'
req='protocol=https\nhost=github.com\n\n'
out=subprocess.run(['git','credential','fill'],input=req,text=True,capture_output=True,check=True).stdout
t=''
for l in out.splitlines():
    if l.startswith('password='):
        t=l.split('=',1)[1].strip(); break
h={'Authorization':f'token {t}','Accept':'application/vnd.github+json','User-Agent':'copilot-codex'}
issues=[]; page=1
while True:
    r=urllib.request.Request(f'https://api.github.com/repos/{owner}/{repo}/issues?state=open&per_page=100&page={page}',headers=h)
    with urllib.request.urlopen(r) as resp:
        d=json.loads(resp.read().decode())
    if not d:
        break
    for it in d:
        if 'pull_request' in it:
            continue
        issues.append({
            'number': it['number'],
            'title': it.get('title',''),
            'labels': [x.get('name','') for x in it.get('labels',[])],
            'updated_at': it.get('updated_at'),
            'url': it.get('html_url'),
        })
    page += 1
snapshot={
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'repository': f'{owner}/{repo}',
    'open_issue_count': len(issues),
    'open_issues': issues,
}
Path(''' + repr(str(output_path)) + r''').write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
print(len(issues))
'''
    count = run(["python3", "-c", script])
    return int(count)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Run one autonomous issue management cycle")
    parser.add_argument(
        "--config",
        default=".github/autonomous_cycle.iac.json",
        help="Path to cycle configuration",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)

    snapshot_path = Path(config["paths"]["snapshot"])
    manifest_path = Path(config["paths"]["manifest"])
    deferred_queue_path = Path(config["paths"]["deferred_queue"])
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    open_issue_count = refresh_open_issue_snapshot(snapshot_path)

    run([
        "python3",
        "scripts/generate_autonomous_batches.py",
        "--snapshot",
        str(snapshot_path),
        "--output",
        str(manifest_path),
        "--wave-size",
        str(config["execution"]["wave_size"]),
    ])

    previous_report = config["execution"].get("resume_from_previous_report", "")
    if not previous_report and deferred_queue_path.exists():
        deferred = load_json(deferred_queue_path)
        previous_report = deferred.get("source_report", "")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wave_report = report_dir / f"wave_assignment_report_{timestamp}.json"

    cmd = [
        "python3",
        "scripts/apply_wave_assignments.py",
        "--repo",
        config["repository"],
        "--manifest",
        str(manifest_path),
        "--comment-budget",
        str(config["execution"]["comment_budget"]),
        "--max-issues",
        str(config["execution"]["max_issues_per_cycle"]),
        "--output",
        str(wave_report),
    ]
    if config["execution"].get("comments_only", False):
        cmd.append("--comments-only")
    if previous_report:
        cmd.extend(["--previous-report", previous_report])

    run(cmd)

    wave_data = load_json(wave_report)
    deferred_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": config["repository"],
        "source_report": str(wave_report),
        "rate_limit_blocked": wave_data.get("rate_limit_blocked", False),
        "deferred_count": len(wave_data.get("deferred_comment_issues", [])),
        "deferred_issues": wave_data.get("deferred_comment_issues", []),
        "retry_instructions": [
            "rerun scripts/run_autonomous_issue_cycle.py --config .github/autonomous_cycle.iac.json",
            "commit the updated wave_assignment_report_*.json and deferred queue after each pass",
            "keep bounded budgets to respect GitHub secondary rate limits",
        ],
    }
    deferred_queue_path.write_text(json.dumps(deferred_payload, indent=2) + "\n", encoding="utf-8")

    cycle_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": config["repository"],
        "open_issue_count": open_issue_count,
        "wave_manifest": str(manifest_path),
        "latest_wave_report": str(wave_report),
        "deferred_queue": str(deferred_queue_path),
        "rate_limit_blocked": wave_data.get("rate_limit_blocked", False),
        "processed": wave_data.get("processed", 0),
        "comments_added": wave_data.get("comments_added", 0),
        "comment_deferred": wave_data.get("comment_deferred", 0),
    }
    summary_path = report_dir / f"autonomous_cycle_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(cycle_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cycle_summary, indent=2))


if __name__ == "__main__":
    main()
