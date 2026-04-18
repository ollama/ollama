#!/usr/bin/env python3
"""
Parallel Issue Triage using Local Ollama + Copilot
Runs Ollama local classification in parallel with GitHub triage work
to accelerate issue assessment and categorization.
"""

import json
import threading
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

class LocalOllamaTriageWorker:
    """Worker thread for parallel Ollama-based issue classification."""

    def __init__(
        self,
        queue_file: Path,
        model: str = "mistral:7b",
        batch_size: int = 20,
        host: str = DEFAULT_OLLAMA_HOST,
    ):
        self.queue_file = queue_file
        self.model = model
        self.batch_size = batch_size
        self.host = host
        self.results = []
        self.error = None

    def run(self):
        """Execute Ollama classification in parallel."""
        try:
            import_script = Path(__file__).parent / "ollama_local_classifier.py"
            output = Path(".github") / f"ollama_local_triage_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"

            cmd = [
                "python3",
                str(import_script),
                "--queue", str(self.queue_file),
                "--output", str(output),
                "--model", self.model,
                "--host", self.host,
                "--limit", str(self.batch_size),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("\n✓ Ollama classification complete")
                print(result.stdout)
                with open(output) as f:
                    self.results = json.load(f)
            else:
                self.error = result.stderr
                print(f"✗ Ollama classification failed: {result.stderr}")

        except Exception as e:
            self.error = str(e)
            print(f"✗ Worker error: {e}")


def run_parallel_triage(
    queue_file: Path,
    model: str = "mistral:7b",
    batch_size: int = 20,
    host: str = DEFAULT_OLLAMA_HOST,
):
    """
    Run Ollama classification in parallel with other triage work.

    Returns tuple of (results, errors) from Ollama worker thread.
    """

    print("=" * 70)
    print("PARALLEL TRIAGE MODE: Local Ollama + Copilot GitHub API")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  Queue file: {queue_file}")
    print(f"  Ollama model: {model}")
    print(f"  Batch size: {batch_size}")
    print(f"  Host: {host}")

    # Start Ollama worker thread
    print(f"\n🚀 Starting Ollama classification worker (non-blocking)...")
    worker = LocalOllamaTriageWorker(queue_file, model, batch_size, host)
    thread = threading.Thread(target=worker.run, daemon=False)
    thread.start()

    print("   Worker thread started in background")
    print("\n💡 While Ollama processes issues in parallel,")
    print("   you can run other triage operations:")
    print()
    print("   # Run autonomous cycles")
    print("   $ python3 scripts/run_autonomous_issue_cycle.py --config .github/autonomous_cycle.iac.json")
    print()
    print("   # Apply shard labels")
    print("   $ python3 scripts/apply_agent_ready_shards.py --manifest .github/agent_ready_shards.json")
    print()
    print("   # Or continue with your workflow...")
    print()

    # Wait for worker to complete
    print(f"⏳ Waiting for Ollama worker to complete...")
    thread.join(timeout=600)  # 10 minute timeout

    if thread.is_alive():
        print("⚠️  Timeout: Worker still running after 10 minutes")
        return None, "Worker timeout"

    if worker.error:
        print(f"❌ Worker failed: {worker.error}")
        return None, worker.error

    print(f"\n✅ Parallel triage complete!")
    return worker.results, None


def create_unified_report(github_report: Path, ollama_results: dict, output: Path) -> None:
    """
    Merge GitHub triage report and Ollama classification results
    into a unified intelligence report.
    """

    try:
        with open(github_report) as f:
            github_data = json.load(f)
    except FileNotFoundError:
        github_data = {}

    # Map Ollama classifications by issue number
    ollama_map = {}
    if ollama_results and ollama_results.get("classifications"):
        for item in ollama_results["classifications"]:
            if item.get("success"):
                ollama_map[item["issue"]] = item.get("classification", {})

    # Enrich with Ollama insights
    unified = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sources": {
            "github_api": str(github_report) if github_report else "none",
            "ollama_local": host,
        },
        "github_triage_summary": github_data,
        "ollama_enrichment": {
            "model_used": ollama_results.get("model") if ollama_results else "none",
            "classifications_count": len(ollama_map),
            "by_issue": ollama_map,
        },
        "merged_insights": [],
    }

    # Merge insights
    if github_data and github_data.get("issues"):
        for issue in github_data["issues"]:
            num = issue.get("number")
            insight = {
                "issue": num,
                "github_data": issue,
                "ollama_classification": ollama_map.get(num),
            }
            unified["merged_insights"].append(insight)

    with open(output, "w") as f:
        json.dump(unified, f, indent=2)

    print(f"\n📄 Unified report: {output}")
    print(f"   - GitHub insights: {len(github_data.get('issues', []))} issues")
    print(f"   - Ollama classifications: {len(ollama_map)} issues")


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel issue triage with local Ollama models"
    )
    parser.add_argument(
        "--queue", "-q",
        type=Path,
        default=Path(".github/agent_ready_queue.json"),
        help="Queue file to classify",
    )
    parser.add_argument(
        "--model", "-m",
        default="mistral:7b",
        help="Ollama model to use",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=20,
        help="Number of issues to classify (default: 20)",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for completion instead of returning immediately",
    )
    parser.add_argument(
        "--merge-report",
        type=Path,
        help="GitHub triage report to merge with Ollama results",
    )

    args = parser.parse_args()

    # Run parallel classification
    results, error = run_parallel_triage(args.queue, args.model, args.batch, args.ollama_host)

    if error:
        sys.exit(1)

    # Optionally merge with GitHub report
    if args.merge_report and results:
        output = Path(".github") / f"unified_triage_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        create_unified_report(args.merge_report, results, output)


if __name__ == "__main__":
    main()
