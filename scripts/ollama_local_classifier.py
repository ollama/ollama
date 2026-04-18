#!/usr/bin/env python3
"""
Local Ollama Model Integration for Issue Classification
Uses a local Ollama instance, configured via OLLAMA_HOST or the host profile,
to classify issues in parallel.
Supports: Mistral, Llama3, Phi3 models for efficient triage.
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import subprocess
import urllib.request
import urllib.error
import re

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = "mistral:7b"  # Fast, good reasoning for classification


class OllamaClassifier:
    """Interface to local Ollama instance for issue classification."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, model: str = DEFAULT_MODEL):
        self.host = host
        self.model = model
        self.api_url = f"{host}/api/generate"

    @staticmethod
    def _extract_json_object(text: str) -> Dict:
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found")

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:idx + 1])

        raise ValueError("Unbalanced JSON object in model response")

    @staticmethod
    def _normalize_classification(classification: Dict) -> Dict:
        priority = str(classification.get("priority", "normal")).lower()
        category = str(classification.get("category", "chore")).lower()
        complexity = str(classification.get("complexity", "moderate")).lower()

        if priority not in {"critical", "high", "normal", "low"}:
            priority = "normal"
        if category not in {"bug", "feature", "docs", "refactor", "chore", "question"}:
            category = "chore"
        if complexity not in {"trivial", "simple", "moderate", "complex"}:
            if any(word in complexity for word in ("trivial", "simple", "moderate", "complex")):
                complexity = next(
                    word for word in ("trivial", "simple", "moderate", "complex") if word in complexity
                )
            else:
                complexity = "moderate"

        confidence = classification.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            numbers = re.findall(r"\d+(?:\.\d+)?", str(confidence))
            confidence = float(numbers[0]) if numbers else 0.5
        confidence = max(0.0, min(1.0, confidence))

        return {
            "priority": priority,
            "category": category,
            "is_duplicate": bool(classification.get("is_duplicate", False)),
            "complexity": complexity,
            "confidence": confidence,
        }

    def classify_issue(self, issue_number: int, title: str, body: str = "") -> Dict:
        """
        Classify a single issue using local Ollama model.
        Returns classification metadata.
        """
        prompt = f"""Classify this GitHub issue concisely:

Issue #{issue_number}: {title}
{f"Description: {body[:500]}" if body else ""}

Respond with JSON only:
{{
  "priority": "critical|high|normal|low",
  "category": "bug|feature|docs|refactor|chore|question",
  "is_duplicate": true|false,
  "complexity": "trivial|simple|moderate|complex",
  "confidence": 0.0-1.0
}}"""

        try:
            payload = json.dumps(
                {"model": self.model, "prompt": prompt, "stream": False}
            )
            req = urllib.request.Request(
                self.api_url,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

            # Extract JSON from response
            try:
                response_text = result.get("response", "{}")
                classification = self._normalize_classification(
                    self._extract_json_object(response_text)
                )
                return {
                    "issue": issue_number,
                    "classification": classification,
                    "model": self.model,
                    "success": True,
                }
            except (json.JSONDecodeError, ValueError):
                return {
                    "issue": issue_number,
                    "error": "Failed to parse model response",
                    "success": False,
                }

        except urllib.error.URLError:
            return {
                "issue": issue_number,
                "error": f"Cannot connect to {self.host}",
                "success": False,
            }
        except Exception as e:
            return {"issue": issue_number, "error": str(e), "success": False}

    def check_available_models(self) -> List[str]:
        """Get list of available models on Ollama instance."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"Error checking models: {e}", file=sys.stderr)
            return []


def batch_classify_from_queue(
    queue_file: Path,
    output_file: Path,
    model: str = DEFAULT_MODEL,
    limit: int = 0,
    host: str = DEFAULT_OLLAMA_HOST,
) -> None:
    """Classify issues from agent-ready queue using local Ollama."""

    classifier = OllamaClassifier(host=host, model=model)

    # Check available models
    available = classifier.check_available_models()
    print(f"Available Ollama models: {available}")

    if model not in available:
        print(f"Warning: {model} not found. Using first available.", file=sys.stderr)
        if available:
            classifier.model = available[0]
        else:
            print("Error: No models available on Ollama instance", file=sys.stderr)
            return

    # Load queue
    try:
        with open(queue_file) as f:
            queue = json.load(f)
    except FileNotFoundError:
        print(f"Queue file not found: {queue_file}", file=sys.stderr)
        return

    issues = queue if isinstance(queue, list) else queue.get("issues", [])
    filtered_issues = []
    skipped_pull_requests = 0
    for issue in issues:
        if isinstance(issue, dict):
            if issue.get("pull_request") or issue.get("is_pull_request"):
                skipped_pull_requests += 1
                continue
            url = issue.get("url") or issue.get("html_url")
            if isinstance(url, str) and "/pull/" in url:
                skipped_pull_requests += 1
                continue
        filtered_issues.append(issue)
    issues = filtered_issues
    if limit > 0:
        issues = issues[:limit]

    print(f"Classifying {len(issues)} issues with {classifier.model}...")
    if skipped_pull_requests:
        print(f"Skipped {skipped_pull_requests} pull request entries from the input queue")

    results = []
    for i, issue in enumerate(issues, 1):
        if isinstance(issue, dict):
            number = issue.get("number")
            title = issue.get("title", "")
            body = issue.get("body", "")
        else:
            number = issue
            title = ""
            body = ""

        print(f"  [{i}/{len(issues)}] Classifying issue #{number}...", end=" ")

        classification = classifier.classify_issue(number, title, body)
        results.append(classification)

        if classification.get("success"):
            print("✓")
        else:
            print(f"✗ ({classification.get('error', 'unknown')})")

    # Write results
    report = {
        "timestamp": subprocess.check_output(
            "date -u +%Y-%m-%dT%H:%M:%SZ", shell=True, text=True
        ).strip(),
        "model": classifier.model,
        "total_classified": len(results),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "skipped_pull_requests": skipped_pull_requests,
        "classifications": results,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nClassification results saved to {output_file}")
    print(
        f"Success: {report['successful']}/{report['total_classified']} issues classified"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Classify GitHub issues using local Ollama models"
    )
    parser.add_argument(
        "--queue", "-q",
        type=Path,
        default=Path(".github/agent_ready_queue.json"),
        help="Input queue file (default: .github/agent_ready_queue.json)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(".github/ollama_classification_report.json"),
        help="Output classification report",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama host URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Limit number of issues to classify (0 = all)",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Check what models are available on Ollama",
    )

    args = parser.parse_args()

    if args.check_models:
        classifier = OllamaClassifier(host=args.host)
        available = classifier.check_available_models()
        print("Available Ollama models:")
        for model in available:
            print(f"  - {model}")
        return

    batch_classify_from_queue(args.queue, args.output, args.model, args.limit, args.host)


if __name__ == "__main__":
    main()
