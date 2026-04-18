#!/usr/bin/env bash

# Deterministic rerun matrix for host-profile-aware scripts.
# Captures first-run/second-run evidence for development and production host profiles.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/scripts/host-profile.sh" ]; then
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/scripts/host-profile.sh"
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${TMPDIR:-/tmp}/ollama-host-profile-idempotency}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$ARTIFACT_ROOT/$RUN_ID"
DEFAULT_PROFILES=(development production)
DEFAULT_COMMANDS=(
  "./scripts/preflight.sh"
  "./scripts/onboard.sh --dry-run --yes"
)
PROFILES=()
COMMANDS=()
OPEN_ISSUE="${OPEN_ISSUE:-true}"

usage() {
  cat <<'EOF'
Usage: scripts/host-profile-matrix.sh [options]

Options:
  --profile NAME     Add a profile to test (repeatable; default: development, production)
  --command CMD      Add a command to run under each profile (repeatable)
  --artifact-root P  Write artifacts under P instead of /tmp
  --no-open-issue    Disable auto-creating a GitHub issue on regression
  -h, --help         Show this help text
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      [ "$#" -ge 2 ] || { echo "Missing value for --profile" >&2; exit 2; }
      PROFILES+=("$2")
      shift 2
      ;;
    --command)
      [ "$#" -ge 2 ] || { echo "Missing value for --command" >&2; exit 2; }
      COMMANDS+=("$2")
      shift 2
      ;;
    --artifact-root)
      [ "$#" -ge 2 ] || { echo "Missing value for --artifact-root" >&2; exit 2; }
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --no-open-issue)
      OPEN_ISSUE="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "${#PROFILES[@]}" -eq 0 ]; then
  PROFILES=("${DEFAULT_PROFILES[@]}")
fi

if [ "${#COMMANDS[@]}" -eq 0 ]; then
  COMMANDS=("${DEFAULT_COMMANDS[@]}")
fi

mkdir -p "$RUN_ROOT"

safe_name() {
  printf '%s' "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'
}

capture_repo_state() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  git -C "$PROJECT_ROOT" status --short > "$output_dir/git-status.txt"
  git -C "$PROJECT_ROOT" diff --stat > "$output_dir/git-diff-stat.txt" || true
  git -C "$PROJECT_ROOT" diff --patch --unified=0 > "$output_dir/git-diff.patch" || true
}

run_profile_pass() {
  local profile="$1"
  local pass_name="$2"
  local output_dir="$3"

  mkdir -p "$output_dir/commands"
  (
    cd "$PROJECT_ROOT"
    export TARGET_ENV="$profile"
    unset HOST_PROFILE_FILE OLLAMA_HOST
    load_host_profile "$PROJECT_ROOT"

    printenv | sort | grep -E '^(TARGET_ENV|TARGET_HOST|BACKEND_HOST|BACKEND_PORT|PUBLIC_API_URL|DEPLOYMENT_ROLE|HOST_PROFILE_FILE|OLLAMA_HOST)=' \
      > "$output_dir/env.snapshot" || true

    printf '%s\n' "profile=$profile" "pass=$pass_name" "host_profile=${HOST_PROFILE_FILE:-unset}" > "$output_dir/pass.metadata"
    capture_repo_state "$output_dir"

    : > "$output_dir/commands-results.tsv"
    local command_index=1
    for command in "${COMMANDS[@]}"; do
      local slug
      slug="$(safe_name "$command")"
      local stdout_file="$output_dir/commands/${command_index}_${slug}.stdout"
      local stderr_file="$output_dir/commands/${command_index}_${slug}.stderr"
      local status_file="$output_dir/commands/${command_index}_${slug}.status"
      local command_status=0

      set +e
      eval "$command" >"$stdout_file" 2>"$stderr_file"
      command_status=$?
      set -e

      printf '%s\n' "$command_status" > "$status_file"
      printf '%s\t%s\t%s\n' "$command_index" "$command_status" "$command" >> "$output_dir/commands-results.tsv"
      printf '%s\n' "$command" > "$output_dir/commands/${command_index}_${slug}.cmd"
      command_index=$((command_index + 1))
    done

    capture_repo_state "$output_dir/final"
  )
}

compare_passes() {
  local profile_dir="$1"
  local comparison_dir="$profile_dir/comparison"
  mkdir -p "$comparison_dir"

  diff -u "$profile_dir/first-pass/git-status.txt" "$profile_dir/second-pass/git-status.txt" \
    > "$comparison_dir/git-status.diff" || true
  diff -u "$profile_dir/first-pass/git-diff-stat.txt" "$profile_dir/second-pass/git-diff-stat.txt" \
    > "$comparison_dir/git-diff-stat.diff" || true
  diff -u "$profile_dir/first-pass/final/git-status.txt" "$profile_dir/second-pass/final/git-status.txt" \
    > "$comparison_dir/final-git-status.diff" || true
  diff -u "$profile_dir/first-pass/final/git-diff-stat.txt" "$profile_dir/second-pass/final/git-diff-stat.txt" \
    > "$comparison_dir/final-git-diff-stat.diff" || true

  {
    printf 'profile=%s\n' "$(basename "$profile_dir")"
    printf 'status_diff=%s\n' "$comparison_dir/git-status.diff"
    printf 'diffstat_diff=%s\n' "$comparison_dir/git-diff-stat.diff"
    printf 'final_status_diff=%s\n' "$comparison_dir/final-git-status.diff"
    printf 'final_diffstat_diff=%s\n' "$comparison_dir/final-git-diff-stat.diff"
  } > "$comparison_dir/comparison-summary.txt"
}

open_bug_issue() {
  local profile="$1"
  local profile_dir="$2"

  if [ "$OPEN_ISSUE" != "true" ]; then
    return 0
  fi

  python3 - "$PROJECT_ROOT" "$profile" "$profile_dir" <<'PY'
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

project_root = Path(sys.argv[1])
profile = sys.argv[2]
profile_dir = Path(sys.argv[3])

comparison_dir = profile_dir / "comparison"
comparison_files = [
    comparison_dir / "git-status.diff",
    comparison_dir / "git-diff-stat.diff",
    comparison_dir / "final-git-status.diff",
    comparison_dir / "final-git-diff-stat.diff",
]

if not any(path.exists() and path.stat().st_size > 0 for path in comparison_files):
    raise SystemExit(0)


def read_text(path: Path, limit: int = 12000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:limit]


def credential_token() -> str:
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("GitHub token not found in credential helper output")


def repo_slug() -> tuple[str, str]:
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root,
    ).stdout.strip()
    match = re.search(r"github.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)", remote)
    if not match:
        raise RuntimeError(f"Cannot parse GitHub remote: {remote}")
    return match.group("owner"), match.group("repo")


owner, repo = repo_slug()
token = credential_token()

issue_body = "\n".join([
    f"Host profile regression detected for `{profile}`.",
    "",
    "Parent: #265",
    "",
    "Observed artifacts:",
    f"- First pass: {profile_dir / 'first-pass'}",
    f"- Second pass: {profile_dir / 'second-pass'}",
    f"- Comparison: {comparison_dir}",
    "",
    "Command summary:",
    read_text(profile_dir / 'commands-summary.txt'),
    "",
    "Comparison summary:",
    read_text(profile_dir / 'comparison-summary.txt'),
    "",
    "Diff excerpts:",
    read_text(comparison_dir / 'git-status.diff'),
    read_text(comparison_dir / 'git-diff-stat.diff'),
    read_text(comparison_dir / 'final-git-status.diff'),
    read_text(comparison_dir / 'final-git-diff-stat.diff'),
])

payload = json.dumps({
    "title": f"[bug][idempotency] {profile} host-profile rerun regression",
    "body": issue_body,
    "labels": ["bug", "idempotency", "needs-evidence"],
}).encode("utf-8")

request = urllib.request.Request(
    f"https://api.github.com/repos/{owner}/{repo}/issues",
    data=payload,
    headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "host-profile-matrix",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        created = json.loads(response.read().decode("utf-8"))
    print(created.get("html_url", ""))
except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
    draft = profile_dir / "draft_issue.json"
    draft.write_text(json.dumps({"error": detail, "issue_body": issue_body}, indent=2) + "\n", encoding="utf-8")
    print(f"Failed to open issue for {profile}: {detail}", file=sys.stderr)
    print(f"Draft saved to {draft}", file=sys.stderr)
    raise SystemExit(1)
PY
}

summary_file="$RUN_ROOT/summary.txt"
{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'artifact_root=%s\n' "$ARTIFACT_ROOT"
  printf 'profiles=%s\n' "${PROFILES[*]}"
  printf 'commands=%s\n' "${COMMANDS[*]}"
} > "$summary_file"

overall_status=0

for profile in "${PROFILES[@]}"; do
  profile_dir="$RUN_ROOT/$profile"
  mkdir -p "$profile_dir"

  first_pass_dir="$profile_dir/first-pass"
  second_pass_dir="$profile_dir/second-pass"

  run_profile_pass "$profile" "first" "$first_pass_dir"
  run_profile_pass "$profile" "second" "$second_pass_dir"
  compare_passes "$profile_dir"

  {
    printf 'profile=%s\n' "$profile"
    printf 'first_pass=%s\n' "$first_pass_dir"
    printf 'second_pass=%s\n' "$second_pass_dir"
    printf 'status_diff=%s\n' "$profile_dir/comparison/git-status.diff"
    printf 'diffstat_diff=%s\n' "$profile_dir/comparison/git-diff-stat.diff"
    printf 'final_status_diff=%s\n' "$profile_dir/comparison/final-git-status.diff"
    printf 'final_diffstat_diff=%s\n' "$profile_dir/comparison/final-git-diff-stat.diff"
    printf '\n[first-pass command results]\n'
    cat "$first_pass_dir/commands-results.tsv"
    printf '\n[second-pass command results]\n'
    cat "$second_pass_dir/commands-results.tsv"
  } > "$profile_dir/commands-summary.txt"

  if [ -s "$profile_dir/comparison/git-status.diff" ] \
    || [ -s "$profile_dir/comparison/git-diff-stat.diff" ] \
    || [ -s "$profile_dir/comparison/final-git-status.diff" ] \
    || [ -s "$profile_dir/comparison/final-git-diff-stat.diff" ]; then
    printf 'profile=%s status=regression\n' "$profile" >> "$summary_file"
    open_bug_issue "$profile" "$profile_dir" || overall_status=1
  else
    printf 'profile=%s status=stable\n' "$profile" >> "$summary_file"
  fi
done

printf '\nMatrix complete. Artifacts: %s\n' "$RUN_ROOT"
cat "$summary_file"

exit "$overall_status"
