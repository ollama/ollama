# Vision Regression Bisect Plan

## Objective
Identify the first container image (or commit) where the vision capability stops working, minimizing test iterations.

## Dimensions to Bisect
1. **Container Image Digest** (preferred if many historical images cached).
2. **Git Commits** affecting vision pipeline (when building locally).

## Prerequisites
- Reproduction confirmed (see `01_reproduction_guide.md`).
- List of available image digests (or commits) sorted chronologically.

## Data Structures
Maintain a CSV (or Markdown table):
```
Digest,Created,Commit,Model,VisionCapability(OK/FAIL),Notes
```

## Image-Level Bisect Steps
1. Gather list:
   ```powershell
   docker images --digests --format "{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}\t{{.Digest}}" | sort > images.txt
   ```
2. Extract candidate set (filter same repository/tag lineage).
3. Select mid-point digest.
4. Run reproduction test (pull model, check capabilities, perform vision prompt).
5. Mark result OK/FAIL.
6. Narrow interval: choose mid in failing/working halves.
7. Repeat until adjacent pair (last good / first bad) isolated.

## Commit-Level Bisect (If Building Locally)
1. Identify commit range using last good / first bad timestamps from image metadata or `ollama version` output inside containers.
2. Use `git bisect`:
   ```bash
   git bisect start
   git bisect bad <first_bad_commit>
   git bisect good <last_good_commit>
   ```
3. For each checkout:
   - Build or run `make` / container build script.
   - Execute minimal test: capability presence + one image prompt.
   - Mark with `git bisect good` or `git bisect bad`.
4. When bisect ends, record offending commit hash.

## Minimal Test Script (Pseudo)
```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL="$1"
curl -s http://localhost:11434/api/pull -d '{"name":"'"$MODEL"'"}' >/dev/null
SHOW=$(curl -s http://localhost:11434/api/show -d '{"name":"'"$MODEL"'"}')
echo "$SHOW" | grep -q 'vision' || exit 1  # fail if vision absent
# Optionally send image request and grep for expected token
```
Exit code 0 = good, non-zero = bad, can be wired into `git bisect run`.

## Prioritizing Commits to Review
Focus on diffs touching:
- `server/images.go`
- `server/create.go`
- `server/routes.go`
- `fs/gguf/` and `fs/ggml/`
- `convert/`
- `model/models/*/model_vision.go`

## Post-Bisect Actions
1. Extract `git show <bad_commit>` summary.
2. Highlight removed / modified checks around `vision.block_count` or projector handling.
3. Draft patch or open issue referencing precise code lines.

## Time Optimization Tips
- Cache models on host volume to avoid re-pulling each iteration (mount same volume into test containers).
- Disable unrelated features (tools, thinking) to reduce noise.
- Fix environment variables to stable values (temperature, parallel).

## Completion Criteria
- Single commit (or minimal diff) identified.
- Reproduced failure on that commit; success on its parent.
- Collected artifacts (logs, capability JSON) attached to report.
