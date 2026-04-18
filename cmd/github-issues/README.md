# GitHub Issues Checker

Command-line tool for managing GitHub issues via the Ollama orchestration pipeline.

## Authentication — GSM Credential Helper (Canonical)

This workspace uses `git-credential-gsm` as the single source of truth for the GitHub PAT:

| Setting | Value |
|---------|-------|
| Binary | `/usr/local/bin/git-credential-gsm` |
| Git config | `~/.gitconfig → credential.helper = gsm` |
| GCP Project | `gcp-eiq` (env: `GSM_PROJECT`) |
| Secret Name | `prod-github-token` (env: `GSM_SECRET_NAME`) |
| Repository | `kushin77/ollama` |

**Prerequisite:** `gcloud` must be authenticated before any GitHub operation:
```bash
gcloud auth login
# OR service account:
gcloud auth activate-service-account --key-file=/path/to/sa.json
```

**To run the orchestrator with GSM-backed auth:**
```bash
export OLLAMA_GSM_ENABLED=true
export GSM_PROJECT=gcp-eiq
export GSM_SECRET_NAME=prod-github-token
python3 cmd/github-issues/orchestrator_enhanced.py \
    --repo kushin77/ollama --all-severities --execute \
    --output .github/orchestrator_report_kushin77_ollama_live.json
```

The orchestrator resolves auth in this priority order:
1. `--token` CLI arg
2. `--token-file` file contents
3. `git credential fill` (→ `git-credential-gsm` → `prod-github-token` from `gcp-eiq`)
4. `OLLAMA_GSM_ENABLED` path (same gcloud command)
5. `OLLAMA_GITHUB_TOKEN` / `GITHUB_TOKEN` / `GH_TOKEN` env vars

## Usage

View open issues for ollama/ollama:

```bash
go run cmd/github-issues/main.go
```

### Options

```
  -owner string
        Repository owner (default "ollama")
  -repo string
        Repository name (default "ollama")
  -state string
        Issue state: open, closed, all (default "open")
  -limit int
        Number of issues to display (default 20)
  -sort string
        Sort by: created, updated, comments (default "created")
  -order string
        Order: asc, desc (default "desc")
  -labels string
        Filter by labels (comma-separated)
```

### Examples

View the 10 most recently created open issues:

```bash
go run cmd/github-issues/main.go -limit 10 -sort created
```

View all closed issues sorted by updates:

```bash
go run cmd/github-issues/main.go -state closed -sort updated
```

View issues with specific labels:

```bash
go run cmd/github-issues/main.go -labels "bug,help wanted"
```

View issues from a different repository:

```bash
go run cmd/github-issues/main.go -owner golang -repo go
```

Fetch every page and write JSON to a file:

```bash
go run cmd/github-issues/main.go -all-pages -output json -out-file issues.json
```

Watch for changes every 30 seconds and print only deltas:

```bash
go run cmd/github-issues/main.go -watch 30s -diff -output table
```

## Output Format

The tool displays issues in a table format:

```
     #        TITLE                      STATE      AUTHOR         UPDATED
   ---           ---                        ---        ---            ---
  1234  Fix memory leak in model loader    open    john-doe    2026-04-17
  1233  Add support for new model format   open    jane-smith  2026-04-16
```

Each issue shows:
- Issue number
- Title (truncated to 45 characters)
- State (open/closed)
- Author username
- Last updated date

## Requirements

- Go 1.18+
- Valid GitHub token (includes scoped access to repositories)
- Internet connection

## Implementation

The CLI is self-contained in `cmd/github-issues/main.go` and uses the GitHub REST API directly via `net/http`.

Implementation guarantees:

- Default 30 second HTTP timeout to avoid hanging requests
- Tunable client options via `NewClientWithOptions`
- `NewClientWithURL` for tests and alternate API endpoints
- Pagination support for full repository scans
- Table, JSON, and CSV output modes
- Watch mode with diff-only output for dashboards and automation

## Troubleshooting

### "OLLAMA_GITHUB_TOKEN not set"

Set your GitHub token:

```bash
export OLLAMA_GITHUB_TOKEN=ghp_xxxxx
```

### "GitHub API error: status 401"

Your token is invalid or expired. Generate a new one at: https://github.com/settings/tokens

### "GitHub API error: status 403"

You may have hit rate limiting. Authenticated requests have higher limits (5000 per hour).

### "No issues found"

The repository may not have issues in the requested state/filter. Try adjusting filters or viewing all states:

```bash
go run cmd/github-issues/main.go -state all
```

## Integration with Ollama

This tool demonstrates how to use the Ollama GitHub integration in practice. The same GitHub client can be used in Ollama applications to:

- Access GitHub repositories programmatically
- Retrieve issue information
- Integrate with GitHub workflows
- Manage repository operations

See `internal/secrets/README.md` for more information about the GitHub client.

## Orchestrator (IaC + Immutable)

Run the full triage/execution orchestrator against a specific repository:

```bash
python3 cmd/github-issues/orchestrator_enhanced.py \
      --repo kushin77/ollama \
      --all-severities \
      --max-issues 50 \
      --execute \
      --output .github/orchestrator_report_kushin77_ollama.json
```

Behavior notes:

- The orchestrator now runs strictly against `--repo` by default.
- If the target repo is inaccessible, it skips execution rather than using demo/fake issues.
- Fallback to another repo is opt-in only:

```bash
python3 cmd/github-issues/orchestrator_enhanced.py --repo kushin77/ollama --allow-fallback
```

- Execution is idempotent for summary comments using a deterministic marker per issue.
- Optional auto-close policy is label-driven (default labels: `duplicate,invalid,wontfix`):

```bash
python3 cmd/github-issues/orchestrator_enhanced.py \
      --repo kushin77/ollama \
      --execute \
      --auto-close-labels duplicate,invalid,wontfix
```

- Immutable operation records are written to `.github/issue_ops_ledger.jsonl`.

## Autonomous Agent Mode (No Human-in-the-Loop)

The repository now includes a fully declarative and idempotent autonomous mode:

- IaC config: `.github/issue-orchestrator.iac.json`
- Idempotent runner: `scripts/run_issue_orchestrator_idempotent.sh`
- Scheduled workflow: `.github/workflows/issue-orchestrator-autonomous.yml`

Run locally (same config used by CI):

```bash
chmod +x scripts/run_issue_orchestrator_idempotent.sh
scripts/run_issue_orchestrator_idempotent.sh .github/issue-orchestrator.iac.json
```

Design guarantees:

- **IaC-driven**: repository, auth mode, execution policy, output paths, and lock path are all declared in one JSON config.
- **Immutable**: append-only issue operations ledger and per-issue state snapshots are persisted under `.github/`.
- **Idempotent**: lock-file + `flock` prevents concurrent duplicate runs, and orchestrator comment markers avoid duplicate comment spam.
- **Global/Autonomous**: workflow runs on schedule and `workflow_dispatch`, then commits generated state artifacts back to git.

Operational principle:

"If it is not committed, it does not exist." The autonomous workflow commits state artifacts produced by each run.
