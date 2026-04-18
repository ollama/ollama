# Host Profile Idempotency Matrix

This report defines the deterministic rerun checks for the two target-server roles used by the on-prem execution model.

## Matrix

| Profile | Inventory | Role |
| --- | --- | --- |
| `development` | `config/hosts/development.env` | `code-server` target node |
| `production` | `config/hosts/production.env` | `eiq-bare-metal` target host |

## Commands

The matrix runner at [scripts/host-profile-matrix.sh](../../scripts/host-profile-matrix.sh) executes the same command set twice per profile.

Default command set:

```bash
./scripts/preflight.sh
./scripts/onboard.sh --dry-run --yes
```

## Evidence Layout

Artifacts are written under an ephemeral directory by default:

```text
/tmp/ollama-host-profile-idempotency/<run-id>/<profile>/
```

Per profile, the runner captures:

- `first-pass/` and `second-pass/` command output
- `git-status.txt`
- `git-diff-stat.txt`
- `git-diff.patch`
- `comparison/` diffs between first and second pass

## Failure Taxonomy

Use the following categories when a rerun is not stable:

1. `profile-resolution-failure` - the inventory file could not be loaded or produced incomplete environment values.
2. `command-failure` - a command exits non-zero on the first pass or second pass.
3. `mutation-regression` - the second pass creates a new repo diff or changes the first-pass diff snapshot.
4. `environment-drift` - the host settings differ from the checked-in inventory, but the repo remains unchanged.
5. `external-dependency-failure` - Docker, Git, or local services are unavailable on the target host.

## Triage Rubric

- If the second pass introduces new repo mutations, open a linked bug issue and attach the comparison artifacts.
- If a command fails but the repo state remains stable, treat it as an execution or environment issue and record the failing command and exit code.
- If the profile cannot load, fix the inventory or loader before retrying the matrix.
- If the command output changes but the repo diff remains empty, review for nondeterministic logging only.

## Auto-Issue Behavior

When the matrix runner is executed with `OPEN_ISSUE=true` and a regression is detected, it opens a bug issue with the failing profile, the comparison artifacts, and a link back to issue #265.

## Expected Outcome

The second pass should produce zero unintended repository mutations for both profiles.
