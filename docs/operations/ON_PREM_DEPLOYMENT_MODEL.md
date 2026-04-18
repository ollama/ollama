# On-Prem Deployment Model

This is the canonical target-server-local execution model for the repository.

Use [On-Prem Execution Index](ON_PREM_EXECUTION_INDEX.md) as the shared navigation entry point. This page is the SSOT for host inventories, immutable configuration, and deterministic rerun behavior.

## Scope

- Covers development-node and production-host execution on checked-in host inventories.
- Defines how repo-local scripts, compose files, and validation runs should behave.
- Keeps host-specific values out of scripts and docs that can derive them from inventory.

## Principles

- Immutable configuration: host values live in `config/hosts/*.env`.
- Repo-relative execution: scripts resolve paths from the repository root.
- Ephemeral evidence: validation output goes to temporary or declared artifact locations.
- No hard-coded IPs or environment-specific literals in canonical docs or scripts.

## Host Profile Contract

- `TARGET_ENV` selects `config/hosts/<env>.env`.
- `HOST_PROFILE_FILE` can override the inventory path when an explicit file is needed.
- `TARGET_HOST`, `BACKEND_HOST`, `BACKEND_PORT`, `PUBLIC_API_URL`, and `DEPLOYMENT_ROLE` come from inventory.
- `scripts/host-profile.sh` loads the selected profile and exports `OLLAMA_HOST` from `BACKEND_HOST` when needed.

## Execution Model

- Use `scripts/preflight.sh` to validate prerequisites before a run.
- Use `scripts/onboard.sh`, `scripts/bootstrap.sh`, `scripts/deploy.sh`, `scripts/local-start.sh`, `scripts/local-stop.sh`, `scripts/local-dev-automation.sh`, and `scripts/restore-postgres.sh` as repo-relative helpers.
- Keep compose manifests under `docker/` and keep runtime state in declared volumes or other IaC-managed storage.
- Prefer host profiles over ad hoc environment variables so reruns stay reproducible.

## Deterministic Validation

- Run `./scripts/host-profile-matrix.sh` to capture first-pass and second-pass evidence.
- Use `--no-open-issue` when you only want local evidence capture.
- Artifacts are written under `/tmp/ollama-host-profile-idempotency/<run-id>/` by default.
- Any second-pass repository mutation is a regression and should be treated as a rerun failure.

## Related Documentation

- [On-Prem Execution Index](ON_PREM_EXECUTION_INDEX.md)
- [On-Prem Redeploy Runbook](ON_PREM_REDEPLOY_RUNBOOK.md)
- [Shared Documentation Navigation](../shared/README.md)
- [Documentation SSOT](../meta/README.md)
- [Repository Rules](../repo-rules/README.md)
- [Host Profile Idempotency Matrix](../reports/HOST_PROFILE_IDEMPOTENCY.md)
- [Deployment Guide](../DEPLOYMENT.md)
- [Quick Reference](../QUICK_REFERENCE.md)
