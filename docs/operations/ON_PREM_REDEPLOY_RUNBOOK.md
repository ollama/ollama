# On-Prem Redeploy Runbook

This runbook is the canonical redeploy procedure for target-server-local environments.

Use this with [On-Prem Deployment Model](ON_PREM_DEPLOYMENT_MODEL.md) and [On-Prem Execution Index](ON_PREM_EXECUTION_INDEX.md).

## Scope

- Redeploys for `development` and `production` host profiles.
- Repo-local execution only.
- Immutable inventory-driven configuration.

## Preconditions

- Repository checkout is clean before deploy actions.
- `config/hosts/development.env` and `config/hosts/production.env` are the source of truth.
- Docker and compose are available on the target node.

## Standard Redeploy Flow

1. Select host profile and load it:

```bash
export TARGET_ENV=production
source ./scripts/host-profile.sh
load_host_profile "$(pwd)"
```

2. Run prerequisites and environment checks:

```bash
./scripts/preflight.sh
./scripts/onboard.sh --dry-run --yes
```

3. Redeploy with repo-local helper:

```bash
./scripts/deploy.sh
```

4. Verify runtime health and model endpoint:

```bash
curl -fsS "${PUBLIC_API_URL}/health"
curl -fsS "${OLLAMA_HOST}/api/tags" | head -20
```

5. Validate deterministic rerun behavior:

```bash
./scripts/host-profile-matrix.sh --no-open-issue
```

## Evidence and Immutability Rules

- Keep host values in `config/hosts/*.env`; do not patch IPs into scripts.
- Keep redeploy evidence ephemeral under `/tmp/ollama-host-profile-idempotency/<run-id>/` unless explicitly promoted.
- If a redeploy requires configuration change, commit inventory/script updates first, then redeploy.

## Rollback Guidance

- If health checks fail, inspect compose service status and logs first:

```bash
docker compose -f docker/docker-compose.elite.yml ps
docker compose -f docker/docker-compose.elite.yml logs --tail=200 ollama-api
```

- Re-run `./scripts/preflight.sh` and verify host profile values before retrying deploy.
- If failure is profile-specific, capture matrix artifacts and open a bug linked to rerun evidence.

## Related Documentation

- [On-Prem Execution Index](ON_PREM_EXECUTION_INDEX.md)
- [On-Prem Deployment Model](ON_PREM_DEPLOYMENT_MODEL.md)
- [Host Profile Idempotency Matrix](../reports/HOST_PROFILE_IDEMPOTENCY.md)
- [Repository Rules](../repo-rules/README.md)
