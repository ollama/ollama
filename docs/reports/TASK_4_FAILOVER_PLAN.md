# Task 4: Automated Failover — Plan Report

## Scope

Design and implement multi-region active–passive failover for Ollama API via GCP Global HTTP(S) Load Balancer. Primary region serves traffic; secondary is passive with `failover=true`. Health checks on `/api/v1/health` port `8000`.

## Deliverables

- Terraform (`docker/terraform/gcp_failover.tf`) with variables and examples
- Documentation (`docs/AUTOMATED_FAILOVER_IMPLEMENTATION.md`)
- Quick Reference (`docs/TASK_4_QUICK_REFERENCE.md`)
- tfvars example (`docker/terraform/failover.auto.tfvars.example`)

## Compliance

- Naming: `{environment}-{application}-{component}`
- Labels: environment, team, application, component, cost-center, managed-by, git_repo, lifecycle_status
- Zero Trust: IAP/Secret Manager/Workload Identity; TLS 1.3; mutual TLS internal
- Single external entry point: `https://elevatediq.ai/ollama`

## Rollout Steps

1. Prepare MIGs in primary and secondary regions
2. Fill tfvars example with project and MIG self-links
3. Apply Terraform
4. Verify health checks and logging
5. Simulate failure of primary; confirm failover < 30s

## Risks & Mitigations

- Misconfigured health checks → Validate path and port
- LB duplication → Attach backend to existing URL map if LB exists
- Cascading failures → Enable outlier detection and circuit breakers on backend service

## Next

- Integrate with existing LB certs and URL map if present
- Add integration tests for health endpoint behavior
- Finalize runbook and backout steps
