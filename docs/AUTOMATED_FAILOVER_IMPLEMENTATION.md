# Automated Failover Implementation (Task 4)

This document describes the design and implementation of a multi-region, active–passive failover for the Ollama API using a global HTTP(S) Load Balancer on Google Cloud.

## Objectives

- Achieve 99.99% uptime with sub-30s failover time
- Maintain a single external entry point: `https://elevatediq.ai/ollama`
- Enforce GCP Landing Zone compliance (labels, naming, zero trust, audit)
- Restrict direct access to internal services; all traffic must go through GCP LB

## Architecture Overview

- Global HTTP(S) Load Balancer
  - Primary backend: regional MIG in `PRIMARY_REGION`
  - Secondary backend (failover): regional MIG in `SECONDARY_REGION`
  - Health check: HTTP to `/api/v1/health` on port `8000`
  - TLS termination at LB; mutual TLS internally
- Docker network (internal only)
  - FastAPI: `0.0.0.0:8000`
  - PostgreSQL: `postgres:5432`
  - Redis: `redis:6379`
  - Ollama: `ollama:11434`

## Naming & Labels

Resources follow `{environment}-{application}-{component}` naming, e.g., `prod-ollama-api` and include mandatory labels:

- `environment`, `team`, `application`, `component`, `cost-center`, `managed-by`, `git_repo`, `lifecycle_status`

## Terraform IaC

A new Terraform file `docker/terraform/gcp_failover.tf` configures:

- `google_compute_health_check` (HTTP) targeting `/api/v1/health`
- `google_compute_backend_service` with two backends:
  - primary: `failover=false` (active)
  - secondary: `failover=true` (passive)
- `google_compute_url_map`, `google_compute_target_https_proxy`, `google_compute_global_forwarding_rule`

Enable via `var.enable_failover=true` and provide self-links for primary and secondary MIGs.

## Health Check Contract

- Endpoint: `GET /api/v1/health`
- Response: `{ "status": "healthy" }`
- Latency target: < 200ms p95 (excluding inference)

## Failover Policy

- If primary backend health drops below threshold (≥3 consecutive failures), traffic is routed to secondary backend
- Connection draining for graceful transition (`10s`)
- Optional outlier detection & circuit breakers to protect from cascading failures

## Security & Compliance

- External access through LB only; internal ports blocked by firewall
- Zero trust authentication (IAP, Secret Manager, Workload Identity)
- TLS 1.3 enforced at LB; mutual TLS internally
- Audit logs enabled through LB log_config

## Rollout Procedure

1. Prepare MIGs in both regions (primary and secondary)
2. Set `enable_failover=true` and provide MIG self-links
3. Apply Terraform
4. Verify health checks and LB configuration
5. Simulate failure (e.g., stop primary MIG) and confirm failover

## Verification

- Health check status: green for both backends (primary active)
- Failover test: stop primary -> secondary serves within <30s
- Firewall: external access to internal ports blocked
- CORS: only `https://elevatediq.ai` origin allowed

## Observability

- LB logging enabled (`log_config.enable=true`)
- Prometheus metrics for health and request routing (existing stack)
- Structured logging via `structlog`

## Backout Plan

- Set `enable_failover=false` to detach failover resources
- Restore primary backend health
- Reapply Terraform

## Notes

- SSL certificates are assumed managed via Certificate Manager; attach to HTTPS proxy
- If an existing LB is configured, attach the backend service to the existing URL map rather than creating new proxies/rules
