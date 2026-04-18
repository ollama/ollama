# Task 4: Automated Failover — Quick Reference

## TL;DR

- Single external endpoint: `https://elevatediq.ai/ollama`
- Primary region serves traffic; secondary region is passive (failover)
- Health check path: `/api/v1/health` on port `8000`
- Enable by setting `enable_failover=true` and providing MIG self-links

## Files

- Terraform: `docker/terraform/gcp_failover.tf`
- Implementation: `docs/AUTOMATED_FAILOVER_IMPLEMENTATION.md`

## Variables to Set

- `project_id`
- `environment` (production|staging|development|sandbox)
- `primary_region`
- `secondary_region`
- `primary_instance_group` (self-link)
- `secondary_instance_group` (self-link)
- `enable_failover` (true)

## Apply (Terraform)

```bash
# Configure variables via tfvars or environment
terraform init
terraform plan -var enable_failover=true \
  -var project_id=$PROJECT \
  -var environment=production \
  -var primary_region=us-central1 \
  -var secondary_region=us-east1 \
  -var primary_instance_group=$PRIMARY_IG \
  -var secondary_instance_group=$SECONDARY_IG
terraform apply -auto-approve \
  -var enable_failover=true \
  -var project_id=$PROJECT \
  -var environment=production \
  -var primary_region=us-central1 \
  -var secondary_region=us-east1 \
  -var primary_instance_group=$PRIMARY_IG \
  -var secondary_instance_group=$SECONDARY_IG
```

## Verify

- LB health checks green for primary; secondary ready
- Simulate failure: stop primary MIG → traffic served by secondary within <30s
- Firewall blocks internal ports (8000, 5432, 6379, 11434)
- CORS only allows `https://elevatediq.ai`

## Rollback

```bash
terraform apply -auto-approve -var enable_failover=false
```
