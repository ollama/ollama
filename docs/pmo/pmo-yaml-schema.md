# PMO pmo.yaml Schema Specification

**Version**: 2.0.0
**Last Updated**: January 26, 2026

---

## Overview

The `pmo.yaml` file is the cornerstone of PMO governance. Every repository **MUST** contain this file at the root with **24 mandatory labels** across 6 categories.

---

## Schema Definition

### 1. Organizational Labels (4 required)

#### environment

- **Type**: String (enum)
- **Required**: Yes
- **Values**: `production`, `staging`, `development`, `sandbox`
- **Description**: Deployment environment
- **Example**: `production`

#### cost_center

- **Type**: String
- **Required**: Yes
- **Values**: `engineering`, `ai-ml`, `data`, `infra`, `product`
- **Description**: Cost center for billing attribution
- **Example**: `engineering`

#### team

- **Type**: String
- **Required**: Yes
- **Description**: Owning team name
- **Example**: `platform-engineering`

#### managed_by

- **Type**: String (enum)
- **Required**: Yes
- **Values**: `terraform`, `manual`, `cloudformation`, `pulumi`
- **Description**: Infrastructure management method
- **Example**: `terraform`

---

### 2. Lifecycle Labels (5 required)

#### created_by

- **Type**: Email
- **Required**: Yes
- **Format**: `email@domain.com`
- **Description**: Creator's email address
- **Example**: `akushnir@elevatediq.ai`

#### created_date

- **Type**: Date
- **Required**: Yes
- **Format**: `YYYY-MM-DD`
- **Description**: Repository creation date
- **Example**: `2026-01-14`

#### lifecycle_state

- **Type**: String (enum)
- **Required**: Yes
- **Values**: `active`, `maintenance`, `sunset`, `archived`
- **Description**: Current lifecycle state
- **Example**: `active`

#### teardown_date

- **Type**: Date or "none"
- **Required**: Yes
- **Format**: `YYYY-MM-DD` or `none`
- **Description**: Scheduled decommission date
- **Example**: `none`

#### retention_days

- **Type**: Integer
- **Required**: Yes
- **Values**: `365`, `3650`, `7300`
- **Description**: Data retention period (days)
- **Example**: `3650` (10 years)

---

### 3. Business Labels (4 required)

#### product

- **Type**: String
- **Required**: Yes
- **Description**: Product or service name
- **Example**: `ollama`

#### component

- **Type**: String
- **Required**: Yes
- **Values**: `api`, `database`, `frontend`, `auth`, `monitoring`, `cache`, `queue`
- **Description**: Technical component type
- **Example**: `api-server`

#### tier

- **Type**: String (enum)
- **Required**: Yes
- **Values**: `critical`, `high`, `medium`, `low`
- **Description**: Business criticality
- **Example**: `high`

#### compliance

- **Type**: String (enum)
- **Required**: Yes (can be "none")
- **Values**: `sox`, `hipaa`, `pci`, `gdpr`, `none`
- **Description**: Regulatory compliance requirements
- **Example**: `none`

---

### 4. Technical Labels (4 required)

#### version

- **Type**: Semantic Version
- **Required**: Yes
- **Format**: `X.Y.Z`
- **Description**: Current application version
- **Example**: `0.1.0`

#### stack

- **Type**: String
- **Required**: Yes
- **Format**: `language-version-framework`
- **Description**: Technology stack
- **Example**: `python-3.11-fastapi`

#### backup_strategy

- **Type**: String (enum)
- **Required**: Yes
- **Values**: `daily`, `weekly`, `monthly`, `none`
- **Description**: Backup frequency
- **Example**: `daily`

#### monitoring_enabled

- **Type**: Boolean
- **Required**: Yes
- **Values**: `true`, `false`
- **Description**: Whether monitoring is configured
- **Example**: `true`

---

### 5. Financial Labels (4 required)

#### budget_owner

- **Type**: Email
- **Required**: Yes
- **Format**: `email@domain.com`
- **Description**: Budget owner's email
- **Example**: `akushnir@elevatediq.ai`

#### project_code

- **Type**: String
- **Required**: Yes
- **Format**: `PROJECT-YYYY-NNN`
- **Description**: Unique project identifier
- **Example**: `OLLAMA-2026-001`

#### monthly_budget_usd

- **Type**: Integer
- **Required**: Yes
- **Description**: Estimated monthly cost (USD)
- **Example**: `500`

#### chargeback_unit

- **Type**: String
- **Required**: Yes
- **Description**: Team/unit for cost chargeback
- **Example**: `ai-division`

---

### 6. Git Labels (3 required)

#### git_repository

- **Type**: URL
- **Required**: Yes
- **Format**: `github.com/owner/repo`
- **Description**: Git repository URL
- **Example**: `github.com/kushin77/ollama`

#### git_branch

- **Type**: String
- **Required**: Yes
- **Description**: Default branch name
- **Example**: `main`

#### auto_delete

- **Type**: Boolean
- **Required**: Yes
- **Values**: `true`, `false`
- **Description**: Auto-delete on lifecycle sunset
- **Example**: `false`

---

## Complete Example

```yaml
# PMO Metadata
# Repository: ollama

# Organizational (4)
environment: "production"
cost_center: "engineering"
team: "platform-engineering"
managed_by: "terraform"

# Lifecycle (5)
created_by: "akushnir@elevatediq.ai"
created_date: "2026-01-14"
lifecycle_state: "active"
teardown_date: "none"
retention_days: "3650"

# Business (4)
product: "ollama"
component: "api-server"
tier: "high"
compliance: "none"

# Technical (4)
version: "0.1.0"
stack: "python-3.11-fastapi"
backup_strategy: "daily"
monitoring_enabled: "true"

# Financial (4)
budget_owner: "akushnir@elevatediq.ai"
project_code: "OLLAMA-2026-001"
monthly_budget_usd: "500"
chargeback_unit: "ai-division"

# Git (3)
git_repository: "github.com/kushin77/ollama"
git_branch: "main"
auto_delete: "false"
```

---

## Validation

Validate pmo.yaml:

```bash
./scripts/pmo/validate-pmo-metadata.sh
```

Expected output:

```
✅ All 24 mandatory labels populated
✅ No validation errors
```

---

## Generation

Auto-generate pmo.yaml:

```bash
./scripts/pmo/generate-pmo-yaml.sh
```

AI-powered generation:

```bash
ollama-pmo onboard my-repo --auto
```

---

## Enforcement

- **Pre-commit hooks**: Validate pmo.yaml before commit
- **GitHub Actions**: Run validation on PR
- **Branch protection**: Require validation to pass
- **Daily checks**: Automated compliance monitoring

---

## References

- Enforcement script: `/scripts/pmo/enforce-pmo-governance.sh`
- Validation script: `/scripts/pmo/validate-pmo-metadata.sh`
- GitHub workflow: `/.github/workflows/pmo-validation.yml`
- Governance mandate: `/docs/pmo/governance/PMO_ENFORCEMENT_MANDATE.md`
