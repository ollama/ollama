# Landing Zone Onboarding: Quick Action Guide
**Date**: January 19, 2026
**Status**: 3 Critical Items to Complete
**Deadline**: February 15, 2026 (4 weeks)

---

## 🚨 CRITICAL: 3 Items Blocking Production

### Item #1: Endpoint Registration in Domain Registry
**Priority**: CRITICAL
**Timeline**: 2 weeks
**Owner**: Infrastructure Team

#### What to Do
1. **Create Terraform configuration** for Ollama in Landing Zone domain registry
2. **Configure Cloud Armor** security policy
3. **Set up health check** endpoint
4. **Submit PR** to `gcp-landing-zone` repository

#### PR Destination
```
Repository: github.com/kushin77/GCP-landing-zone
Directory: terraform/modules/networking/domain-registry/
File: ollama-domain-entry.tf (new file)
```

#### Minimum PR Template
```hcl
# terraform/modules/networking/domain-registry/ollama-domain-entry.tf

domain_entries = {
  "ollama" = {
    # Basic Configuration
    domain             = "elevatediq.ai"
    subdomains         = ["ollama"]

    # TLS & Security
    tls_enabled        = true
    min_tls_version    = "1.3"
    oauth_protected    = false  # Machine-to-machine API

    # Backend Service
    backend_service    = "ollama-api-backend"
    health_check_path  = "/api/v1/health"
    timeout_sec        = 30

    # Security Policy
    cloud_armor_policy = "global-armor"

    # Features
    enable_cdn         = true
    enable_logging     = true

    # Logging
    audit_log_retention = 2555  # 7 years
  }
}
```

#### Testing After Merge
```bash
# Once PR is merged and Terraform applied:

# Test endpoint accessibility
curl -H "Authorization: Bearer sk-test-key" \
     https://elevatediq.ai/ollama/api/v1/health

# Check health status
gcloud compute backend-services get-health ollama-api-backend --global

# Load test (verify rate limiting)
for i in {1..150}; do curl https://elevatediq.ai/ollama/api/v1/health; done
# Should see some 429 responses (rate limited)
```

**Success Criteria**:
- ✅ PR merged to Landing Zone
- ✅ Endpoint accessible via https://elevatediq.ai/ollama
- ✅ Health checks passing
- ✅ Rate limiting active

---

### Item #2: 7-Year Audit Logging
**Priority**: CRITICAL
**Timeline**: 2 weeks
**Owner**: Infrastructure Team

#### What to Do
1. **Add Google Cloud Logging** integration to Python code
2. **Configure Cloud Logging** with 7-year retention
3. **Set up GCS bucket** for audit logs
4. **Test** log collection and query

#### Code Change Required
**File**: `ollama/config.py` (add this section)

```python
import google.cloud.logging
import logging
from pythonjsonlogger import jsonlogger

def setup_cloud_logging() -> None:
    """Configure Cloud Logging for audit trail."""

    # Initialize Cloud Logging client
    client = google.cloud.logging.Client()
    client.setup_logging()

    # Create audit logger
    audit_logger = logging.getLogger("ollama.audit")

    # Configure JSON format
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)

    return audit_logger

# In main.py
if ENVIRONMENT == "production":
    audit_logger = setup_cloud_logging()
```

#### Environment Variable Changes
**File**: `config/production.yaml` (add)

```yaml
# Cloud Logging Configuration
ENABLE_CLOUD_LOGGING: true
LOG_LEVEL: INFO
AUDIT_LOG_LEVEL: DEBUG
JSON_LOGGING_FORMAT: true
GCP_LOG_PROJECT_ID: gcp-landing-zone
GCP_LOG_SINK_NAME: ollama-audit-sink
GCP_AUDIT_RETENTION_DAYS: 2555  # 7 years
```

#### Terraform Configuration
**File**: `docker/terraform/logging.tf` (create new file)

```hcl
# Cloud Logging Sink
resource "google_logging_project_sink" "ollama" {
  name        = "ollama-audit-sink"
  destination = "storage.googleapis.com/${google_storage_bucket.audit_logs.name}"

  filter = "jsonPayload.component='ollama.audit'"

  unique_writer_identity = true
}

# GCS Bucket for 7-Year Retention
resource "google_storage_bucket" "audit_logs" {
  name          = "ollama-audit-logs-${var.environment}"
  location      = var.gcp_region
  force_destroy = false

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 2555  # 7 years
    }
  }
}
```

#### Verification
```bash
# 1. Deploy code and Terraform
cd docker/terraform
terraform plan
terraform apply

# 2. Generate test log entry
curl -X POST https://elevatediq.ai/ollama/api/v1/generate \
     -H "Authorization: Bearer sk-test-key" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "test"}'

# 3. Check Cloud Logging
gcloud logging read \
  "jsonPayload.component='ollama.audit'" \
  --limit 5 --format=json

# 4. Verify GCS bucket
gsutil ls gs://ollama-audit-logs-prod/

# 5. Check retention policy
gsutil lifecycle get gs://ollama-audit-logs-prod/
```

**Success Criteria**:
- ✅ Cloud Logging integration deployed
- ✅ Logs visible in Cloud Logging UI
- ✅ GCS bucket storing audit logs
- ✅ 7-year retention configured
- ✅ Immutable log storage verified

---

### Item #3: Documentation Cross-Reference & Index
**Priority**: MEDIUM
**Timeline**: 3-4 days
**Owner**: Documentation Team

#### What to Do
1. **Update README.md** to reference all 4 core documents
2. **Create docs/INDEX.md** as navigation hub
3. **Add Landing Zone compliance section** to README

#### Changes to README.md

**Find this section** (top of file after description):
```markdown
## Getting Started
```

**Add this before it**:
```markdown
## 📚 Documentation

This project maintains comprehensive documentation for enterprise deployment:

- **[API.md](API.md)** - Complete API reference with all endpoints, authentication, and examples
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, scaling strategy, and security architecture
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Step-by-step deployment procedures for dev/staging/production
- **[RUNBOOKS.md](RUNBOOKS.md)** - Operational procedures, incident response, and troubleshooting

For a complete index, see [docs/INDEX.md](docs/INDEX.md).

## Landing Zone Compliance

This repository is an integrated spoke of the **[GCP Landing Zone](https://github.com/kushin77/GCP-landing-zone)**
and follows all enterprise governance standards. See [docs/LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md](docs/LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md)
for current compliance status.
```

#### Create docs/INDEX.md

```markdown
# Complete Documentation Index

## 🎯 Quick Navigation

### Core Documentation (Required)
- [API Documentation](../API.md) - API endpoints, authentication, rate limiting
- [Architecture Design](../ARCHITECTURE.md) - System design and scaling strategy
- [Deployment Guide](../DEPLOYMENT.md) - How to deploy Ollama
- [Operational Runbooks](../RUNBOOKS.md) - How to operate and troubleshoot

### Enterprise Governance
- [Landing Zone Compliance Audit](LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md) - Current compliance status
- [Endpoint Registration](ENDPOINT_REGISTRATION.md) - Domain registry integration
- [Deployment Readiness](../DEPLOYMENT_READINESS_CHECKLIST.md) - Production checklist

### Architecture & Design
- [Architecture Overview](ARCHITECTURE.md) - System design documents
- [System Design](architecture.md) - Detailed system design

### Operations & Monitoring
- [Monitoring & Alerting](MONITORING_AND_ALERTING.md) - Metrics and dashboards
- [Implementation Guide](IMPLEMENTATION_SUMMARY.md) - Feature implementation details

### Compliance & Governance
- [PMO Metadata](../pmo.yaml) - Project ownership and cost tracking
- [Contributing Guidelines](CONTRIBUTING.md) - Development guidelines

## 📖 By Role

### Platform Engineers
1. Start with [ARCHITECTURE.md](../ARCHITECTURE.md)
2. Review [Deployment Guide](../DEPLOYMENT.md)
3. Check [Operational Runbooks](../RUNBOOKS.md)
4. Verify [Landing Zone Compliance](LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md)

### Application Developers
1. Read [API Documentation](../API.md)
2. Review [Integration Examples](INTEGRATION_EXAMPLES.md)
3. Check [Contributing Guidelines](CONTRIBUTING.md)

### Security & Compliance Teams
1. Review [Landing Zone Compliance Audit](LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md)
2. Check [Security Architecture](../ARCHITECTURE.md#security-architecture)
3. Verify [Deployment Checklist](../DEPLOYMENT_READINESS_CHECKLIST.md)

## 🔍 Finding Information

### I want to...

**...understand how Ollama works**
→ Read [ARCHITECTURE.md](../ARCHITECTURE.md)

**...deploy Ollama**
→ Follow [DEPLOYMENT.md](../DEPLOYMENT.md)

**...call an API endpoint**
→ Reference [API.md](../API.md)

**...troubleshoot an incident**
→ Consult [RUNBOOKS.md](../RUNBOOKS.md)

**...ensure compliance**
→ Check [Landing Zone Compliance Audit](LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md)

**...contribute code**
→ Review [Contributing Guidelines](CONTRIBUTING.md)

## 📋 Documentation Status

| Document | Owner | Last Updated | Status |
|----------|-------|--------------|--------|
| [API.md](../API.md) | AI Infra | Jan 19, 2026 | ✅ Current |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | AI Infra | Jan 18, 2026 | ✅ Current |
| [DEPLOYMENT.md](../DEPLOYMENT.md) | AI Infra | Jan 18, 2026 | ✅ Current |
| [RUNBOOKS.md](../RUNBOOKS.md) | AI Infra | Jan 18, 2026 | ✅ Current |
| [Landing Zone Compliance](LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md) | AI Infra | Jan 19, 2026 | ✅ Current |

## 🔗 External Links

- **[GCP Landing Zone](https://github.com/kushin77/GCP-landing-zone)** - Hub governance repository
- **[Ollama Official](https://ollama.ai)** - Official Ollama documentation
- **[FastAPI](https://fastapi.tiangolo.com)** - FastAPI framework documentation
- **[PostgreSQL](https://www.postgresql.org/docs)** - Database documentation

---

**Last Updated**: January 19, 2026
**Maintained By**: AI Infrastructure Team
```

**Success Criteria**:
- ✅ README updated with documentation links
- ✅ docs/INDEX.md created with navigation
- ✅ Landing Zone section added
- ✅ All internal links working

---

## Checklist: This Week

### Monday-Tuesday (Days 1-2)
- [ ] Read Landing Zone domain registry documentation
- [ ] Draft Terraform configuration for endpoint registration
- [ ] Begin Cloud Logging code integration
- [ ] Start Google Cloud API setup

### Wednesday (Day 3)
- [ ] Complete Terraform domain registry entry
- [ ] Implement Cloud Logging in code
- [ ] Create PR for Landing Zone (endpoint registration)
- [ ] Configure Cloud Logging bucket in Terraform

### Thursday (Day 4)
- [ ] Deploy Cloud Logging to staging environment
- [ ] Test audit log collection
- [ ] Update README.md with documentation links
- [ ] Create docs/INDEX.md

### Friday (Day 5)
- [ ] Address PR review feedback (from Landing Zone team)
- [ ] Finalize Cloud Logging configuration
- [ ] Test audit log querying
- [ ] Merge documentation changes

---

## What Happens Next?

### Week 2
- Endpoint Registration PR merges to Landing Zone
- Cloud Logging deployed to production
- 7-year retention configured and tested
- Begin health check validation

### Week 3
- Verify endpoint through Hub Load Balancer
- Load test (100+ req/min)
- Validate Cloud Armor DDoS protection
- Test audit log immutability

### Week 4
- Final compliance verification
- Team training on new architecture
- Production deployment
- Document lessons learned

---

## Key Contacts

**Landing Zone Questions**:
- Repository: https://github.com/kushin77/GCP-landing-zone
- Check docs/ directory for guidance
- Create GitHub issue for clarifications

**Cloud Logging Help**:
- GCP Cloud Logging Docs: https://cloud.google.com/logging/docs
- Google Cloud Support: support.google.com/cloud

**Ollama Team**:
- Slack: #ai-infrastructure
- GitHub: github.com/kushin77/ollama

---

**Priority**: 🚨 CRITICAL - Complete by Feb 15, 2026
**Effort**: ~120 engineering hours (3 weeks)
**Impact**: Full Landing Zone onboarding and production compliance
