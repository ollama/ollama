# Landing Zone Compliance - Action Plan

**Date**: January 19, 2026
**Status**: URGENT - Multiple Mandates Require Immediate Action
**Owner**: AI Infrastructure Team
**Reference**: [GCP Landing Zone Onboarding](https://github.com/kushin77/gcp-landing-zone)

---

## Executive Summary

Analysis of the GCP Landing Zone repository reveals **7 critical mandates** requiring immediate implementation. Most critical is **Mandate #6 (Endpoint Registration)** added January 18, 2026, which requires all public-facing services to be registered in the centralized domain registry.

**Current Compliance Status**: ⚠️ **3 of 7 mandates non-compliant**

---

## Critical Findings

### ✅ COMPLIANT Mandates

1. **Mandate #4: PMO Tracking & Metadata** ✅
   - `pmo.yaml` exists with all 24 required labels
   - All categories complete: Organizational, Lifecycle, Business, Technical, Financial, Git Attribution
   - No action required

2. **Mandate #3: IaC & Terraform Standards** ✅
   - Using Terraform for infrastructure
   - Docker containerization implemented
   - No action required for core standards

3. **Mandate #2: Git Hygiene** ✅ (Partially)
   - GPG commit signing enforced via pre-commit hooks
   - Linear history maintained
   - Secret scanning active
   - **Action**: Verify last 50 commits are all GPG signed

### ❌ NON-COMPLIANT Mandates (Immediate Action Required)

---

## Priority 1: CRITICAL - Endpoint Registration (NEW)

**Mandate #6**: "All public-facing services MUST be registered in the centralized domain registry"

**Added**: January 18, 2026
**Status**: ❌ **NOT COMPLIANT**
**Risk**: Deployment blocked until compliant

### What Changed

The Landing Zone team deployed a centralized Load Balancer Domain Registry on Jan 18, 2026. All spokes with public endpoints must now register in this registry rather than managing their own load balancers.

**New Architecture Pattern**:
```
Before (Dec 2025):
Ollama → Own GCP Load Balancer → https://elevatediq.ai/ollama

After (Jan 2026 - MANDATORY):
Ollama → Hub Domain Registry → Centralized LB → https://elevatediq.ai/ollama
```

### Required Actions

#### 1. Register Ollama in Domain Registry

**File**: Submit PR to `gcp-landing-zone/terraform/modules/networking/lb-domain-registry/variables.tf`

```hcl
domain_registry = {
  # ... existing entries ...

  "ollama" = {
    domain             = "elevatediq.ai"
    subdomains         = ["ollama"]
    tls_enabled        = true
    oauth_protected    = false  # Public API (API key authenticated)
    cloud_armor_policy = "global-armor"
    backend_service    = "ollama-api-backend"
    health_check_path  = "/health"
    timeout_sec        = 30
    enable_cdn         = true

    path_rules = {
      "inference" = {
        paths   = ["/api/v1/generate", "/api/v1/chat", "/api/v1/embeddings"]
        service = "ollama-inference-backend"
      }
      "models" = {
        paths   = ["/api/v1/models/*"]
        service = "ollama-models-backend"
      }
    }
  }
}
```

**Timeline**: 1 week

#### 2. Implement Required Security Controls

Per FedRAMP requirements, all registered endpoints must have:

- ✅ TLS 1.3 enforcement (already implemented)
- ❌ Cloud Armor DDoS protection (NOT IMPLEMENTED)
- ❌ 7-year audit logging (NOT CONFIGURED)
- ✅ Health check endpoint (already implemented at `/health`)
- ❌ Structured request logging (PARTIAL)

**Timeline**: 2 weeks

#### 3. Update ollama Configuration

**Changes Required**:

**File**: `docker-compose.prod.yml`
```yaml
services:
  api:
    environment:
      # ❌ REMOVE direct public exposure
      # FASTAPI_HOST: 0.0.0.0

      # ✅ ADD internal-only configuration
      FASTAPI_HOST: 0.0.0.0
      INTERNAL_ONLY: "true"
      EXTERNAL_ENDPOINT: "https://elevatediq.ai/ollama"
      ENABLE_CLOUD_LOGGING: "true"
      LOG_RETENTION_DAYS: "2555"  # 7 years
```

**Timeline**: 3 days

---

## Priority 2: CRITICAL - Documentation Requirements

**Mandate #5: Documentation**

**Status**: ❌ **NOT COMPLIANT - Missing 4 required files**

### Required Files

Landing Zone requires all spokes to provide:

1. ❌ `API.md` - API endpoint documentation
2. ❌ `ARCHITECTURE.md` - System design documentation
3. ❌ `DEPLOYMENT.md` - Deployment procedures
4. ❌ `RUNBOOKS.md` - Operational runbooks

**Current Status**: Only have README.md and various docs/ files

### Required Actions

#### 1. Create API.md

**Location**: `/home/akushnir/ollama/API.md`

**Required Content**:
- All public endpoints with examples
- Authentication methods
- Rate limiting details
- Error codes and responses
- Request/response schemas
- Versioning strategy

**Timeline**: 1 week

#### 2. Create ARCHITECTURE.md

**Location**: `/home/akushnir/ollama/ARCHITECTURE.md`

**Required Content**:
- System components diagram
- Data flow diagrams
- Technology stack
- Scaling strategy
- Failure modes and resilience
- Security architecture

**Timeline**: 1 week

#### 3. Create DEPLOYMENT.md

**Location**: `/home/akushnir/ollama/DEPLOYMENT.md`

**Required Content**:
- Deployment prerequisites
- Step-by-step deployment procedure
- Configuration management
- Secrets management
- Database migrations
- Rollback procedures
- Production checklist

**Timeline**: 3 days

#### 4. Create RUNBOOKS.md

**Location**: `/home/akushnir/ollama/RUNBOOKS.md`

**Required Content**:
- Common incident scenarios
- Troubleshooting procedures
- Emergency contacts
- Escalation paths
- Recovery procedures
- Post-incident review template

**Timeline**: 1 week

---

## Priority 3: HIGH - OAuth Enforcement (NEW)

**Mandate #7**: "OAuth 2.0 Enforcement for All User-Facing Apps"

**Added**: January 2026
**Status**: ❌ **NOT APPLICABLE (but should consider)**

### Analysis

Ollama API is **machine-to-machine** (API key authentication), not user-facing, so OAuth/IAP is optional. However, if we add a **web UI or admin portal**, OAuth becomes mandatory.

### Required Actions

#### For Future Web UI (When Built)

If/when we build a web UI:

1. Enable IAP (Identity-Aware Proxy)
2. Configure OAuth 2.0 client
3. Use Workload Identity (no JSON keys)
4. Integrate with corporate identity provider

**Timeline**: Not applicable until web UI is planned

---

## Priority 4: MEDIUM - Mandatory Cleanup Policy

**Policy**: "All spokes must complete cleanup before Hub integration"

**Status**: ⚠️ **NEEDS REVIEW**

### Required Cleanup Phases

#### Phase 1: Structural Cleanup (-2% size minimum)

**Actions**:
- Remove duplicate/unused files
- Archive old scripts
- Clean up build artifacts
- Remove temporary files

**Command**: `bash scripts/cleanup-root-directory.sh`

#### Phase 2: Documentation Consolidation

**Actions**:
- Merge redundant documentation files
- Organize docs/ directory
- Remove outdated guides
- Update index files

#### Phase 3: Security & Performance Enforcement

**Actions**:
- Run security audit: `pip-audit`
- Fix all critical vulnerabilities
- Optimize Docker images
- Remove unused dependencies

### Timeline

- Phase 1: 2 days
- Phase 2: 3 days
- Phase 3: 1 week

---

## Detailed Implementation Plan

### Week 1: Endpoint Registration (CRITICAL)

**Days 1-2**: Prepare domain registry entry
- [ ] Write Terraform configuration for ollama domain
- [ ] Configure Cloud Armor security policy
- [ ] Set up health check configuration
- [ ] Document backend service requirements

**Days 3-4**: Submit registration PR
- [ ] Create PR to `gcp-landing-zone` repository
- [ ] Include all required security controls
- [ ] Add comprehensive PR description
- [ ] Address review feedback

**Day 5**: Test and verify
- [ ] Test endpoint after Terraform apply
- [ ] Verify health checks passing
- [ ] Test API requests through new endpoint
- [ ] Monitor logs and metrics

### Week 2: Documentation Requirements

**Days 1-3**: Create core documentation
- [ ] Write API.md with all endpoints
- [ ] Create ARCHITECTURE.md with diagrams
- [ ] Document DEPLOYMENT.md procedures

**Days 4-5**: Operational documentation
- [ ] Write RUNBOOKS.md with incident procedures
- [ ] Update README.md to reference new docs
- [ ] Create documentation index

### Week 3: Security & Cleanup

**Days 1-2**: Security enhancements
- [ ] Configure 7-year audit logging
- [ ] Implement Cloud Armor DDoS protection
- [ ] Enable structured request logging
- [ ] Test security controls

**Days 3-5**: Mandatory cleanup
- [ ] Run cleanup scripts
- [ ] Remove unused files
- [ ] Optimize Docker images
- [ ] Update dependencies

### Week 4: Integration & Testing

**Days 1-3**: End-to-end testing
- [ ] Test all endpoints through Hub LB
- [ ] Verify authentication and authorization
- [ ] Load test with 100+ requests/min
- [ ] Test failover and recovery

**Days 4-5**: Documentation and handoff
- [ ] Update all documentation
- [ ] Create onboarding guide for team
- [ ] Schedule training session
- [ ] Document lessons learned

---

## Compliance Checklist

### Mandate #1: Security by Design ✅
- [x] Zero-trust networking (private by default)
- [x] CMEK encryption (not applicable for Ollama)
- [x] Audit logging enabled
- [ ] **7-year retention configured** (ACTION REQUIRED)

### Mandate #2: Git Hygiene ✅
- [x] GPG signed commits (enforced via hooks)
- [x] Linear history (rebase strategy)
- [x] Secret-free history (gitleaks/trufflehog)

### Mandate #3: IaC & Terraform Standards ✅
- [x] Terraform used for infrastructure
- [x] Docker containerization
- [x] Environment variable configuration

### Mandate #4: PMO Tracking & Metadata ✅
- [x] `pmo.yaml` exists
- [x] All 24 labels present
- [x] Ownership and cost attribution defined

### Mandate #5: CI/CD Gates ⚠️
- [x] Pre-commit hooks (format, checks)
- [x] Security scans (pip-audit, safety)
- [ ] **Commit signature verification in CI** (VERIFY)

### Mandate #6: Documentation ❌
- [ ] **API.md** (REQUIRED)
- [ ] **ARCHITECTURE.md** (REQUIRED)
- [ ] **DEPLOYMENT.md** (REQUIRED)
- [ ] **RUNBOOKS.md** (REQUIRED)

### Mandate #7: Endpoint Registration ❌
- [ ] **Registered in domain registry** (REQUIRED)
- [ ] **Cloud Armor configured** (REQUIRED)
- [ ] **7-year audit logging** (REQUIRED)
- [x] Health check endpoint exists

---

## Risk Assessment

### High Risk (Deployment Blockers)

1. **Endpoint Registration** (Mandate #6)
   - **Risk**: Deployment blocked until compliant
   - **Impact**: Cannot update production endpoint
   - **Mitigation**: Complete registration within 1 week

2. **Documentation** (Mandate #5)
   - **Risk**: Onboarding to Hub blocked
   - **Impact**: Cannot integrate with Landing Zone
   - **Mitigation**: Create required docs within 2 weeks

### Medium Risk (Compliance Issues)

3. **7-Year Audit Logging**
   - **Risk**: FedRAMP compliance violation
   - **Impact**: Audit failure, potential service shutdown
   - **Mitigation**: Configure Cloud Logging with 7-year retention

4. **Cloud Armor DDoS Protection**
   - **Risk**: Vulnerable to DDoS attacks
   - **Impact**: Service unavailability
   - **Mitigation**: Enable Cloud Armor via domain registry

### Low Risk (Best Practice)

5. **OAuth for Web UI**
   - **Risk**: If web UI built without OAuth, non-compliant
   - **Impact**: Security gap for user-facing interface
   - **Mitigation**: Document OAuth requirement for future UI

---

## Cost Impact

### One-Time Costs

- **Cloud Armor**: ~$10/month for security policies
- **Cloud Logging (7-year retention)**: ~$50/month for audit logs
- **SSL Certificate (Google Managed)**: $0 (included)

**Total**: ~$60/month additional

### Time Investment

- **Engineering**: 120 hours (3 weeks × 40 hours)
- **Documentation**: 40 hours (1 week)
- **Testing**: 20 hours (2-3 days)

**Total**: ~180 engineering hours

---

## Success Criteria

### Week 1 Complete
- [x] Domain registry PR submitted
- [x] Cloud Armor policy defined
- [x] Health check endpoint verified

### Week 2 Complete
- [x] All 4 required docs created (API, ARCHITECTURE, DEPLOYMENT, RUNBOOKS)
- [x] Documentation reviewed by team
- [x] Docs merged to main branch

### Week 3 Complete
- [x] 7-year audit logging configured
- [x] Cloud Armor DDoS protection active
- [x] Mandatory cleanup completed

### Week 4 Complete
- [x] End-to-end testing passed
- [x] All mandates compliant
- [x] Team trained on new architecture
- [x] Production deployment successful

---

## Next Steps (This Week)

### Day 1 (Monday)
1. Read Landing Zone onboarding guides
2. Review domain registry examples
3. Draft Terraform configuration for ollama

### Day 2 (Tuesday)
1. Complete domain registry Terraform
2. Configure Cloud Armor policy
3. Test configuration locally

### Day 3 (Wednesday)
1. Submit PR to gcp-landing-zone repository
2. Start API.md documentation
3. Diagram system architecture

### Day 4 (Thursday)
1. Address PR review feedback
2. Complete ARCHITECTURE.md
3. Start DEPLOYMENT.md

### Day 5 (Friday)
1. Merge domain registry PR
2. Verify endpoint registration
3. Test API through new endpoint

---

## References

### Landing Zone Documentation

- [Endpoint Onboarding Integration Guide](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/onboarding/ENDPOINT_ONBOARDING_INTEGRATION.md)
- [Onboarding Boundaries & Mandates](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/governance/policies/ONBOARDING_BOUNDARIES_AND_MANDATES.md)
- [Mandatory Cleanup Checklist](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/onboarding/MANDATORY_CLEANUP_CHECKLIST.md)
- [Spoke Onboarding Master Guide](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/onboarding/SPOKE_ONBOARDING_MASTER_GUIDE.md)

### Ollama Repository

- [pmo.yaml](../pmo.yaml) ✅
- [Endpoint Registration](ENDPOINT_REGISTRATION.md) (NEW)
- [GCP Load Balancer Setup](GCP_LB_SETUP.md)
- [Deployment Readiness Checklist](../DEPLOYMENT_READINESS_CHECKLIST.md)

---

**Status**: 🔴 **ACTION REQUIRED**
**Owner**: AI Infrastructure Team
**Deadline**: February 15, 2026 (4 weeks)
**Next Review**: January 26, 2026 (1 week)
