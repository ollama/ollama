# GitHub Issue Templates for Landing Zone Onboarding

## Issue 1: Endpoint Registration in Domain Registry

```markdown
---
name: Landing Zone - Endpoint Registration
about: Register Ollama in GCP Landing Zone domain registry
title: 'Landing Zone: Register endpoint in domain registry'
labels: landing-zone,critical,infrastructure
assignees: ''
---

## 🎯 Objective
Register Ollama in the GCP Landing Zone centralized domain registry to enable Hub integration for centralized governance, security, and infrastructure management.

## 📋 Requirements

### Terraform Configuration
- [ ] Create Terraform module for Ollama domain entry
- [ ] Configure Cloud Armor security policy reference
- [ ] Set up health check endpoint (GET /api/v1/health)
- [ ] Define backend service routing rules
- [ ] Add path-based routing for inference/models/conversations

### PR to Landing Zone
- [ ] Fork gcp-landing-zone repository
- [ ] Create feature branch: `feature/ollama-endpoint-registration`
- [ ] Submit PR with Terraform configuration
- [ ] Address review feedback
- [ ] Get approval and merge

### Verification & Testing
- [ ] Test endpoint accessibility via Hub LB
- [ ] Verify health checks passing
- [ ] Load test: 100+ requests/min
- [ ] Validate Cloud Armor is active
- [ ] Check rate limiting working

## 📊 Acceptance Criteria
- Endpoint registered in domain registry
- Accessible via https://elevatediq.ai/ollama
- Health checks: 100% passing
- Load test: 100% success rate
- Cloud Armor: Active and logging

## 📚 References
- [Action Plan](../LANDING_ZONE_ACTION_ITEMS.md#item-1-endpoint-registration-in-domain-registry)
- [GCP Landing Zone](https://github.com/kushin77/GCP-landing-zone)
- [Domain Registry Module](https://github.com/kushin77/GCP-landing-zone/tree/main/terraform/modules/networking/domain-registry)

## ⏰ Timeline
- **Start**: This week
- **Complete**: 2 weeks
- **Effort**: 40 hours

## 👥 Owner
TBD

## 📝 Notes
Blocked by: None
Blocks: #2 (partially)
```

## Issue 2: 7-Year Audit Logging

```markdown
---
name: Landing Zone - 7-Year Audit Logging
about: Configure Google Cloud Logging for FedRAMP compliance
title: 'Landing Zone: Configure 7-year audit logging'
labels: landing-zone,critical,compliance
assignees: ''
---

## 🎯 Objective
Implement Google Cloud Logging integration with 7-year (2,555-day) retention for FedRAMP compliance and security audit trail requirements.

## 📋 Requirements

### Code Integration
- [ ] Install google-cloud-logging package
- [ ] Add Cloud Logging setup in ollama/config.py
- [ ] Implement structured JSON logging
- [ ] Create audit middleware for FastAPI
- [ ] Add environment variable configuration

### Infrastructure (Terraform)
- [ ] Create GCS bucket with 7-year retention policy
- [ ] Configure Cloud Logging sink
- [ ] Set up immutable log storage (WORM)
- [ ] Enable encryption at rest (CMEK)
- [ ] Configure access logging for bucket

### Configuration
- [ ] Update config/production.yaml with logging settings
- [ ] Add Cloud Logging environment variables
- [ ] Configure log level and format
- [ ] Set up audit event types

### Testing & Validation
- [ ] Deploy to staging environment
- [ ] Generate test audit events
- [ ] Verify logs in Cloud Logging UI
- [ ] Confirm GCS bucket storage
- [ ] Test log querying and filtering

## 📊 Acceptance Criteria
- All API requests logged to Cloud Logging
- 7-year retention configured and verified
- Logs in both Cloud Logging and GCS
- Structured JSON format
- All audit events captured
- Log queries working in UI

## 📚 References
- [Action Plan](../LANDING_ZONE_ACTION_ITEMS.md#item-2-7-year-audit-logging)
- [Cloud Logging Docs](https://cloud.google.com/logging/docs)
- [Code Template](../LANDING_ZONE_ACTION_ITEMS.md#code-change-required)

## ⏰ Timeline
- **Start**: This week (parallel with #1)
- **Complete**: 2 weeks
- **Effort**: 40 hours

## 👥 Owner
TBD

## 📝 Notes
Blocked by: None
Blocks: None
Dependencies: Terraform expertise needed
```

## Issue 3: Documentation Cross-Reference

```markdown
---
name: Landing Zone - Documentation Cross-Reference
about: Update README and create documentation index
title: 'Landing Zone: Complete documentation cross-referencing'
labels: landing-zone,documentation,medium
assignees: ''
---

## 🎯 Objective
Complete the Documentation mandate by ensuring all 4 core documents are properly linked and discoverable through README and a centralized index.

## 📋 Requirements

### README.md Updates
- [ ] Add "📚 Documentation" section after intro
- [ ] Link to all 4 core docs (API, Architecture, Deployment, Runbooks)
- [ ] Add "Landing Zone Compliance" section
- [ ] Link to compliance audit and status
- [ ] Update table of contents

### Create docs/INDEX.md
- [ ] Create documentation index
- [ ] Organize by role (Platform Eng, App Dev, Security, Ops)
- [ ] Add quick navigation links
- [ ] Add "Finding Information" Q&A section
- [ ] Add documentation status table

### Verification
- [ ] All links working
- [ ] README renders correctly on GitHub
- [ ] INDEX.md accessible and useful
- [ ] All 4 core docs discoverable

## 📊 Acceptance Criteria
- README has explicit doc links
- docs/INDEX.md created with navigation
- All internal links working
- "Documentation" mandate complete
- Users can easily find docs

## 📚 References
- [Action Plan](../LANDING_ZONE_ACTION_ITEMS.md#item-3-documentation-cross-reference--index)
- [README.md](../README.md)
- [Current API.md](../API.md)

## ⏰ Timeline
- **Start**: This week
- **Complete**: 3 days
- **Effort**: 8 hours

## 👥 Owner
TBD

## 📝 Notes
Blocked by: None
Blocks: None
Quick win - high priority for manifest completion
```

---

## Mass Issue Creation Command

If using GitHub CLI:

```bash
# Create all three issues
gh issue create -t "Landing Zone: Register endpoint in domain registry" \
  -b "$(cat issue-1-endpoint-registration.md)" \
  -l landing-zone,critical,infrastructure

gh issue create -t "Landing Zone: Configure 7-year audit logging" \
  -b "$(cat issue-2-audit-logging.md)" \
  -l landing-zone,critical,compliance

gh issue create -t "Landing Zone: Complete documentation cross-referencing" \
  -b "$(cat issue-3-documentation.md)" \
  -l landing-zone,documentation,medium
```

---

## Status Tracking Issue

```markdown
---
name: Landing Zone - Onboarding Progress
about: Track overall progress toward 100% compliance
title: '🎯 Landing Zone Onboarding Progress Tracker'
labels: landing-zone,tracking
assignees: ''
---

## 📊 Overall Progress

Current Compliance: **84%** → Target: **100%** (Feb 15, 2026)

### Progress by Week

#### ✅ Week 1 (Jan 22-26)
- [ ] Team reads audit documents
- [ ] Planning kickoff meeting scheduled
- [ ] Owners assigned to 3 action items
- [ ] GitHub issues created
- [ ] Endpoint registration work started
- [ ] Audit logging work started
- [ ] Documentation updates started

#### 🔄 Week 2 (Jan 29-Feb 2)
- [ ] Endpoint registration Terraform complete
- [ ] Cloud Logging integration complete
- [ ] Documentation updates complete
- [ ] PR submitted to Landing Zone
- [ ] Staging deployments done

#### 🔄 Week 3 (Feb 5-9)
- [ ] Endpoint PR merged to Landing Zone
- [ ] Testing complete in staging
- [ ] Production deployment
- [ ] Verification tests passing
- [ ] 100% compliance achieved

#### 🔄 Week 4 (Feb 12-15)
- [ ] Team training completed
- [ ] Lessons documented
- [ ] Final compliance verification
- [ ] Celebration! 🎉

### Related Issues
- #X: Endpoint Registration
- #X: Audit Logging
- #X: Documentation

### Blockers
None yet.

### Notes
- Regular status updates every Friday
- Team sync Wednesdays 2pm PT
- Escalation path: @akushnir
```
