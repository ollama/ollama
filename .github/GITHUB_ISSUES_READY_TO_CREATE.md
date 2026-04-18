# GitHub Issues Ready to Create (Jan 23)

**Purpose**: Copy-paste these into GitHub to create 4 tracking issues for the Landing Zone compliance project.

**When**: Tuesday, Jan 23, 2026 (after kickoff meeting)
**Who**: Project Manager
**Labels to Add**: `landing-zone`, `compliance`, `critical-path`

---

## Issue #1: Endpoint Registry Integration

**Title**: `[Landing Zone] Endpoint Registry Integration (40 hrs) - Action Item #1`

**Assignee**: @endpoint-owner (TBD)

**Labels**: `landing-zone`, `compliance`, `critical-path`, `infrastructure`

**Milestone**: Week 2 (Jan 29 - Feb 2)

**Body**:
```markdown
## Overview
Register Ollama service endpoints in GCP Landing Zone's centralized domain registry.

## Current Status
- ❌ Not Started
- 🎯 Target Completion: Feb 2, 2026
- ⏱️ Estimated Effort: 40 engineering hours

## Requirements
1. **Domain Registry Registration**
   - Register service in domain registry
   - Provide service IP/hostname
   - Configure DNS entries
   - Test DNS resolution

2. **Terraform Configuration**
   - Create terraform/domain_entries.tf
   - Define domain entries with all required metadata
   - Include cost center tags
   - Version control configuration

3. **Testing & Validation**
   - Verify DNS resolution from external clients
   - Test through GCP Load Balancer
   - Validate from external network
   - Document test results

4. **Documentation**
   - Update DEPLOYMENT.md with registry details
   - Add DNS configuration to README
   - Document registration process

## Success Criteria
- ✅ Service registered in domain registry
- ✅ DNS resolves correctly from external network
- ✅ Service accessible through GCP LB
- ✅ All terraform code merged to main
- ✅ Documentation updated

## Blockers/Dependencies
- None (can start immediately)

## References
- See: LANDING_ZONE_ACTION_ITEMS.md (Action #1)
- See: docs/LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md (Mandate #6)
- GCP Landing Zone docs: https://github.com/kushin77/GCP-landing-zone

## Related Issues
- #2: Audit Logging Implementation
- #3: Documentation Linking
```

---

## Issue #2: Audit Logging Implementation

**Title**: `[Landing Zone] 7-Year Audit Logging (40 hrs) - Action Item #2`

**Assignee**: @logging-owner (TBD)

**Labels**: `landing-zone`, `compliance`, `critical-path`, `infrastructure`, `security`

**Milestone**: Week 2-3 (Jan 29 - Feb 9)

**Body**:
```markdown
## Overview
Implement 7-year audit logging using Google Cloud Logging with GCS backend per Landing Zone Mandate #9.

## Current Status
- ❌ Not Started
- 🎯 Target Completion: Feb 9, 2026
- ⏱️ Estimated Effort: 40 engineering hours

## Requirements
1. **Cloud Logging Integration**
   - Add google-cloud-logging library
   - Configure Python logging sink
   - Stream all API requests to Cloud Logging
   - Include request/response metadata

2. **GCS Backend Setup**
   - Create GCS bucket for audit logs
   - Configure lifecycle policies (7-year retention)
   - Set encryption (CMEK preferred)
   - Enable versioning for immutability

3. **Terraform Configuration**
   - Create terraform/audit_logging.tf
   - Define GCS bucket with proper policies
   - Configure Cloud Logging sink
   - Set IAM roles for access control

4. **Testing & Validation**
   - Verify logs appear in Cloud Logging
   - Verify logs transfer to GCS
   - Test 7-year lifecycle policy
   - Validate immutability settings

5. **Production Deployment**
   - Deploy to staging first
   - Validate for 24 hours
   - Deploy to production
   - Monitor for issues

## Success Criteria
- ✅ All API requests logged to Cloud Logging
- ✅ Logs transfer to GCS automatically
- ✅ 7-year retention configured
- ✅ Immutability enforced
- ✅ All terraform code merged
- ✅ Documentation updated

## Blockers/Dependencies
- Must be completed before production deployment (Feb 9)

## References
- See: LANDING_ZONE_ACTION_ITEMS.md (Action #2)
- See: docs/LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md (Mandate #9)
- Google Cloud Logging docs: https://cloud.google.com/logging/docs

## Related Issues
- #1: Endpoint Registry Integration
- #3: Documentation Linking
```

---

## Issue #3: Documentation Linking

**Title**: `[Landing Zone] Documentation Linking (8 hrs) - Action Item #3`

**Assignee**: @documentation-owner (TBD)

**Labels**: `landing-zone`, `compliance`, `documentation`

**Milestone**: Week 1 (Jan 22-26)

**Body**:
```markdown
## Overview
Link all Landing Zone compliance documentation so new users can easily navigate and understand project status.

## Current Status
- ❌ Not Started
- 🎯 Target Completion: Jan 29, 2026
- ⏱️ Estimated Effort: 8 engineering hours

## Requirements
1. **Update README.md**
   - Add "Landing Zone Compliance" section
   - Link to START_HERE.md
   - Add compliance status badge (84%)
   - Link to all key documents

2. **Create docs/INDEX.md**
   - Comprehensive index of all docs
   - Organized by topic/audience
   - Search-friendly structure
   - Reading recommendations by role

3. **Cross-Linking**
   - Link ACTION_ITEMS to technical docs
   - Link MILESTONES to specific owners
   - Link DASHBOARD to current issues
   - Verify all links work

4. **Verification**
   - Test all markdown links
   - Verify no 404s
   - Check from different starting points
   - Ensure discoverability

## Success Criteria
- ✅ README has compliance section
- ✅ docs/INDEX.md created and complete
- ✅ All cross-links working
- ✅ New users can navigate easily
- ✅ PR merged to main

## Blockers/Dependencies
- None (can be done in parallel with other items)

## References
- See: LANDING_ZONE_ACTION_ITEMS.md (Action #3)
- See: docs/LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md (Mandate #7)

## Related Issues
- #1: Endpoint Registry Integration
- #2: Audit Logging Implementation
```

---

## Issue #4: Progress Tracking

**Title**: `[Landing Zone] Compliance Progress Tracking - Status Updates`

**Assignee**: @project-manager

**Labels**: `landing-zone`, `compliance`, `tracking`

**Milestone**: Ongoing (Jan 22 - Feb 15)

**Body**:
```markdown
## Overview
Track compliance progress throughout 4-week execution (Jan 22 - Feb 15, 2026).

## Current Status
- 📊 Compliance: 84% → Target: 100% by Feb 15
- 📅 Week: 1 of 4
- 🎯 Critical Path: On Track

## Weekly Status Updates

### Week 1 (Jan 22-26): Planning & Kickoff
**Target**: Team aligned, owners assigned, issues created
- [ ] Kickoff meeting held (Jan 23)
- [ ] 3 owners confirmed
- [ ] GitHub issues created (#1-4)
- [ ] Owner 1:1 meetings scheduled (Jan 24-25)
- [ ] Week 1 summary posted

### Week 2 (Jan 29-Feb 2): Development
**Target**: Endpoint registry & documentation complete
- [ ] Daily standups held
- [ ] Endpoint registry Terraform started
- [ ] Audit logging Python code started
- [ ] Documentation linking started
- [ ] Weekly sync held (Feb 2)

### Week 3 (Feb 5-9): Testing & Deployment
**Target**: All 3 action items complete, merged to main
- [ ] Endpoint registry merged
- [ ] Audit logging deployed to staging
- [ ] Documentation complete
- [ ] Production deployment approved
- [ ] 100% compliance verified

### Week 4 (Feb 12-15): Training & Closure
**Target**: Team trained, project closed
- [ ] Training session held (Feb 12)
- [ ] Final verification complete
- [ ] Project closure documented
- [ ] Team celebration 🎉

## How to Update This Issue
Every Friday at end of day:
1. Update the relevant week's checklist
2. Add blockers/risks if any
3. Post link to status report in comments
4. Update milestone if needed

## Status Report Template
Use this format in comments:

```
## Status Report: Week X (Date)
**Compliance**: XX% (was XX%)
**On Track**: Yes/No
**Items Complete**: #/#
**Blockers**: (if any)
**Next Week Focus**: (top 3 items)
```

## Success Criteria
- ✅ Weekly updates posted every Friday
- ✅ All blocker issues tracked
- ✅ Team visibility maintained
- ✅ Feb 15 deadline on track

## References
- LANDING_ZONE_DASHBOARD.md (for template)
- THIS_WEEK_JAN_22-26.md (for weekly plans)
```

---

## How to Create These Issues

### Option 1: Manual Creation (1-2 minutes per issue)
1. Go to https://github.com/kushin77/ollama/issues/new
2. Copy title from above
3. Paste body content
4. Add assignee and labels
5. Set milestone (if available)
6. Click "Create issue"

### Option 2: Bulk Creation with GitHub CLI
```bash
# If you have GitHub CLI installed:
cd /home/akushnir/ollama

# Create Issue #1
gh issue create \
  --title "[Landing Zone] Endpoint Registry Integration (40 hrs) - Action Item #1" \
  --body "$(cat - << 'EOF'
## Overview
Register Ollama service endpoints in GCP Landing Zone's centralized domain registry...
EOF
)" \
  --assignee "" \
  --label "landing-zone,compliance,critical-path,infrastructure"

# (Repeat for issues #2, #3, #4)
```

### Option 3: Use GitHub API
See `.github/ISSUE_TEMPLATES.md` for cURL examples.

---

## Next Steps

**✅ TODAY (Jan 19 - Saturday)**
- [ ] Read this file
- [ ] Identify 3 action item owners
- [ ] Ask them: "Can you commit Jan 22-Feb 15?"
- [ ] Get leadership approval

**📅 MONDAY (Jan 22)**
- [ ] Confirm owners can start immediately
- [ ] Send kickoff meeting invitation for Jan 23

**📅 TUESDAY (Jan 23)**
- [ ] Hold 2-hour kickoff meeting
- [ ] Create these 4 GitHub issues
- [ ] Assign owners to issues #1-3
- [ ] Announce in team Slack

**📅 WED-THU (Jan 24-25)**
- [ ] 1:1 meeting with each owner
- [ ] Review their task breakdown
- [ ] Confirm timeline is realistic
- [ ] Identify any blockers early

**📅 FRIDAY (Jan 26)**
- [ ] Final check - all owners ready
- [ ] Post status report to #4 (this issue)
- [ ] Celebrate team alignment!

---

## Tracking Board Setup (Optional)

If you want to track visually:
1. Go to your GitHub Project board
2. Create 4 columns: `Todo`, `In Progress`, `Review`, `Done`
3. Add issues #1-4 to the board
4. Update as work progresses

Or use the simpler approach:
- Use the LANDING_ZONE_DASHBOARD.md weekly
- Update issue #4 (this tracking issue) every Friday

---

**That's it!** These 4 issues are your entire tracking system for the next 4 weeks.

Everything you need to know is in the issue descriptions. Owners can check their issue for detailed requirements, success criteria, and references.

🚀 Ready to create? Follow Option 1 or 2 above!
