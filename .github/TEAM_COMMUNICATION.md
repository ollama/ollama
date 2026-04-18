# Landing Zone Onboarding: Team Communication Templates

---

## 📢 Kickoff Meeting Agenda

**Duration**: 2 hours
**Attendees**: Engineering team, product, infrastructure
**Date**: This week (Jan 22-26, 2026)

### Agenda

#### 1. Overview (15 min)
- Landing Zone compliance status (84%)
- What this means for Ollama
- Why it matters (hub integration, compliance, governance)
- Business impact

#### 2. Audit Findings (30 min)
- What's working great (5 mandates ✅)
- What needs attention (3 critical items)
- Risk assessment
- Timeline overview

#### 3. The 3 Action Items (45 min)

**Item 1: Endpoint Registration** (15 min)
- What it is
- Why it matters
- Who owns it
- Dependencies
- Timeline

**Item 2: Audit Logging** (15 min)
- What it is
- Why it matters
- Who owns it
- Dependencies
- Timeline

**Item 3: Documentation** (15 min)
- What it is
- Why it matters
- Who owns it
- Dependencies
- Timeline

#### 4. Implementation Plan (20 min)
- 2-week sprint structure
- Weekly cadence
- How to track progress
- Questions and answers

#### 5. Q&A & Close (10 min)
- Open discussion
- Confirm assignments
- Next meeting

### Pre-Meeting Reading
- QUICK_REFERENCE.md (5 min)
- LANDING_ZONE_ENFORCEMENT_STATUS.md (10 min)

### Post-Meeting Actions
- [ ] Create GitHub issues
- [ ] Set up project board
- [ ] Schedule weekly syncs
- [ ] Owners read detailed docs

---

## 📧 Email Templates

### Email 1: Announcing Audit Complete

**Subject**: Landing Zone Compliance Audit Complete - 84% Ready ✅

```
Team,

The Landing Zone compliance audit for Ollama is complete!

**Status**: 84% compliant (5 of 7 mandates complete)

**Bottom Line**: We're in excellent shape with strong security, code quality, and
governance. Three focused work items will bring us to 100% compliance by Feb 15.

**The 3 Action Items**:
1. Endpoint Registration (2 weeks, 40 hrs)
2. Audit Logging (2 weeks, 40 hrs)
3. Documentation Linking (3 days, 8 hrs)

**What You Need to Know**:
- Read: QUICK_REFERENCE.md (5 minutes)
- Review: LANDING_ZONE_ACTION_ITEMS.md (30 minutes)
- Attend: Kickoff meeting (this week)

**Timeline**:
- Week of Jan 22: Planning & kickoff
- Weeks of Jan 29 & Feb 5: Implementation
- Week of Feb 12: Verification & celebration

**Questions?**
- See audit documents in repo root
- Ask in #ai-infrastructure Slack
- Schedule 1:1 with @akushnir

Let's get to 100%! 🚀

-[Your Name]
```

### Email 2: Weekly Status Template

**Subject**: Landing Zone Onboarding - Week X Progress Update

```
Team,

**This Week's Progress** (Jan 22-26, 2026)

✅ **Completed**:
- [ ] Kickoff meeting (Jan 23)
- [ ] Audit documents reviewed by team
- [ ] GitHub issues created (#1, #2, #3)
- [ ] Owners confirmed for each item

🔄 **In Progress**:
- [ ] Endpoint registration planning (Owner: ___)
- [ ] Cloud Logging implementation (Owner: ___)
- [ ] Documentation updates (Owner: ___)

🔴 **Blockers**:
None currently.

📊 **Status**:
- Endpoint Registration: 5% (planning)
- Audit Logging: 5% (planning)
- Documentation: 20% (mostly done, linking needed)
- **Overall**: 84% → Target 100% by Feb 15

**Next Week (Jan 29 - Feb 2)**:
- Endpoint registration implementation
- Cloud Logging code integration
- Begin staging deployments

**Metrics**:
- Issues created: 3/3 ✅
- Team trained: 8/8 ✅
- Code reviews needed: TBD
- PRs in progress: 0 (starting this week)

**Questions/Concerns**:
Please reply or reach out in #ai-infrastructure

---
Regular updates every Friday 5pm PT
Next sync: Wednesday Jan 24, 2pm PT

Let's keep the momentum! 🚀
```

### Email 3: Action Item Owner Assignment

**Subject**: Landing Zone - Action Item Assignment

```
Hi [Owner Name],

You're assigned to lead: [Action Item #X - Description]

**What**: [Brief description]
**Why**: [Business impact]
**Effort**: [Hours] over [Timeline]
**Success Criteria**: [Clear checklist]

**Your First Steps** (This Week):
1. Read LANDING_ZONE_ACTION_ITEMS.md section for your item
2. Review code templates / Terraform examples
3. Attend kickoff meeting (Jan 23)
4. Post questions in #ai-infrastructure

**Resources**:
- Action Plan: [Link to specific section]
- Code Templates: [Link]
- References: [Links]
- Contacts: [Names for help]

**Timeline**:
- Week 1: Planning & setup
- Week 2: Implementation
- Week 3: Testing & verification

**Questions?**
- Reply to this email
- Slack @akushnir
- Attend sync meeting Wed 2pm PT

Let's make this happen!

-[Your Name]
```

### Email 4: Blocker/Escalation Template

**Subject**: Landing Zone Onboarding - BLOCKER: [Item Name]

```
Team,

We have a blocker on [Action Item #X]:

**Issue**: [Clear description]

**Impact**:
- Timeline impact: [Days/weeks]
- Risk level: [Critical/High/Medium]
- Affected parties: [Who]

**Root Cause**: [Analysis]

**Proposed Solution**: [Recommendation]

**Decision Needed From**: [Who]

**Timeline for Decision**: [When]

**Next Steps**:
1. [Action]
2. [Action]
3. [Action]

Please respond by [Date] to keep us on track.

Contact: @akushnir for urgent discussion.

---
Status: WAITING FOR DECISION
Escalation Level: [L1/L2/L3]
```

---

## 📊 Weekly Sync Meeting Agenda

**When**: Every Wednesday 2:00 PM PT
**Duration**: 30 minutes
**Attendees**: Owners of 3 action items + PM + Infra lead

### Agenda (30 min)

1. **Status Update** (10 min)
   - Each owner: 2-3 min update on progress
   - Blockers or challenges
   - Confidence level

2. **Review Metrics** (5 min)
   - Overall compliance progress
   - Code review status
   - Testing completion
   - Risk level

3. **Remove Blockers** (10 min)
   - Address any issues
   - Re-assign if needed
   - Escalate if necessary

4. **Plan Next Week** (5 min)
   - Confirm work for next week
   - Identify dependencies
   - Adjust timeline if needed

---

## 💬 Slack Channel Templates

### Channel: #ai-infrastructure (Existing)

#### Pinned Messages (Update these weekly):

```
📌 Landing Zone Onboarding - Status Tracker

Current Compliance: 84% → Target 100% (Feb 15)

🎯 3 Critical Items:
1️⃣ Endpoint Registration (2 wks) - Owner: [Name]
2️⃣ Audit Logging (2 wks) - Owner: [Name]
3️⃣ Documentation (3 days) - Owner: [Name]

📚 Documents:
• Quick Ref: QUICK_REFERENCE.md
• Action Plan: LANDING_ZONE_ACTION_ITEMS.md
• Status: LANDING_ZONE_ENFORCEMENT_STATUS.md
• Deep Dive: docs/LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md

⏰ Schedule:
• Kickoff: Wed Jan 23, 2pm PT
• Weekly Sync: Wed 2pm PT
• Status Updates: Friday EOD

🚀 Getting Started:
1. Read QUICK_REFERENCE.md (5 min)
2. Skim action items (30 min)
3. Attend kickoff Wed
4. Ask questions anytime!
```

#### Daily Standup Template (Post each morning):

```
🌅 Landing Zone Onboarding - Daily Standup [Date]

Yesterday's Progress:
- [ ] Item 1: [Brief update]
- [ ] Item 2: [Brief update]
- [ ] Item 3: [Brief update]

Today's Plan:
- [ ] Item 1: [Planned work]
- [ ] Item 2: [Planned work]
- [ ] Item 3: [Planned work]

Blockers:
None / [List any blockers]

🎯 Confidence Level: 🟢 On Track / 🟡 At Risk / 🔴 Blocked
```

---

## 📋 Weekly Status Report Template

**Week of**: [Jan 22-26]
**Report Date**: [Friday 5pm]
**Overall Status**: 🟢 On Track

### Summary
One line summary of where we are.

### Progress This Week
- ✅ Item 1: [Specific progress]
- ✅ Item 2: [Specific progress]
- ✅ Item 3: [Specific progress]

### Metrics
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Overall Compliance | 100% | 84% | On Track |
| Issues Created | 3 | 3 | ✅ Done |
| Team Trained | 8/8 | 8/8 | ✅ Done |
| Code PRs Needed | 3 | 0 | In Progress |
| Tests Passing | 100% | N/A | Not Started |

### Blockers
None currently.

### Risk Assessment
🟢 **Low Risk** - Team ready, clear path, good timeline

### Next Week Plan
- Endpoint registration starts
- Cloud Logging implementation begins
- Documentation updates complete

### Confidence Level
🟢 **High** - Team engaged, resources allocated, clear plan

---

## 🎯 Communication Frequency

| Format | Frequency | Owner | Audience |
|--------|-----------|-------|----------|
| Email Update | Weekly (Friday) | PM | All stakeholders |
| Slack Standup | Daily | Team | #ai-infrastructure |
| Sync Meeting | Weekly (Wed) | Owner | Team leads |
| Status Report | Weekly (Friday) | PM | Leadership |
| Escalation | As needed | Owner | Leads |

---

## ✅ Communication Checklist

Before starting the project:
- [ ] Send kickoff announcement email
- [ ] Invite team to kickoff meeting
- [ ] Create GitHub issues from templates
- [ ] Pin status message in Slack
- [ ] Schedule weekly sync meetings
- [ ] Set up project board

During the project:
- [ ] Daily standup in Slack
- [ ] Weekly sync meetings
- [ ] Weekly status emails
- [ ] Friday status reports
- [ ] Address blockers immediately

After completion:
- [ ] Final status report
- [ ] Lessons learned document
- [ ] Team celebration
- [ ] Archive project materials

---

**Communication Owner**: [PM Name]
**Last Updated**: January 19, 2026
**Next Review**: January 24, 2026
