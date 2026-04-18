# PMO Weekly Status Report Implementation Guide

**Version**: 1.0
**Effective**: 2026-01-26
**Maintained By**: Project Management Office

---

## Overview

The PMO Weekly Status Report is the single source of truth for project progress. It provides visibility into:

- What was completed this week
- What's in progress and on track
- What risks or blockers exist
- What the team is working on next
- Whether we're on schedule for the phase

---

## Report Schedule

### Weekly Cadence

**Friday 4:00 PM UTC**: Status Report Due

- Compiled from issue tracking, team updates, metrics
- Posted to #pmo-status Slack channel
- Tagged @leadership for review
- Typical report creation time: 30 minutes

**Monday 9:00 AM UTC**: Weekly Review Meeting

- 30-minute meeting
- Attendees: Engineering lead, PM, critical path issue owners
- Review Friday status report
- Address any blockers
- Adjust next week priorities if needed
- Decisions made documented in report

### Monthly Cadence (End of Month)

**Last Friday of Month**: Monthly Summary

- Compile all 4 weekly reports
- Calculate burn-down metrics
- Assess velocity trend
- Risk reassessment
- Leadership review + approval for next month

---

## Report Creation Workflow

### Step 1: Gather Data (Thursday EOD)

**From GitHub Issues**:

```bash
# Closed this week
gh issue list --state closed --created "after:2026-01-19" \
  --repo kushin77/ollama

# In progress
gh issue list --state open --label "in-progress" \
  --repo kushin77/ollama

# Risk/Blocker labels
gh issue list --state open --label "blocker" \
  --repo kushin77/ollama
```

**From Metrics**:

- Test coverage: Last CI/CD run report
- Code quality: Last ruff/mypy run
- Security: Last pip-audit/snyk scan
- Deployment: Cloud Build logs

**From Team**:

- Ask: "What did you finish this week?"
- Ask: "What are you working on next?"
- Ask: "Any blockers or risks?"

### Step 2: Fill Template (Friday 2:00 PM)

1. Copy `WEEKLY_STATUS_TEMPLATE.md`
2. Fill in dates and reporting period
3. Add completed issues (from GitHub closed list)
4. Add in-progress issues (from GitHub open list)
5. Add starting next week (from pipeline/backlog)
6. Fill in metrics from last CI/CD run
7. List active blockers/risks
8. Summarize any decisions made
9. List top 3 priorities for next week

### Step 3: Review & Refine (Friday 3:30 PM)

- Review for accuracy with engineering lead
- Verify all numbers/dates are correct
- Ensure action items have owners and dates
- Flag any blockers that need escalation

### Step 4: Publish (Friday 4:00 PM)

```bash
# Create file in repo
git checkout -b pmo/status-report-2026-01-26
cp wiki/pmo-status-reports/WEEKLY_STATUS_TEMPLATE.md \
   wiki/pmo-status-reports/2026-01-26-status-report.md

# Edit and commit
git add wiki/pmo-status-reports/2026-01-26-status-report.md
git commit -S -m "docs(pmo): weekly status report for week of 2026-01-26"
git push origin pmo/status-report-2026-01-26

# Create PR
gh pr create --fill  # or manually create

# Merge to main
gh pr merge  # After approval
```

Post to Slack:

```
📊 WEEKLY STATUS REPORT
Week of [Dates]

Status: 🟢 ON TRACK | 🟡 AT RISK | 🔴 CRITICAL

✅ Completed: [# issues]
🔵 In Progress: [# issues]
📅 Starting Next: [# issues]

🚨 Blockers: [# blockers]
🎯 Top Priority: Issue #[#] - [task]

📄 Full report: [link to document]
```

---

## Key Sections Explained

### Executive Summary

Quick snapshot for leadership. Should answer:

- Are we on schedule?
- Are there any fires?
- What's the risk level?

If status is 🟡 AT RISK or 🔴 CRITICAL, this section should explain why and what we're doing about it.

### Completed This Week

List every issue closed, along with:

- **What was delivered**: Description of work
- **Quality metrics**: Did it meet our standards?
- **Next step**: What does this unblock?

### In Progress

List every in-progress issue with:

- **Current status**: What % complete?
- **Current milestone**: What are they working on right now?
- **Next milestone**: What comes next?
- **Blockers**: Is anything preventing progress?
- **Risk level**: Are they on track?

### Starting Next Week

Preview of what's coming. Helps leadership see forward trajectory and spot emerging conflicts (e.g., two tasks both need same person).

### Metrics & Velocity

Show burn-down chart trends:

- Are we completing faster/slower than expected?
- Will we finish on schedule?
- Do we need to add/remove resources?

### Risks & Blockers

#### Blocker vs. Risk

**Blocker**: Preventing work RIGHT NOW

- Example: "Can't deploy because CI/CD is broken"
- Status: Blocking [issue #], must resolve by [DATE]
- Owner: Person responsible for fixing it

**Risk**: Might prevent work in FUTURE

- Example: "Vendor hasn't confirmed hardware delivery yet"
- Probability: Might happen (LOW/MEDIUM/HIGH)
- Mitigation: What are we doing to prevent it?

### Decisions Made

Document any decisions from this week (e.g., architectural choices, tradeoffs, scope changes). Include:

- What we decided
- Why we decided it
- Who approved it
- What it enables/prevents

### Top 3 Priorities

What must get done next week. Critical path items get priority 1. This helps team focus if there are conflicts.

---

## Sign-Off Process

### Who Approves?

- **Report Owner**: PM or engineering lead (who writes it)
- **Technical Reviewer**: Engineering lead (verifies accuracy)
- **Leadership Review**: CTO or VP Eng (for decisions/escalations)

### Approval Criteria

Before approving, verify:

- [ ] All closed issues actually closed
- [ ] All in-progress percentages accurate
- [ ] All blockers have owners
- [ ] All dates are realistic
- [ ] All decisions documented with approvals
- [ ] No surprises for leadership

---

## Distribution & Archival

### Weekly Distribution

- **Slack**: #pmo-status channel (for team visibility)
- **Email**: [Leadership distribution list] (for record)
- **GitHub**: Merged to main in `/wiki/pmo-status-reports/`
- **Wiki**: Searchable by date

### Monthly Archive

Create monthly summary:

- Consolidate 4-5 weekly reports
- Calculate trend metrics
- Identify recurring blockers
- Recommend process improvements

### Retention

- Keep all reports in `/wiki/pmo-status-reports/`
- Organized by year/month: `2026/01-status-report-2026-01-26.md`
- Searchable by keyword or date

---

## Escalation Checklist

If status report shows 🟡 AT RISK or 🔴 CRITICAL:

**Escalation Level 1**: Engineering Lead Review

- [ ] Review blockers and mitigation plans
- [ ] Determine if escalation needed to CTO

**Escalation Level 2**: CTO Review (if level 1 can't fix)

- [ ] Review decision log for roadblocks
- [ ] Determine if scope/timeline adjustment needed
- [ ] Assign emergency resources if critical

**Escalation Level 3**: Founder Review (if >1 week delay)

- [ ] Review overall project health
- [ ] Decision: continue, reduce scope, or delay?
- [ ] Communicate to stakeholders

---

## Common Report Patterns

### Pattern: "On Track" Report (🟢 GREEN)

```
Status: ON TRACK
✅ Completed 4 issues (met target of 4)
🔵 In Progress: 3 issues (on schedule)
Blockers: None
Risks: None identified

Priority: Focus on Issue #11 (CI/CD) - on critical path
```

### Pattern: "At Risk" Report (🟡 YELLOW)

```
Status: AT RISK
✅ Completed 2 issues (missed target of 4)
🔵 In Progress: 3 issues (slipping on schedule)
Blockers: Issue #10 blocked by [dependency]

Mitigation:
- Adding 1 engineer to Issue #11
- Pushing Issue #13 to next week
- Extending Phase 2 timeline by 3 days
```

### Pattern: "Critical" Report (🔴 RED)

```
Status: CRITICAL
✅ Completed 1 issue (major miss)
🔵 In Progress: [Critical issue is blocked]
Blockers: Multiple blocking issues preventing progress

Action Taken:
- Escalated to CTO
- Emergency meeting Friday 5 PM to replan
- Considering scope reduction for Phase 2
```

---

## Tools & Templates

### Template File

- Location: `/wiki/pmo-status-reports/WEEKLY_STATUS_TEMPLATE.md`
- Usage: Copy → Fill in → Commit → Post to Slack

### Example Reports

- Location: `/wiki/pmo-status-reports/`
- For reference: See past weeks' reports for examples

### Metrics Dashboard

- Location: [Link to Grafana/metrics dashboard]
- Used for: Quick metrics data (coverage, quality, etc.)

---

## FAQ

**Q: How long should the status report be?**
A: 1-2 pages typically. If it's >3 pages, you're including too much detail.

**Q: What if something changes between Friday and Monday meeting?**
A: Note it in the meeting notes. Next week's report should reflect any decisions made in Monday meeting.

**Q: Do we need to report on every single issue?**
A: No. Only report on open/closed issues that week. Issues not touched don't need update.

**Q: What if a blocker is unresolved at end of week?**
A: Escalate! Don't wait. Page the owner immediately if it blocks critical path.

**Q: Should we update the report after it's published?**
A: No, publish on Friday and lock it. Next week's report can correct any mistakes noted.

---

## Success Metrics for PMO Reporting

| Metric                    | Target                        | How to Measure                                 |
| ------------------------- | ----------------------------- | ---------------------------------------------- |
| **Report Accuracy**       | 95% (matches actual progress) | Compare report to actual closed issues         |
| **Report Completeness**   | 100% of open issues tracked   | All issues should appear in report             |
| **Blocker Resolution**    | <4 hours to fix               | Blocker marked resolved same day or next day   |
| **Timeline Adherence**    | 95% on-schedule delivery      | Actual vs. projected completion                |
| **Leadership Confidence** | High (no surprises)           | Leadership should never be surprised by status |

---

**Next Review**: 2026-02-26 (one month)
**Maintained By**: PMO
**Version**: 1.0
**Last Updated**: 2026-01-26
