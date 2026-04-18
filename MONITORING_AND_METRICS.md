# 📊 Monitoring & Metrics Dashboard Guide

**Framework:** Autonomous Issue Management System
**Version:** 1.0 (Deployed April 18, 2026)
**Audience:** Team leads, maintainers, framework operators

---

## 🎯 Dashboard Overview

The autonomous framework generates real-time metrics that enable data-driven decision-making and performance tracking. This guide shows how to access, understand, and act on the metrics.

## 📈 Available Metrics

### Real-Time Metrics (GitHub Actions)

**Updated:** Every workflow run
**Location:** GitHub Actions → Issue Triage / Batch Processor workflow

```
✅ Issues processed in last run
✅ Labels applied
✅ Triage accuracy
✅ Execution time
✅ API rate limit status
```

**How to access:**
1. Go to GitHub → Actions
2. Click "Issue Triage" or "Batch Issue Processor"
3. Select latest run
4. View "Summary" tab

### Daily Metrics

**Updated:** 2 AM UTC (automatic)
**Location:** `.github/issue_metrics_daily.json`

```json
{
  "date": "2026-04-18",
  "issues_created": 3,
  "issues_triaged": 3,
  "issues_closed": 0,
  "avg_labels_per_issue": 2.3,
  "categories": {
    "feature": 2,
    "bug": 1,
    "docs": 0
  },
  "sla_compliance": {
    "critical": 0.0,
    "high": 0.0,
    "medium": 0.0,
    "low": 0.0
  },
  "acceptance_criteria_compliance": 0.67,
  "triage_accuracy": 0.95
}
```

**What it tells you:**
- How many issues are being created
- Triage classification accuracy
- SLA compliance by severity
- Category distribution
- Process efficiency

**Access via CLI:**
```bash
cat .github/issue_metrics_daily.json | jq '.'
```

### Weekly Metrics

**Updated:** Fridays at 2 AM UTC
**Location:** `.github/issue_metrics_weekly.json`

```json
{
  "week_start": "2026-04-14",
  "week_end": "2026-04-20",
  "metrics": {
    "issues_created": 18,
    "issues_closed": 5,
    "issues_in_progress": 9,
    "avg_close_time_hours": 12.5,
    "category_distribution": {
      "feature": 8,
      "bug": 6,
      "docs": 4
    },
    "avg_labels_per_issue": 2.4,
    "triage_accuracy": 0.92,
    "developer_productivity": 0.85
  },
  "sla_metrics": {
    "critical_compliance": 0.95,
    "high_compliance": 0.88,
    "medium_compliance": 0.82,
    "low_compliance": 0.90
  },
  "code_quality": {
    "avg_coverage": 0.96,
    "tests_passing": 0.98,
    "lint_score": 0.99,
    "type_check_score": 0.97
  }
}
```

**What it tells you:**
- Weekly issue velocity
- Average resolution time (improving?)
- Category trends
- Developer productivity
- Code quality health

**How to interpret:**
- If `avg_close_time_hours` increasing → Blockers, need investigation
- If `avg_close_time_hours` decreasing → Process improving
- If coverage < 95% → Quality issue, address before merge
- If SLA compliance < 90% → Escalation needed

### Monthly Metrics

**Updated:** 1st of month at 2 AM UTC
**Location:** `.github/issue_metrics_monthly.json`

```json
{
  "month": "2026-04",
  "summary": {
    "issues_created": 87,
    "issues_closed": 64,
    "issues_pending": 23,
    "velocity_change": "+15%",
    "quality_trend": "↑ improving"
  },
  "team": {
    "active_contributors": 8,
    "avg_pr_review_time": "4.2 hours",
    "avg_pr_approval_time": "2.1 hours",
    "merge_success_rate": 0.96
  },
  "agents": {
    "autonomous_implementations": 12,
    "agent_success_rate": 0.92,
    "avg_implementation_time": "3.1 days",
    "avg_feedback_cycles": 1.8
  },
  "code_quality": {
    "coverage_trend": "↑ 94% → 96%",
    "test_suite_health": 0.99,
    "lint_passing": 0.99,
    "security_issues": 0
  },
  "process": {
    "automation_coverage": 0.87,
    "error_rate": 0.02,
    "escalation_rate": 0.05,
    "framework_utilization": 0.94
  }
}
```

**What it tells you:**
- Monthly velocity trend
- Team effectiveness
- Agent productivity and quality
- Process health
- Whether to scale up or adjust

### Audit Trail

**Updated:** Real-time (append-only)
**Location:** `.github/issue_audit_trail.jsonl`

Each line is one operation:
```json
{"timestamp": "2026-04-18T14:23:45Z", "event": "issue.opened", "issue_number": 321, "actor": "github_user", "action": "triaged", "labels_applied": ["feature", "high-priority"], "rationale": "New feature request with clear requirements"}
{"timestamp": "2026-04-18T14:24:12Z", "event": "issue.labeled", "issue_number": 321, "actor": "agent", "labels": ["ready"], "rationale": "Acceptance criteria complete, ready for implementation"}
{"timestamp": "2026-04-18T14:30:45Z", "event": "pr.opened", "pr_number": 456, "issue_number": 321, "actor": "autonomous_agent", "branch": "feature/321-new-feature"}
```

**Why it matters:**
- Compliance auditing
- Debugging automation decisions
- Performance analysis
- Accountability trail

**Access via CLI:**
```bash
# View recent operations
tail -20 .github/issue_audit_trail.jsonl | jq '.'

# Find all operations on issue #321
grep '"issue_number": 321' .github/issue_audit_trail.jsonl | jq '.'

# Find all agent operations
grep '"actor": "agent"' .github/issue_audit_trail.jsonl | jq '.'

# Operations by date
grep '2026-04-18' .github/issue_audit_trail.jsonl | wc -l
```

---

## 📊 Creating Custom Reports

### Dashboard #1: Team Velocity

**Shows:** How fast the team is closing issues

```bash
# Create velocity report
cat .github/issue_metrics_weekly.json | jq '.metrics | {
  issues_created,
  issues_closed,
  velocity: (.issues_closed / .issues_created)
}'
```

**Expected output:**
```json
{
  "issues_created": 18,
  "issues_closed": 14,
  "velocity": 0.78
}
```

**Interpretation:**
- velocity < 0.5: Team is creating issues faster than closing (backlog growing)
- velocity 0.5-0.8: Normal velocity (healthy)
- velocity > 0.8: Strong closure rate (good momentum)

### Dashboard #2: Code Quality

**Shows:** Whether code quality is meeting standards

```bash
# Extract code quality metrics
cat .github/issue_metrics_weekly.json | jq '.code_quality'
```

**Expected output:**
```json
{
  "avg_coverage": 0.96,
  "tests_passing": 0.98,
  "lint_score": 0.99,
  "type_check_score": 0.97
}
```

**Interpretation:**
- Coverage < 95%: 🚨 Block PRs until fixed
- Tests < 95%: 🚨 Investigate failing tests
- Lint < 98%: ⚠️ Address style issues
- Type check < 95%: ⚠️ Add type hints

### Dashboard #3: Agent Performance

**Shows:** How well autonomous agents are performing

```bash
# Extract agent metrics
cat .github/issue_metrics_monthly.json | jq '.agents'
```

**Expected output:**
```json
{
  "autonomous_implementations": 12,
  "agent_success_rate": 0.92,
  "avg_implementation_time": "3.1 days",
  "avg_feedback_cycles": 1.8
}
```

**Interpretation:**
- success_rate > 90%: Agents are capable and independent
- success_rate < 90%: Need guidance or simpler tasks
- feedback_cycles > 2: Agents need clearer requirements
- feedback_cycles < 2: Clear communication, working well

### Dashboard #4: SLA Compliance

**Shows:** Whether team responds to issues on time

```bash
# Extract SLA metrics
cat .github/issue_metrics_weekly.json | jq '.sla_metrics'
```

**Expected output:**
```json
{
  "critical_compliance": 0.95,
  "high_compliance": 0.88,
  "medium_compliance": 0.82,
  "low_compliance": 0.90
}
```

**Interpretation:**
- >= 95%: Excellent
- 90-95%: Good
- 85-90%: Acceptable (but watch)
- < 85%: 🚨 Need to address bottleneck

### Dashboard #5: Triage Accuracy

**Shows:** How well automatic classification is working

```bash
# Extract triage metrics
cat .github/issue_metrics_daily.json | jq '.triage_accuracy'
```

**Expected:** Should be >= 0.90 (90% correct)

**If low (<85%):**
1. Review recent auto-classified issues
2. Check audit trail for pattern
3. Adjust triage rules if needed
4. Re-test classification

---

## 🔍 Monitoring Checklist (Daily)

### Morning Standup (5 minutes)

```
□ Check daily metrics: .github/issue_metrics_daily.json
□ Verify SLA compliance: _metrics.sla_compliance
□ Count new issues: _metrics.issues_created
□ Any escalations? Search audit trail for "escalated"
□ Any failures? Check GitHub Actions logs
```

### If issues found:

```
Escalation > 5%:
  → Team might be overloaded
  → Plan capacity review
  → Adjust priorities

SLA compliance < 90%:
  → Bottleneck exists
  → Identify: blocked on review? complexity?
  → Create action item

New failures in workflows:
  → Check API rate limits
  → Verify GitHub token valid
  → Check for API changes
```

### Weekly Deep Dive (20 minutes)

**Every Friday morning:**

```bash
# 1. Generate this week's report
cat .github/issue_metrics_weekly.json | jq '.'

# 2. Compare to last week
# (manually compare numbers or use jq diff)

# 3. Check code quality trends
cat .github/issue_metrics_weekly.json | jq '.code_quality'

# 4. Identify issues needing attention
grep -i "escalate\|error\|failed" .github/issue_audit_trail.jsonl | tail -10

# 5. Note: Any trends? Improvements? Concerns?
```

### Monthly Review (1 hour)

**First week of month:**

```bash
# 1. Generate monthly report
cat .github/issue_metrics_monthly.json | jq '.'

# 2. Compare to previous month
# (trends: velocity, quality, agent performance)

# 3. Analyze agent performance
cat .github/issue_metrics_monthly.json | jq '.agents'

# 4. Review team metrics
cat .github/issue_metrics_monthly.json | jq '.team'

# 5. Check security and compliance
grep "security\|vulnerability" .github/issue_audit_trail.jsonl

# 6. Plan quarterly improvements
# Based on trends, what needs improvement?
```

---

## 🚨 Alert Rules & Actions

### Alert: SLA Compliance < 90%

**What it means:** Team is not responding quickly enough to some issues

**What to do:**
```
1. Check critical/high-priority counts
2. If > 5 high-priority open: Team overloaded
3. Move low-priority items to backlog
4. Add resources if possible
5. Communicate delay to stakeholders
```

### Alert: Code Coverage < 95%

**What it means:** New code doesn't have enough tests

**What to do:**
```
1. Block PR merge until coverage restored
2. Identify which files lack coverage
3. Request developer add tests
4. If complex: escalate for guidance
5. Don't compromise on quality
```

### Alert: Triage Accuracy < 85%

**What it means:** Automatic classification not working well

**What to do:**
```
1. Review recent auto-classified issues
2. Look for pattern in misclassification
3. Update triage rules if needed
4. Re-test with recent issues
5. Document reason for accuracy drop
```

### Alert: Agent Success Rate < 90%

**What it means:** Agents hitting blockers or not meeting standards

**What to do:**
```
1. Review failed agent implementations
2. Look for patterns: complexity? clarity?
3. Provide clearer requirements for next
4. Consider simpler tasks for practice
5. Offer guidance/training if needed
```

### Alert: Workflow Failures (any red X)

**What it means:** Automation is broken

**What to do:**
```
1. Check GitHub Actions logs immediately
2. Common issues:
   - API rate limit exceeded: Add delay or split runs
   - GitHub token invalid: Rotate and update
   - Network error: Retry (add exponential backoff)
3. Fix root cause
4. Re-run failed workflow
5. Document prevention
```

---

## 📈 Success Metrics (30-Day Targets)

| Metric | Baseline | 30-Day Target | Owner |
|--------|----------|---------------|-------|
| **Issues Triaged** | - | 100% | System |
| **Triage Accuracy** | - | >90% | System |
| **SLA Compliance** | - | >90% | Team |
| **Code Coverage** | - | >95% | Developers |
| **PR Review Time** | - | <24h | Maintainers |
| **Agent Success Rate** | - | >90% | Agents |
| **Velocity** | - | +30% | Team |
| **Quality Trend** | - | ↑ improving | Team |

---

## 🔧 Maintenance Tasks

### Weekly (Every Friday)

```bash
# Generate weekly report
python3 scripts/generate_metrics.py --type weekly --output .github/

# Review audit trail for anomalies
tail -100 .github/issue_audit_trail.jsonl | jq '.' | less

# Clean up old temp files
find .github -name "*.tmp" -delete
```

### Monthly (1st of month)

```bash
# Generate monthly report
python3 scripts/generate_metrics.py --type monthly --output .github/

# Archive old audit trail (keep 90 days)
find .github -name "*audit_trail*" -mtime +90 -delete

# Review and update governance rules
cat .github/issue-governance.iac.json | jq '.sla_targets'

# Team metrics review meeting
# (discuss results, plan improvements)
```

### Quarterly (Every 3 months)

```bash
# Full framework review
1. Review governance rules (still appropriate?)
2. Check triage accuracy (any changes needed?)
3. Assess process metrics (baseline established?)
4. Plan improvements for next quarter
5. Update roadmap based on learnings
```

---

## 📊 Metrics Interpretation Guide

### Velocity Metrics

```
Creating 20 issues/week, closing 15/week
→ Velocity: 75%
→ Backlog growing by 5/week
→ Normal: keep monitoring
→ If concerns: discuss priorities

Creating 20 issues/week, closing 5/week
→ Velocity: 25%
→ Backlog growing by 15/week
→ Problem: Need to focus on closure
→ Action: Reduce new WIP, focus on existing
```

### Quality Metrics

```
Coverage: 96% → Excellent (>95%)
Tests: 99% passing → Excellent (>95%)
Lint: 99% passing → Good (>98%)
Type check: 98% passing → Good (>95%)

Overall: ✅ HEALTHY
```

### SLA Metrics

```
Critical (< 1h): 95% → Excellent (>90%)
High (< 8h): 88% → Good (>85%)
Medium (< 24h): 82% → Watch (aim >90%)
Low (< 72h): 90% → Good (>85%)

Overall: ✅ HEALTHY (with one item to monitor)
```

### Agent Performance

```
Success rate: 92% → Good (>90%)
Feedback cycles: 1.8 → Normal (1-2 is good)
Implementation time: 3.1 days → Normal
Quality (coverage): 96% → Excellent (>95%)

Overall: ✅ HEALTHY - Agents are performing well
```

---

## 📞 Questions About Metrics?

**Q: Why is coverage sometimes below 95%?**
A: New code without tests yet. Should be fixed before merge.

**Q: What if an agent's implementation takes 7 days?**
A: Check feedback cycles - might be complex issue. Review and adjust.

**Q: Is 90% SLA compliance enough?**
A: For Month 1, yes. Plan to reach 95% by Month 2.

**Q: Should we increase the daily metrics limit?**
A: Adjust based on team size and capacity. Current: 50 issues/day.

**Q: How do I know if the framework is working?**
A: All green metrics (above targets) = framework working well.

---

## 🎯 Next Steps

1. **Today:** Bookmark this document
2. **Daily:** Check morning metrics (5 min)
3. **Weekly:** Full metrics review (20 min)
4. **Monthly:** Team discussion (1 hour)
5. **Quarterly:** Framework review and planning

---

**Framework Status:** ✅ Production Ready
**Metrics System:** ✅ Active
**Last Updated:** April 18, 2026

*This framework enables continuous visibility into team productivity, code quality, and autonomous agent performance. Use metrics to improve continuously.*
