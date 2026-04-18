# 📋 Post-Incident Review (PIR) Template

**Incident ID**: PIR-2026-XXX
**Date**: [Date]
**Duration**: [Start Time] - [End Time] (X minutes)
**Severity**: [P1/P2/P3]

---

## Executive Summary

**What happened**: [1-2 sentence description of the incident]

**Impact**:
- Users affected: [X] users
- Service degradation: [X]%
- Revenue impact: $[X]
- Reputation impact: [Low/Medium/High]

**Time to resolution**: [X minutes]
**Time to detection**: [X minutes]
**Time to mitigation**: [X minutes]

**Outcome**: [Resolved/Mitigated/Escalated]

---

## Timeline

| Time | Event | Owner | Notes |
|------|-------|-------|-------|
| 12:00 | Error rate spike detected | Monitoring | Alert triggered |
| 12:02 | On-call engineer paged | Pager Duty | Response time: 2 min |
| 12:05 | Root cause identified | Engineer | Database connections exhausted |
| 12:08 | Mitigation applied | Engineer | Restarted connection pool |
| 12:12 | Service recovered | Engineer | Error rate normalized |
| 12:15 | All-clear signal given | Engineer | Monitoring verified |

---

## Root Cause Analysis

### Primary Cause
**What was the underlying issue?**

[Detailed description of what actually failed]

**Why did it happen?**
- Contributing factor 1
- Contributing factor 2
- Contributing factor 3

**Detection: Why wasn't this caught sooner?**
- Alert threshold too high
- Monitoring gap
- No proactive check in place

### Timeline of Failure
```
1. [Time] - Initial condition exists (not visible)
2. [Time] - First symptom appears (metric exceeds threshold)
3. [Time] - Alert fires (but escalation delayed)
4. [Time] - User impact begins (customers see errors)
5. [Time] - On-call response (diagnosis begins)
6. [Time] - Root cause identified (clear action taken)
7. [Time] - Mitigation applied (system recovers)
```

---

## Impact Assessment

### User Impact
- **Total users affected**: X
- **Critical functions down**: [Yes/No]
- **Data loss**: [Yes/No - if yes, describe recovery]
- **Customer complaints**: X tickets
- **SLA breach**: [Yes/No]

### Business Impact
- **Revenue lost**: $X
- **Reputation damage**: [Describe]
- **Compliance issues**: [Yes/No]

### System Impact
- **Services affected**: [List]
- **Dependencies impacted**: [List]
- **Cascading failures**: [Yes/No - if yes, describe]

---

## Mitigation & Resolution

### Immediate Actions Taken
1. [Action 1] - By [Owner] - At [Time] - [Result]
2. [Action 2] - By [Owner] - At [Time] - [Result]
3. [Action 3] - By [Owner] - At [Time] - [Result]

### Why This Fixed It
[Explanation of how these actions resolved the incident]

### Alternative Solutions Considered
1. [Alternative 1] - Pros: [P1, P2] - Cons: [C1, C2] - Rejected because: [reason]
2. [Alternative 2] - Pros: [P1, P2] - Cons: [C1, C2] - Rejected because: [reason]

---

## Preventive Measures

### Short-term Fixes (Next 24-48 hours)

| Fix | Priority | Owner | Deadline | Status |
|-----|----------|-------|----------|--------|
| Lower alert threshold from X to Y | P1 | Engineer Name | Jan 14, 2026 | [Todo/In-Progress/Done] |
| Increase connection pool size to Z | P1 | DBA | Jan 14, 2026 | [Todo/In-Progress/Done] |
| Add dashboard metric for early detection | P2 | Engineer Name | Jan 15, 2026 | [Todo/In-Progress/Done] |
| Document in runbook | P2 | Engineer Name | Jan 15, 2026 | [Todo/In-Progress/Done] |

### Medium-term Improvements (1-4 weeks)

| Improvement | Description | Owner | Target Date | Status |
|------------|-------------|-------|-------------|--------|
| Implement auto-scaling | Automatically increase instances on high load | Platform Team | Jan 27, 2026 | [Todo/In-Progress/Done] |
| Add health checks | Implement proactive health checks | DevOps | Jan 27, 2026 | [Todo/In-Progress/Done] |
| Improve monitoring | Add predictive alerting for trending issues | SRE | Jan 27, 2026 | [Todo/In-Progress/Done] |
| Capacity planning | Audit current capacity vs. expected growth | Engineering | Feb 10, 2026 | [Todo/In-Progress/Done] |

### Long-term Preventive Measures (1-3 months)

| Measure | Description | Owner | Target Date | Estimated Effort |
|---------|-------------|-------|-------------|-----------------|
| Architecture redesign | Refactor connection pooling architecture | Platform Team | Mar 13, 2026 | 40 hrs |
| Load testing | Implement regular load testing | QA | Feb 24, 2026 | 20 hrs |
| Chaos engineering | Add chaos engineering to test suite | Platform Team | Mar 13, 2026 | 30 hrs |
| Incident response automation | Build auto-remediation for common issues | Platform Team | Mar 13, 2026 | 60 hrs |

---

## Lessons Learned

### What Went Well
- ✅ [Good thing 1 - quick response from team]
- ✅ [Good thing 2 - clear communication channels]
- ✅ [Good thing 3 - monitoring caught it quickly]

### What Could Be Better
- ⚠️ [Issue 1 - alert threshold was too loose]
- ⚠️ [Issue 2 - no runbook existed for this scenario]
- ⚠️ [Issue 3 - communication was delayed]

### Surprising Findings
- 🔍 [Finding 1 - customer impact was higher than expected]
- 🔍 [Finding 2 - underlying cause was different than suspected]
- 🔍 [Finding 3 - we discovered secondary issue while investigating]

### Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Create/update runbook for this scenario | [Owner] | [Date] | [Status] |
| Review alert threshold with team | [Owner] | [Date] | [Status] |
| Schedule training on new tools/procedures | [Owner] | [Date] | [Status] |
| Test failover procedure | [Owner] | [Date] | [Status] |

---

## Contributing Factors

### Technical Factors
- [ ] Inadequate monitoring
- [ ] Missing alerting
- [ ] Poor error handling
- [ ] Resource constraints (CPU/Memory/Disk)
- [ ] Database connection issues
- [ ] Third-party service failure
- [ ] Configuration error
- [ ] Code bug
- [ ] Infrastructure issue

### Process Factors
- [ ] Incomplete runbook
- [ ] Slow communication
- [ ] Unclear responsibilities
- [ ] Lack of automation
- [ ] No playbook for scenario

### Human Factors
- [ ] Fatigue/tired on-call engineer
- [ ] Lack of training
- [ ] Knowledge gap
- [ ] Miscommunication
- [ ] Time pressure

---

## Specific Improvement Recommendations

### If caused by monitoring gap:
```
RECOMMENDATION: Implement metric X tracking [specific behavior]
RATIONALE: This would have caught the issue Y minutes earlier
IMPLEMENTATION: Add histogram in app/monitoring/metrics.py
TESTING: Add test case in tests/unit/test_metrics.py
DOCUMENTATION: Update docs/monitoring.md
ESTIMATED EFFORT: X hours
SUCCESS METRICS: Alert fires within Y minutes of issue
```

### If caused by capacity issue:
```
RECOMMENDATION: Increase [resource] from X to Y
RATIONALE: Current capacity insufficient for peak load
IMPLEMENTATION: Update docker-compose.yml and k8s manifests
TESTING: Run load tests to verify new capacity
DOCUMENTATION: Update DEPLOYMENT.md with new limits
ESTIMATED EFFORT: X hours
SUCCESS METRICS: No resource exhaustion under peak load
```

### If caused by automation gap:
```
RECOMMENDATION: Implement auto-remediation for [scenario]
RATIONALE: Manual intervention took X minutes; automation could reduce to Y
IMPLEMENTATION: Add script in scripts/auto-remediation/
TESTING: Run integration tests with all scenarios
DOCUMENTATION: Add to operational runbooks
ESTIMATED EFFORT: X hours
SUCCESS METRICS: Incident resolved within 5 minutes automatically
```

---

## Follow-up Questions for Team

1. **Was the root cause the first problem we suspected?** How did we verify?
2. **Could this happen again?** What's our confidence level?
3. **Are there similar issues elsewhere in our system?** How do we audit for them?
4. **Is our monitoring/alerting effective?** What would we change?
5. **Did our runbooks/procedures help?** What was missing?
6. **What surprised us?** Why weren't we expecting it?

---

## Metrics & Statistics

### This Incident vs. Historical Average
| Metric | This Incident | Average | Better/Worse |
|--------|---------------|---------|-------------|
| Detection time | [X min] | [Y min] | [+/- Z%] |
| Response time | [X min] | [Y min] | [+/- Z%] |
| Resolution time | [X min] | [Y min] | [+/- Z%] |
| User impact | [X min] | [Y min] | [+/- Z%] |
| Escalations | [X] | [Y] | [+/- Z%] |

### Trend Analysis
- **Incident frequency**: [Increasing/Decreasing/Stable]
- **Average resolution time**: [Improving/Degrading]
- **Root cause patterns**: [List common causes]
- **MTBF (Mean Time Between Failures)**: X hours
- **MTTR (Mean Time To Recovery)**: X minutes

---

## Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Incident Commander | [Name] | [Date] | [Signature] |
| Team Lead | [Name] | [Date] | [Signature] |
| VP Engineering | [Name] | [Date] | [Signature] |

---

## Document Metadata

- **Document Status**: [Draft/Review/Final/Archived]
- **Last Updated**: [Date]
- **Next Review Date**: [Date + 30 days]
- **Archive Date**: [Date + 6 months if stable]
- **Owner**: [Incident Commander]
- **Reviewers**: [Team members who reviewed]

---

## Additional Resources

- **Incident Tickets**: [Link to Jira/GitHub issue]
- **Monitoring Dashboards**: [Links]
- **Related Runbooks**: [Links]
- **Previous Similar Incidents**: [Links to other PIRs]
- **Slack Channel**: [Link to incident-postmortem channel]

---

## Notes Section

[Space for additional notes, observations, or follow-up items]

---

## Distribution

This PIR should be shared with:
- [ ] On-call rotation
- [ ] Team leads
- [ ] VP Engineering
- [ ] Customer success team (if external impact)
- [ ] Security team (if security-related)
- [ ] All engineering team (wiki/knowledge base)

---

**Thank you for the detailed incident response and for contributing to continuous improvement!**
