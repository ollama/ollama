# Postmortem Template

**Incident ID**: [YYYY-MM-DD]-[incident-type]
**Severity**: SEV1 | SEV2 | SEV3
**Report Date**: [YYYY-MM-DD]
**Duration**: [Start Time] - [End Time] ([minutes/hours])

---

## Executive Summary

**What Happened**: [2-3 sentence description of the incident]

**Impact**:

- Users Affected: [number] ([percentage] of user base)
- Data Loss: Yes/No ([describe if yes])
- Systems Down: [list of affected systems]
- Revenue Impact: $[estimated loss] or [percentage loss]

**Root Cause**: [1-sentence summary of root cause]

**Timeline**: Incident lasted [X minutes/hours]. Detection was [X minutes/hours] after start.

---

## Incident Timeline

**15-Minute Granularity**:

| Time  | Event                               | Owner             | Notes                                                           |
| ----- | ----------------------------------- | ----------------- | --------------------------------------------------------------- |
| HH:MM | Incident starts (trigger: [metric]) | N/A               | Hallucination rate spike detected in production                 |
| HH:MM | Alert fires                         | On-call           | Prometheus alert: hallucination_rate > 0.02                     |
| HH:MM | Initial response                    | @engineer-on-call | Checked dashboards, confirmed hallucination in agent responses  |
| HH:MM | War room opened                     | @team-lead        | Created #incident-hallucination Slack channel                   |
| HH:MM | Root cause identified               | @ml-engineer      | Agent model had prompt injection vulnerability                  |
| HH:MM | Mitigation started                  | @engineer         | Deployed previous model version as hotfix                       |
| HH:MM | Mitigation verified                 | @qa               | Tested rollback, confirmed hallucination rate returned to <0.5% |
| HH:MM | Incident declared resolved          | @engineering-lead | All metrics normal, customers notified                          |

---

## Root Cause Analysis

### What We Think Happened

**Layer 1: Immediate Cause**
[Technical description of what broke]

**Layer 2: Contributing Factors**

- Factor 1: [description] — could have been caught if [process]
- Factor 2: [description] — would have been prevented by [monitoring]
- Factor 3: [description] — required [procedure that didn't exist]

**Layer 3: Systemic Issues**

- We didn't test for [scenario]
- We didn't monitor [metric]
- We didn't document [procedure]
- We didn't have [runbook]

### Why It Happened

[5-Whys Analysis]:

1. **Why did the incident occur?** → Because [technical cause]
2. **Why did that technical cause exist?** → Because [process failure]
3. **Why did that process failure exist?** → Because [missing procedure]
4. **Why was that procedure missing?** → Because [root cause]
5. **Why was that root cause never fixed?** → Because [systemic issue]

---

## Impact Assessment

### Customer Impact

- **Users Affected**: [number]
- **Duration**: [minutes/hours]
- **Service Degradation**: Complete outage | Partial degradation | Performance impact
- **Data Loss**: Yes/No — if yes, describe what was lost
- **SLA Breach**: Yes/No — if yes, list affected SLAs

### Business Impact

- **Revenue Impact**: $[amount] direct loss, $[amount] projected churn
- **Reputation Impact**: [number] customer complaints, [number] Twitter mentions
- **Security Impact**: Was any customer data exposed? Any new vulnerabilities discovered?

### Internal Impact

- **On-Call Engineer**: [number] hours response time
- **Engineering Team**: [number] engineers engaged, [time] spent on mitigation
- **Leadership**: Escalation to CTO/CEO required? Yes/No

---

## Lessons Learned

### What We Did Well

1. **Fast Detection** → Alert fired 3 minutes after incident start
   - Why: Prometheus monitoring on hallucination rate was working
   - Continue: Keep this monitoring active and improve sensitivity

2. **Clear Communication** → War room opened immediately, status updated every 5 minutes
   - Why: On-call engineer followed escalation protocol
   - Continue: Maintain this communication standard in future incidents

3. **Effective Mitigation** → Rollback completed in 15 minutes
   - Why: Previous version was staged in registry and tested
   - Continue: Maintain staging environment for rollbacks

### What We'll Do Better

1. **Better Testing** → Agent responses weren't tested for prompt injection
   - Action: Add adversarial prompt testing to pre-deployment validation
   - Owner: @ml-engineer
   - Timeline: By [DATE]
   - Acceptance: 100+ adversarial test cases passing

2. **Better Monitoring** → We detected hallucination in production, not testing
   - Action: Add hallucination detection to staging smoke tests
   - Owner: @qa-engineer
   - Timeline: By [DATE]
   - Acceptance: Smoke test failures prevent deployment

3. **Better Documentation** → On-call engineer wasn't sure about rollback procedure
   - Action: Create runbook: `/docs/runbooks/agent-hallucination-detected.md`
   - Owner: @engineering-lead
   - Timeline: By [DATE]
   - Acceptance: All on-call engineers trained and acknowledge

---

## Action Items

### Immediate (Due [DATE])

- [ ] **Fix hallucination detection in model** → Assigned to @ml-engineer
  - Details: Add input validation to detect and reject prompt injection attempts
  - Acceptance: No hallucination detected in 1000-sample adversarial test set

- [ ] **Update on-call runbook** → Assigned to @engineering-lead
  - Details: Document exact rollback procedures with screenshots
  - Acceptance: New on-call engineer can execute rollback in <5 minutes

### Short-Term (Due [DATE + 2 WEEKS])

- [ ] **Add adversarial testing to CI/CD** → Assigned to @qa-engineer
  - Details: Create test suite with 100+ adversarial prompts
  - Acceptance: CI/CD blocks deployment if any test fails

- [ ] **Improve monitoring sensitivity** → Assigned to @data-engineer
  - Details: Lower hallucination alert threshold from 2% to 1%
  - Acceptance: Alert fires within 5 minutes of hallucination spike

### Long-Term (Due [DATE + 6 WEEKS])

- [ ] **Create agent robustness framework** → Assigned to @ml-team
  - Details: Design comprehensive agent testing suite for injection attacks, hallucination, etc.
  - Acceptance: Framework covers 50+ adversarial scenarios

- [ ] **Build knowledge base of incident patterns** → Assigned to @platform-team
  - Details: Document all agent failure modes and detection strategies
  - Acceptance: Internal wiki page with 10+ documented patterns

---

## Monitoring & Prevention

### What We'll Monitor Going Forward

| Metric                    | Threshold             | Alert                      | Owner          |
| ------------------------- | --------------------- | -------------------------- | -------------- |
| Hallucination Rate        | >1% (lowered from 2%) | Page on-call in 5 min      | @data-engineer |
| Action Accuracy           | <90% (new metric)     | Page team lead             | @qa-engineer   |
| Model Response Latency    | >10s p95 (new metric) | Log event, don't alert yet | @ml-engineer   |
| Failed Prompt Validations | >5/hour (new metric)  | Log event, don't alert yet | @ml-engineer   |

### How We'll Prevent This in the Future

1. **Adversarial Testing** (By [DATE])
   - Run 100+ prompt injection tests against model
   - Include in pre-deployment validation

2. **Staging Environment** (By [DATE])
   - Deploy all models to staging for 24 hours before production
   - Run smoke tests including adversarial scenarios
   - Require explicit approval before prod promotion

3. **Rate Limiting** (By [DATE])
   - Implement per-agent rate limiting on unusual prompt patterns
   - Alert on repeated similar prompts from same user

4. **Input Validation** (By [DATE])
   - Validate prompts for injection attack patterns
   - Block suspicious prompts before sending to model

---

## Runbooks & Related Documentation

- **Runbook**: `/docs/runbooks/agent-hallucination-detected.md` (create if doesn't exist)
- **Related ADR**: ADR-004: Agent Safety Boundaries and Validation
- **Related Incident**: [Link to similar past incident if exists]
- **Wiki Article**: Agent Security Best Practices

---

## Sign-Off

| Role             | Name    | Date   | Signature  |
| ---------------- | ------- | ------ | ---------- |
| On-Call Engineer | @[name] | [DATE] | [Approved] |
| Engineering Lead | @[name] | [DATE] | [Approved] |
| CTO              | @[name] | [DATE] | [Approved] |

**Meeting Notes**: [Link to war room notes or recording if recorded]

**Postmortem Owner**: [Name] — responsible for driving action items to completion

---

## Action Item Tracking

| Item # | Action                                | Owner          | Due Date | Status  | Notes                   |
| ------ | ------------------------------------- | -------------- | -------- | ------- | ----------------------- |
| 1      | Add hallucination detection model fix | @ml-engineer   | [DATE]   | 🔵 TODO | Blocked by [if blocked] |
| 2      | Create hallucination runbook          | @eng-lead      | [DATE]   | 🔵 TODO |                         |
| 3      | Add adversarial tests to CI/CD        | @qa-engineer   | [DATE]   | 🔵 TODO |                         |
| 4      | Lower monitoring threshold to 1%      | @data-engineer | [DATE]   | 🔵 TODO |                         |
| 5      | [Additional actions from discussion]  | [@owner]       | [DATE]   | 🔵 TODO |                         |

---

## Appendix A: Detailed Timeline

[Minute-by-minute log if needed for complex incidents]

---

## Appendix B: Metrics During Incident

[Graphs/screenshots of metrics showing incident period]

---

## Appendix C: Communication Log

[Screenshots of Slack/email communications during incident]

---

**Document Version**: 1.0
**Last Updated**: [YYYY-MM-DD]
**Review Schedule**: Reviewed at [date] team meeting
**Next Incident Review**: [DATE when similar incident is expected to be fully prevented]
