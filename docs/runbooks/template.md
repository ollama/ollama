# Runbook Template: [Incident Type]

**Version**: 1.0
**Last Updated**: [YYYY-MM-DD]
**Maintained By**: [Team Name]
**Review Frequency**: Quarterly

---

## Overview

**Incident Type**: [Name]
**Severity Classification**: [SEV1/SEV2/SEV3]
**Estimated Time to Resolution**: [X minutes/hours]
**Last Occurrence**: [Date or "Never"]
**Frequency**: [Rare/Occasional/Frequent]

---

## Detection

### Alert Triggers

- **Primary Alert**: [Alert name] fires when [metric] > [threshold]
  - Example: `ollama_agent_hallucination_rate > 0.02`
  - Response time: Alert should fire within [X minutes]
  - False positive rate: [X]%

- **Secondary Signal**: [Alternative detection method]
  - Example: [Log pattern] appears in [log stream]
  - Example: [Dashboard metric] shows [unusual pattern]

- **Manual Detection**: [How to spot this manually]
  - Check: [Dashboard/metric/log]
  - Look for: [specific pattern]

### Confirmation Steps

1. Log into [system]
2. Navigate to [dashboard]
3. Check [metric] - should be [normal value]
4. If [abnormal indicator], proceed to Diagnosis

### Example Alert Configuration

```
alert: [AlertName]
  expr: [prometheus query]
  for: [duration]
  labels:
    severity: [critical/warning/info]
    component: [component]
  annotations:
    summary: "[Alert summary]"
    runbook: "[URL to this runbook]"
```

---

## Immediate Actions (0-5 Minutes)

**Goal**: Get visibility, declare incident, assemble team

### Step 1: Page On-Call Engineer ✅ AUTOMATIC

- **Trigger**: [Alert system] automatically pages via PagerDuty
- **Expected**: On-call engineer responds within 5 minutes
- **If not**: Escalate to secondary on-call

### Step 2: Create War Room 🔵 MANUAL (On-Call)

```bash
# Create Slack channel for incident
/slash create-channel #incident-[type]-[timestamp]

# Post initial status
@incident-commander: INCIDENT DECLARED
Severity: SEV[1/2/3]
Symptom: [What went wrong]
Potential Cause: [Quick hypothesis]
Runbook: [Link to this runbook]
ETA: [Estimated time to next update in 10 min]
```

### Step 3: Assess Scope 📊 MANUAL (On-Call)

Run these checks immediately:

```bash
# Check 1: Is this a real incident or false alarm?
curl https://elevatediq.ai/ollama/api/v1/health

# Check 2: How many customers/systems affected?
gcloud logging read 'severity=ERROR AND [component]' --limit=10

# Check 3: When did it start?
# Check the alert timestamp vs. first error in logs
gcloud logging read 'severity=ERROR' --limit=100 | grep timestamp

# Check 4: Is there active customer impact?
# Check Slack #customer-escalations and #support-tickets
```

### Step 4: Escalation Decision 🚨 MANUAL (On-Call)

- **If SEV1**: Immediately notify @engineering-lead, @cto, @on-call-backup
- **If SEV2**: Notify @engineering-lead, keep as standby
- **If SEV3**: Handle solo, document decision

### Step 5: Status Communication ✉️ MANUAL (On-Call)

Post in incident channel:

```
⚠️ INCIDENT IN PROGRESS
Severity: SEV[#]
Status: Investigating
Next Update: [Time + 10 min]
```

---

## Diagnosis (5-15 Minutes)

**Goal**: Understand what broke and why

### Diagnosis Checklist

#### Check 1: Verify the Alert ✅

```bash
# Confirm the metric that fired
# Example for hallucination detection:
gcloud monitoring time-series list \
  --filter='metric.type="custom.googleapis.com/ollama_agent_hallucination_rate"' \
  --format="table(metric.labels.agent_id, points[0].value.double_value)"

# Check last 60 minutes
# Should show spike at incident start time
```

#### Check 2: Check System Health 📊

```bash
# Check 1: API Health
curl -H "Authorization: Bearer $API_KEY" \
  https://elevatediq.ai/ollama/api/v1/health

# Response should be: {"status": "healthy", "timestamp": "..."}
# If not: System is DOWN, go to "Remediation" section

# Check 2: Database Health
# Connect to prod database
psql $PROD_DATABASE_URL -c "SELECT 1"

# Should return: 1 (success)

# Check 3: Cache Health (Redis)
redis-cli -h redis.prod ping

# Should return: PONG

# Check 4: Model Service Health
curl http://ollama:11434/api/tags

# Should list available models
```

#### Check 3: Check Recent Changes 🔍

```bash
# What changed in the last 30 minutes?
git log --oneline --since="30 minutes ago"

# Any recent deployments?
gcloud run services describe ollama-api --region=us-central1

# Any recent model updates?
ollama list  # This shows current model versions
```

#### Check 4: Analyze Logs 📋

```bash
# Pull relevant logs
gcloud logging read '[filter]' --limit=50 --format=json

# Examples:
# For agent hallucination:
gcloud logging read 'severity=ERROR AND resource.labels.function_name="ollama-inference"' \
  --since="20m" --limit=50

# For database issues:
gcloud logging read 'protoPayload.methodName="cloudsql.instances.connect"' \
  --since="20m" --limit=50

# For deployment issues:
gcloud logging read 'resource.type="cloud_run_revision"' \
  --since="20m" --limit=50

# Look for patterns:
# 1. First error timestamp (matches alert time?)
# 2. Error message (gives clue to root cause?)
# 3. Which component (agent, API, DB, model)?
```

#### Check 5: Review Dashboards 📈

Navigate to each dashboard:

- **Main Metrics Dashboard**: [URL]
  - Look for: Any red/orange lights?
  - Compare to: Baseline from yesterday

- **Distributed Tracing (Jaeger)**: [URL]
  - Search: Traces from incident start time
  - Look for: Spike in error rate or latency

- **Logs Dashboard (Cloud Logging)**: [URL]
  - Search: ERROR level logs from incident time
  - Look for: Patterns in errors

- **GCP Status Dashboard**: [URL]
  - Check: Any GCP service outages affecting us?

### Likely Diagnoses (Decision Tree)

**Is the API returning 5xx errors?**

- YES → Go to Remediation, Section A (API Crash)
- NO → Go to next question

**Is the database connected?**

- YES → Go to next question
- NO → Go to Remediation, Section B (Database Down)

**Are model inference requests timing out?**

- YES → Go to Remediation, Section C (Inference Timeout)
- NO → Go to next question

**Are agent responses hallucinating?**

- YES → Go to Remediation, Section D (Hallucination Detected)
- NO → Go to Remediation, Section E (Unknown - Escalate)

---

## Remediation (15+ Minutes)

**Goal**: Stop the bleeding and restore service

### Section A: API Crash Recovery

**Symptoms**:

- GET/POST requests returning 500 errors
- Cloud Run service showing errors in logs
- Error message: `[specific error]`

**Resolution**:

```bash
# Step 1: Check Cloud Run service status
gcloud run services describe ollama-api --region=us-central1

# Look for: Status should be "Active"
# If status is "Error": Service failed to deploy

# Step 2: Restart the service
gcloud run services update-traffic ollama-api --to-revisions LATEST=100 --region=us-central1

# Step 3: Wait for rollout (takes 2-3 minutes)
watch gcloud run services describe ollama-api --region=us-central1

# Step 4: Verify health
curl https://elevatediq.ai/ollama/api/v1/health

# Should return: {"status": "healthy"}

# If still failing: Proceed to rollback (Section F)
```

### Section B: Database Connection Pool Exhausted

**Symptoms**:

- All database queries timing out
- Error: `FATAL: remaining connection slots reserved for non-replication superuser connections`
- API errors after database queries

**Resolution**:

```bash
# Step 1: Check current connections
psql $PROD_DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"

# Should show: [count] connections
# Threshold: If > 95 connections, pool is exhausted

# Step 2: Identify long-running queries
psql $PROD_DATABASE_URL -c "SELECT pid, usename, state, query FROM pg_stat_activity WHERE state != 'idle';"

# Look for: Queries running for >30 seconds

# Step 3: Kill long-running queries
psql $PROD_DATABASE_URL -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '30 seconds';"

# Step 4: Restart connection pool
# (May require restarting FastAPI service - see Rollback section)

# Step 5: Verify connections returned to normal
psql $PROD_DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"

# Should show: < 50 connections
```

### Section C: Model Inference Timeout

**Symptoms**:

- POST /api/v1/generate hangs for >30 seconds
- Model service is running but slow
- GPU memory usage high

**Resolution**:

```bash
# Step 1: Check model service health
curl http://ollama:11434/api/tags

# Step 2: Check GPU memory usage
nvidia-smi  # If GPU available
# OR
gcloud compute instances describe [instance] --zone=[zone] | grep acceleratorCount

# Step 3: Check model size vs. available memory
# Symptom: GPU out of memory → swap to CPU → slow inference

# Option A: Restart model service (clears GPU memory)
kubectl restart deployment ollama-inference -n production

# Step 4: Verify inference performance
curl -X POST http://ollama:11434/api/generate \
  -d '{"model": "llama3.2", "prompt": "Hello", "stream": false}'

# Should complete in < 30 seconds

# If still slow: May need to reduce batch size or use quantized model
```

### Section D: Hallucination Detected in Production

**Symptoms**:

- Agent responses factually incorrect
- Hallucination rate > 2% (threshold)
- Alert: `ollama_agent_hallucination_rate > 0.02`

**Resolution**:

```bash
# Step 1: Identify affected agent
# From alert: Should show agent_id label

# Step 2: Sample recent outputs
# Check: Last 10 agent responses
gcloud logging read 'protoPayload.request.model="[agent_id]"' --limit=10 --format=json | grep output

# Step 3: Assess: Is this new hallucination or expected output?
# For each output, ask: "Is this factually accurate?"
# If >2% are hallucinating: Proceed

# Step 4: Rollback to previous model version
# Fast option: Deploy previous container revision
gcloud run services update-traffic ollama-api --to-revisions [PREVIOUS_REVISION]=100 --region=us-central1

# Verify: Hallucination rate drops back to <0.5%

# Step 5: Investigate root cause
# Questions to ask:
# - What changed in model/prompt/validation?
# - Did training data change?
# - Did prompt template change?
# - Is there a new user behavior pattern?

# Step 6: Post-incident
# Create postmortem issue: YYYY-MM-DD-hallucination-incident
# Root cause analysis + prevention measures
```

### Section E: Unknown Issue - Escalate

**If none of the above diagnoses match**:

1. **Gather context**:

   ```bash
   # Collect all diagnostic data
   gcloud logging read '[error-filter]' --limit=100 > incident-logs.json
   gcloud monitoring time-series list --limit=100 > incident-metrics.json
   gcloud run services describe ollama-api > service-status.json
   ```

2. **Page CTO**:

   ```
   @cto: Need escalation on incident [type]
   Logs: [shared above]
   Metrics: [shared above]
   Diagnosis: Unknown - symptoms don't match playbook
   Request: Expert analysis and decision
   ```

3. **Continue troubleshooting** while waiting for escalation

---

### Section F: Rollback Procedure (When All Else Fails)

**Use only if**:

- Issue is blocking all traffic
- Root cause unknown
- Quicker to rollback than fix

**Process**:

```bash
# Step 1: Identify previous working revision
gcloud run revisions list ollama-api --region=us-central1 --limit=5

# Should show: [REVISION_NAME] with timestamp
# Choose: Most recent one without errors in logs

# Step 2: Rollback to previous revision
gcloud run services update-traffic ollama-api \
  --to-revisions [PREVIOUS_REVISION_NAME]=100 \
  --region=us-central1

# Step 3: Wait for traffic to shift (30-60 seconds)
sleep 30

# Step 4: Verify health
curl https://elevatediq.ai/ollama/api/v1/health

# Should return: {"status": "healthy"}

# Step 5: Monitor error rate
# Should drop to <1% within 1 minute

# Step 6: Create post-incident
# Issue: "Incident: Rollback to [revision] - Investigate"
# Owner: @engineering-lead
# Next: Root cause analysis
```

---

## Escalation Criteria

**Escalate to Engineering Lead if**:

- Incident not resolved within 15 minutes
- Customer-facing data loss detected
- Multiple systems affected simultaneously
- Unclear root cause after 10 minutes diagnosis

**Escalate to CTO if**:

- Incident not resolved within 30 minutes
- Revenue impact or customer churn risk
- Security vulnerability involved
- Need for architectural decision

**Escalate to Founders if**:

- Incident not resolved within 60 minutes
- Major customer at risk
- Public reputation damage
- Need for emergency resource allocation

---

## Post-Incident

### Immediate (Same Day)

- [ ] Create postmortem issue: GitHub issue #[auto-number]
  - Title: `[POSTMORTEM] YYYY-MM-DD - [Incident Type]`
  - Link to this runbook
  - Link to war room notes/transcript

- [ ] Post summary in #incident-postmortems
  - Duration: [X minutes/hours]
  - Root cause: [Brief summary]
  - Prevention: [Brief summary]

### Within 24 Hours

- [ ] Complete postmortem document using `/incidents/POSTMORTEM_TEMPLATE.md`
- [ ] Assign action items with owners and due dates
- [ ] Update this runbook if new learnings
- [ ] Schedule postmortem meeting for team review (optional for SEV3, mandatory for SEV1/SEV2)

### Within 1 Week

- [ ] Complete all immediate action items
- [ ] Document new procedures or updates to existing runbooks
- [ ] Update monitoring/alerts if needed

---

## Related Resources

- **Postmortem Template**: `/incidents/POSTMORTEM_TEMPLATE.md`
- **Monitoring Dashboard**: [URL]
- **On-Call Handoff Doc**: [URL]
- **Escalation Contact List**: [URL]
- **Related Incidents**: [Links to similar past incidents]

---

## Change Log

| Version | Date       | Changes          | Author  |
| ------- | ---------- | ---------------- | ------- |
| 1.0     | YYYY-MM-DD | Initial template | @[name] |
| [Next]  | [Date]     | [Changes]        | @[name] |

---

**Last Reviewed**: [Date]
**Next Review Due**: [Date + 3 months]
**Reviewer**: @[engineering-lead]
