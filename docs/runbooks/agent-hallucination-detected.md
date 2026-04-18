# Runbook: Agent Hallucination Detected in Production

**Version**: 1.0
**Severity**: SEV2
**Last Occurrence**: Never
**Time to Resolution**: 15 minutes

---

## Detection

**Alert**: `ollama_agent_hallucination_rate > 0.02`
**Dashboard**: [Metrics Dashboard - Hallucination Rate](https://grafana.example.com)
**False Positive Rate**: <5%

### Confirmation

1. Check Grafana: Agent hallucination rate spike
2. Sample recent outputs: Are responses factually incorrect?
3. Check timeline: When did spike start?

---

## Immediate Actions (0-5 min)

1. **Create war room**: `#incident-hallucination-[timestamp]`
2. **Post initial status**: "Hallucination spike detected at [time], agent [agent_id] affected"
3. **Assess scope**: How many responses? Which agent versions?
4. **Check logs**: Recent model updates or prompt changes?

---

## Diagnosis (5-15 min)

```bash
# Which agent is hallucinating?
gcloud logging read 'metric.type="custom.googleapis.com/ollama_agent_hallucination_rate"' \
  --format="table(metric.labels.agent_id, points[0].value.double_value)" --limit=5

# How many outputs affected?
gcloud logging read 'severity=WARNING AND "hallucination"' --limit=20

# When did it start?
# Compare to deployment logs - did something deploy in last 30 min?
gcloud run services describe ollama-api --region=us-central1
```

---

## Remediation

### Option A: Rollback Model (< 5 min)

```bash
# Get previous revision
gcloud run revisions list ollama-api --region=us-central1 --limit=3

# Rollback
gcloud run services update-traffic ollama-api \
  --to-revisions [PREVIOUS_REVISION]=100 --region=us-central1

# Verify hallucination rate drops
watch gcloud monitoring time-series list \
  --filter='metric.type="custom.googleapis.com/ollama_agent_hallucination_rate"'
```

### Option B: Disable Problem Agent (< 2 min)

```bash
# If only one agent is hallucinating:
gcloud firestore update documents/agents/[agent_id] \
  --update-mask="enabled=false"

# Responses will route to backup agent
# Verify: Traffic shifts to [backup-agent]
```

### Option C: Update Agent Parameters (< 10 min)

```bash
# Reduce temperature (less randomness)
gcloud firestore update documents/agents/[agent_id] \
  --update-mask="temperature=0.3"  # from 0.7

# Lower max_tokens (shorter responses)
gcloud firestore update documents/agents/[agent_id] \
  --update-mask="max_tokens=256"  # from 512

# Verify: Next 10 responses are more accurate
```

---

## Escalation

- **If not resolved in 10 min**: Escalate to @ml-engineer + @engineering-lead
- **If affects multiple agents**: Escalate to @cto
- **If persists >30 min**: May need model retraining

---

## Post-Incident

1. Create postmortem: `/incidents/YYYY-MM-DD-hallucination.md`
2. Questions to answer:
   - What changed? (Model, prompt, data?)
   - Why didn't testing catch this?
   - How do we prevent recurrence?
3. Add adversarial test case to prevent this specific hallucination
4. Update monitoring sensitivity if needed

---

**Created**: 2026-01-26
**Last Updated**: 2026-01-26
