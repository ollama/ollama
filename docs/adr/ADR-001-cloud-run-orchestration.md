# ADR-001: Cloud Run for Agent Orchestration

**Status**: Accepted
**Date**: 2026-01-26
**Author**: @[architecture-team]

---

## Context

### Problem

We need to deploy and scale agent inference services in production. Requirements:

- Support 100+ concurrent requests per second
- Auto-scale based on demand (0-1000 req/s spikes)
- Deploy new agent versions in < 5 minutes
- No ops team needed (serverless)
- Cost-effective (pay only for usage)

### Constraints

- Must run on Google Cloud Platform (GCP)
- Must support Python 3.11+ with dependencies (PyTorch, Ollama)
- Must be deployed within 2 weeks
- Must cost < $10k/month at current traffic levels

### Scope

- Agent inference execution (what we chose for)
- CI/CD integration
- Monitoring and logging
- NOT in scope: Model training, batch processing

---

## Decision

**Chosen**: Google Cloud Run (fully managed serverless)

We selected Cloud Run over alternatives because it:

1. **Zero ops burden**: Automatic scaling, patching, infrastructure management
2. **Cost efficient**: Pay ~$0.00001667 per CPU-second, $0.0000025 per GB-second
3. **Fast deployments**: 30-60 second cold start is acceptable for inference
4. **Python support**: Native support for Python 3.11 with all required libraries
5. **Deep GCP integration**: Native logging to Cloud Logging, metrics to Cloud Monitoring

---

## Consequences

### Positive

1. **Reduced Operational Burden**: No Kubernetes clusters to manage, no patching VMs
   - Saves 10+ hours/month on ops tasks
   - Frees team to focus on features

2. **Automatic Scaling**: Handles traffic spikes without manual intervention
   - 0→1000 req/s in <30 seconds
   - Perfect for unpredictable agent demand patterns

3. **Cost Efficient**: Only pay for actual compute time
   - Saves ~60% vs. reserved instances
   - No idle capacity waste

4. **Rapid Iteration**: Deploy new agent versions in <5 minutes
   - Enables A/B testing
   - Faster feedback loops

### Negative

1. **Cold Starts**: New instances take 30-60 seconds to start
   - Impact: First request to new instance slower
   - Mitigation: Keep minimum 2 instances warm

2. **Memory Limitations**: Max 8GB RAM per instance (as of 2026)
   - Impact: Large models may need quantization
   - Mitigation: Use 4-bit or 8-bit quantized models

3. **Concurrency Cap**: Max 1000 concurrent requests per instance
   - Impact: At peak traffic, may need 50+ instances
   - Mitigation: Use regional instances + load balancing

4. **No Local Storage**: Ephemeral filesystem only
   - Impact: Must cache models in memory or use Cloud Storage
   - Mitigation: Implemented model caching in Cloud Storage

### Risks

1. **GCP Region Outage**: If us-central1 goes down, service unavailable
   - Mitigation: Implement failover to us-east1 region (future phase)
   - SLA: GCP guarantees 99.95% availability

2. **Vendor Lock-in**: Difficult to migrate away from Cloud Run
   - Mitigation: Containerized approach means we can migrate to GKE if needed
   - Impact: Would require 3-4 weeks of refactoring

3. **Cost Increase**: If traffic grows unexpectedly, costs could spike
   - Mitigation: Set spend alerts, implement rate limiting
   - Cap: Hard limit at $50k/month

---

## Alternatives Considered

### Alternative A: Google Kubernetes Engine (GKE)

**Pros**:

- Full control over infrastructure
- Can optimize resource utilization
- Portability (can migrate to other cloud providers)

**Cons**:

- Requires full ops team (cluster management, patching, scaling)
- Higher baseline cost (even with 0 traffic)
- Slower deployment (5-10 minutes typical)
- More complex monitoring and debugging

**Why Not Chosen**: Too much operational overhead for current team size. GKE makes sense at >$100k/month infrastructure spend.

---

### Alternative B: AWS Lambda

**Pros**:

- Similar serverless model
- Competitive pricing

**Cons**:

- We're committed to GCP (landing zone framework)
- Different tooling/monitoring
- Training team on new platform = 2+ weeks lost productivity

**Why Not Chosen**: Doesn't fit GCP landing zone strategy.

---

### Alternative C: Traditional VM Deployment

**Pros**:

- Full control
- Predictable costs
- No cold starts

**Cons**:

- Requires ops team (24/7 on-call for patching, scaling)
- High baseline cost ($5k+/month just for baseline instances)
- Difficult to auto-scale (needs managed instance groups)
- Slower deployments (15+ minutes typical)

**Why Not Chosen**: Too expensive and operationally complex.

---

## Implementation

### Steps

1. **Phase 1 (Complete)**: Deploy inference service to Cloud Run
   - Owner: @ml-team
   - Deliverable: ollama-inference service running

2. **Phase 2 (In Progress)**: Implement model caching strategy
   - Owner: @platform-team
   - Deliverable: Models load 10x faster

3. **Phase 3 (Next)**: Add regional failover
   - Owner: @infrastructure-team
   - Timeline: Q2 2026

### Success Criteria

- ✅ Inference service deployed to production
- ✅ Handles 100+ concurrent requests without scaling delays
- ✅ Cold start time < 60 seconds
- ✅ Monthly cost < $10k at current traffic
- ✅ Team confidence in managing platform

---

## Monitoring & Maintenance

### Metrics to Track

- **Cold Start Duration**: Should be 30-60 seconds
- **P95 Latency**: Should be < 5 seconds (including model inference)
- **Cost Per Request**: Track to catch unexpected increases
- **Error Rate**: Should be < 1%

### Review Schedule

- Quarterly cost review
- Monthly performance review
- Annual architectural assessment (should we graduate to GKE?)

---

## Related Decisions

- ADR-002: BigQuery for metrics aggregation
- Issue #11: CI/CD Pipeline implementation

---

## References

- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Cloud Run Limits](https://cloud.google.com/run/quotas)
- [PyTorch on Cloud Run](https://cloud.google.com/run/docs/quickstarts/build-and-deploy)

---

## Sign-Off

| Role   | Name          | Date       | Status      |
| ------ | ------------- | ---------- | ----------- |
| Author | @architecture | 2026-01-26 | ✅ Accepted |
| CTO    | @cto          | 2026-01-26 | ✅ Approved |
| VP Eng | @vp-eng       | 2026-01-26 | ✅ Approved |

**Created**: 2026-01-26
**Status**: Production (Active since 2025-12-15)
