# Issue #45: Canary & Progressive Deployment Implementation Guide

**Issue**: [#45 - Canary & Progressive Deployments](https://github.com/kushin77/ollama/issues/45)  
**Status**: OPEN - Ready for Assignment  
**Priority**: HIGH  
**Estimated Hours**: 85h (12.1 days)  
**Timeline**: Week 2-3 (Feb 10-21, 2026)  
**Dependencies**: #42 (Federation), #43 (Security)  
**Parallel Work**: #44, #46, #47, #48, #50  

## Overview

Implement automated canary and progressive deployment strategies using **Flagger**, **Istio**, and **Prometheus metrics**. Enable safe, low-risk deployments with automatic rollback on anomalies.

## Architecture

```
New Version (5% traffic) ─┐
                          ├─→ Metrics Analysis → Rollback/Promote
Current Version (95%)    ─┘
                          ↓
              Flagger Canary Controller
                          ↓
              Prometheus Metrics + Analysis
```

## Phase 1: Flagger Setup (Week 2, 25 hours)

### 1.1 Flagger Installation & Configuration
- Helm chart deployment
- Istio integration
- Prometheus metrics binding
- Slack/webhook notifications

**Flagger HelmValues** (80 lines):
```yaml
replicaCount: 2
image:
  repository: fluxcd/flagger
  tag: v1.25.0
prometheus:
  install: true
istio:
  namespace: istio-system
slack:
  url: https://hooks.slack.com/...
metricsServer: prometheus
```

### 1.2 Canary Resource Definition
- Service definition
- Traffic splitting rules
- Metric thresholds
- Success criteria

**Code** (150 lines - `k8s/deployment/canary.yaml`):
```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: ollama-api
  namespace: ollama
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ollama-api
  progressDeadlineSeconds: 300
  service:
    port: 8000
    targetPort: 8000
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 5
  metrics:
  - name: request-success-rate
    interval: 1m
    thresholdRange:
      min: 95
  - name: request-duration
    interval: 1m
    thresholdRange:
      max: 500
  webhooks:
  - name: slack-notification
    url: https://hooks.slack.com/...
    timeout: 5s
    metadata:
      cmd: "curl -sd POST"
```

## Phase 2: Deployment Strategies (Week 2-3, 35 hours)

### 2.1 Canary Deployment (10% → 50%)
- Progressive traffic shifting
- Metric-based promotion
- Automatic rollback
- Deployment time: 10-15 minutes

**Code** (200 lines - `ollama/deployment/canary_manager.py`):
```python
class CanaryManager:
    async def execute_canary(
        self,
        service: str,
        new_version: str,
        metrics_threshold: float = 95.0,
        max_canary_weight: int = 50
    ) -> DeploymentResult:
        """Execute canary deployment."""
        # Create Flagger Canary resource
        canary = self._create_canary_resource(service, new_version)
        
        # Watch metrics during deployment
        async for status in self._watch_canary_status(canary):
            if status.success_rate < metrics_threshold:
                await self._rollback_canary(canary)
                return DeploymentResult(success=False, rolled_back=True)
            elif status.weight >= max_canary_weight:
                await self._promote_canary(canary)
                return DeploymentResult(success=True, promoted=True)
```

### 2.2 Blue-Green Deployment
- Two identical environments
- Zero-downtime cutover
- Instant rollback capability
- Pre-production testing

### 2.3 A/B Testing Support
- User cohort routing
- Metric comparison
- Statistical significance testing

## Phase 3: Automated Rollback & Monitoring (Week 3, 25 hours)

### 3.1 Anomaly Detection
- Metric deviation detection (>2 std dev)
- Error rate spikes
- Latency increases
- Automatic rollback triggering

**Code** (250 lines - `ollama/deployment/anomaly_detector.py`)

### 3.2 Health Checks & Readiness Probes
- Startup probe (max 60s)
- Readiness probe (30s interval)
- Liveness probe (10s interval)
- Custom health check logic

### 3.3 Post-Deployment Validation
- Smoke tests
- Integration tests
- Load tests verification
- Data consistency checks

## Acceptance Criteria

- [ ] Canary deployments working (5→50% traffic shift)
- [ ] Automated rollback on metric threshold
- [ ] Deployment time <15 minutes
- [ ] Zero data loss on rollback
- [ ] Slack notifications working
- [ ] E2E canary test passing

## Testing Strategy

- Unit tests for canary logic (20 tests)
- Integration tests for Flagger (15 tests)
- Deployment tests with real K8s (10 tests)
- Chaos tests for rollback scenarios (8 tests)

## Success Metrics

- **Deployment Success Rate**: ≥99%
- **Rollback Time**: <2 minutes
- **Canary Duration**: 10-15 minutes
- **Zero data loss**: 100%
- **Manual intervention needed**: <5% of deployments

---

**Next Steps**: Assign to DevOps/SRE engineer, begin Week 2
