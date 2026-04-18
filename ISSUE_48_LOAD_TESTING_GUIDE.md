# Issue #48: Load Testing Baseline & Performance Framework Implementation Guide

**Issue**: [#48 - Load Testing Baseline](https://github.com/kushin77/ollama/issues/48)  
**Status**: OPEN - Ready for Assignment  
**Priority**: MEDIUM  
**Estimated Hours**: 70h (10 days)  
**Timeline**: Week 1-2 (Feb 3-14, 2026)  
**Dependencies**: #42 (Federation - for realistic load scenarios)  
**Parallel Work**: #43, #44, #45, #46, #47, #49, #50  

## Overview

Implement comprehensive load testing framework using **K6**, establish baseline performance metrics, and set up continuous load testing in CI/CD pipeline.

## Baselines (Target)

**Tier 1 (Internal)**:
- 10 concurrent users
- 100 req/s
- 95% success rate
- P95 latency: 55ms
- P99 latency: 85ms

**Tier 2 (Production)**:
- 50 concurrent users
- 500 req/s
- 99.5% success rate
- P95 latency: 75ms
- P99 latency: 150ms

**Tier 3 (Enterprise)**:
- 200 concurrent users
- 2,000 req/s
- 99.9% success rate
- P95 latency: 100ms
- P99 latency: 200ms

## Architecture

```
K6 Test Scripts → Real-world Scenarios → Metrics → InfluxDB → Grafana Dashboards
                                      ↓
                            Pass/Fail Criteria
                                      ↓
                            CI/CD Integration (PR gating)
```

## Phase 1: K6 Framework Setup (Week 1, 25 hours)

### 1.1 K6 Installation & Project Structure
- K6 container setup
- Node.js/npm environment
- Project structure
- Local vs remote execution

**Project Structure**:
```
load-tests/
├── scripts/
│   ├── api_performance.js       # API endpoint testing
│   ├── federation_flow.js       # Federation testing
│   ├── inference_load.js        # Inference workload
│   └── smoke_test.js            # Quick smoke test
├── thresholds.js                # Pass/fail criteria
├── setup.sh
└── README.md
```

### 1.2 K6 Scripts Development
- API endpoint testing (health, generate, chat, embeddings)
- Federation flow testing (hub communication)
- Inference workload simulation
- Smoke test for quick validation

**Example K6 Script** (200 lines):
```javascript
// load-tests/scripts/api_performance.js
import http from 'k6/http';
import { check, group } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-key';

export const options = {
  stages: [
    { duration: '30s', target: 5 },   // Warm up
    { duration: '1m30s', target: 10 }, // Ramp up
    { duration: '5m', target: 10 },    // Stay at load
    { duration: '30s', target: 0 }     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<100', 'p(99)<200'],
    http_req_failed: ['rate<0.05']
  }
};

export default function() {
  group('API Health Check', () => {
    const res = http.get(`${BASE_URL}/api/v1/health`);
    check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 100ms': (r) => r.timings.duration < 100
    });
  });

  group('Generate Endpoint', () => {
    const payload = JSON.stringify({
      prompt: 'What is the capital of France?',
      model: 'llama3.2',
      max_tokens: 100
    });

    const res = http.post(
      `${BASE_URL}/api/v1/generate`,
      payload,
      {
        headers: {
          'Authorization': `Bearer ${API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 500ms': (r) => r.timings.duration < 500,
      'has text output': (r) => r.json('data.text').length > 0
    });
  });
}
```

### 1.3 Metric Collection
- Request duration (min, p50, p95, p99, max)
- Error rates
- Success rates
- Request counts
- Data upload/download rates

## Phase 2: Baseline Testing (Week 1-2, 25 hours)

### 2.1 Tier 1 Baseline (10 users, 100 req/s)
- Run local load test
- Record baseline metrics
- Document system specs
- Identify bottlenecks

**Baseline Results** (Template):
```json
{
  "tier": 1,
  "concurrent_users": 10,
  "target_rps": 100,
  "duration_minutes": 10,
  "results": {
    "success_rate": 100.0,
    "total_requests": 1000,
    "avg_latency_ms": 25,
    "p50_latency_ms": 20,
    "p95_latency_ms": 55,
    "p99_latency_ms": 85,
    "max_latency_ms": 150,
    "error_rate": 0.0
  },
  "system_stats": {
    "cpu_usage": 45,
    "memory_usage": 60,
    "disk_io": "normal"
  }
}
```

### 2.2 Tier 2 Baseline (50 users, 500 req/s)
- Scale up test
- Monitor scaling behavior
- Identify scaling limits
- Resource consumption analysis

### 2.3 Tier 3 Baseline (200 users, 2000 req/s)
- Peak load test
- Breaking point identification
- Recovery testing
- Long-duration stability test

## Phase 3: CI/CD Integration & Continuous Testing (Week 2, 20 hours)

### 3.1 GitHub Actions Integration
- PR gate: Run smoke test
- Nightly: Run full Tier 1 baseline
- Weekly: Run Tier 2 baseline
- Monthly: Run Tier 3 baseline

**GitHub Workflow** (100 lines):
```yaml
# .github/workflows/load-test.yml
name: Load Testing

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: grafana/k6-action@v0.3.0
        with:
          filename: load-tests/scripts/smoke_test.js
          cloud: true
          vus: 5
          duration: 30s
```

### 3.2 Performance Dashboard
- K6 Cloud integration
- Trend tracking
- Regression detection
- Weekly reports

### 3.3 Threshold Management
- Pass/fail criteria per endpoint
- SLO definitions
- Alert triggers
- Automated reporting

## Acceptance Criteria

- [ ] K6 scripts for all major endpoints
- [ ] Tier 1 baseline established and documented
- [ ] Tier 2 baseline established
- [ ] Tier 3 baseline established (optional Week 1)
- [ ] CI/CD integration working
- [ ] Smoke test < 2 minutes
- [ ] Full Tier 1 test < 15 minutes
- [ ] All baselines documented in repo

## Testing Strategy

- Smoke tests (5 min, < 10 users)
- Tier 1 baseline (15 min, 10 users)
- Tier 2 baseline (30 min, 50 users)
- Tier 3 baseline (60 min, 200 users)
- Stress tests (identify breaking point)
- Soak tests (8-hour stability)

## Success Metrics

- **P95 Latency**: <100ms for Tier 1
- **Success Rate**: ≥99.5% at Tier 2
- **Error Rate**: <0.5% at all tiers
- **Resource Scaling**: Linear up to Tier 2, graceful degradation at Tier 3
- **Test Execution Time**: <2min smoke, <15min Tier 1

---

**Next Steps**: Assign to performance engineer, begin Week 1 (parallel with Federation)
