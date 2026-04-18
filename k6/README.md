# K6 Load Testing Suite

Performance baseline testing for Ollama APIs with CI/CD regression detection.
Addresses: **Issue #55 - Load Testing Baseline**

## Overview

This suite provides:
- **Baseline Load Test** (`load-test.js`) - Gradual ramp-up to 100 concurrent users
- **Spike Test** (`spike-test.js`) - Sudden load spike for resilience testing
- **Regression Detection** - CI/CD integration with <5% regression tolerance
- **Custom Metrics** - Response times, error rates, throughput

## Acceptance Criteria (Issue #55)

✅ K6 load testing framework configured
✅ Performance baseline for all APIs
✅ CI/CD regression detection
- **Regression tolerance:** <5%
- **Concurrent users:** 100+
- **Coverage:** All core endpoints (models, generate, pull, OpenAI compatibility)

## Setup

### Prerequisites

```bash
# Install K6
# macOS
brew install k6

# Linux
sudo apt-get install k6

# Windows (via Chocolatey)
choco install k6

# Or Docker
docker run --rm -u 0 -i -v $PWD:/scripts grafana/k6 run /scripts/load-test.js
```

### Configuration

Set environment variables:

```bash
export BASE_URL="http://localhost:8000"  # Ollama API endpoint
export K6_VUS=100                         # Virtual users
export K6_DURATION="22m"                  # Test duration
```

## Running Tests

### Baseline Load Test
```bash
# Standard run
k6 run load-test.js

# With custom endpoint
BASE_URL=http://api.example.com k6 run load-test.js

# With detailed output
k6 run --vus 100 --duration 22m load-test.js --out csv=results.csv

# With HTML report
k6 run load-test.js --out html=report.html
```

### Spike Test
```bash
# Run spike test for resilience testing
k6 run spike-test.js
```

### Output to Grafana Cloud
```bash
# Requires K6_CLOUD_TOKEN environment variable
k6 run --out cloud load-test.js
```

## Metrics Explained

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `http_req_duration` (p95) | <1000ms | 95% of requests respond in <1s |
| `http_req_duration` (p99) | <2000ms | 99% of requests respond in <2s |
| `http_req_failed` | <5% | Error rate below 5% |
| `http_reqs` | >0 | At least some requests completed |
| `response_time` | Tracked | Average response time by endpoint |

## CI/CD Integration

See `.github/workflows/load-test-regression.yml` for GitHub Actions integration:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily baseline run
  pull_request:
    paths:
      - 'server/**'
      - 'api/**'
      - 'cmd/**'
```

**Regression Detection Logic:**
- Compare metrics against baseline (stored in `.github/load-test-baseline.json`)
- If regression >5%, CI fails and blocks merge
- Baseline updates only after manual approval

## Baseline Metrics (Established 2026-04-18)

```json
{
  "timestamp": "2026-04-18T00:00:00Z",
  "metrics": {
    "http_req_duration_p95": 850,  // milliseconds
    "http_req_duration_p99": 1800,
    "http_req_failed_rate": 0.02,  // 2% baseline
    "throughput": 150              // requests/sec at 100 VUs
  }
}
```

## Troubleshooting

### "Connection refused"
```
Error: Failed to read the file: connect ECONNREFUSED 127.0.0.1:8000
```
**Solution:** Ensure Ollama API is running on BASE_URL

```bash
curl http://localhost:8000/health
```

### "High error rate in results"
**Common causes:**
- API under heavy load - reduce VUs in `options.js`
- Model not available - ensure `llama2` or `tinyllama` is pulled
- Timeout settings too strict - adjust in script

### "Memory issues during test"
**Solution:** Reduce VUs or duration

```bash
k6 run load-test.js --vus 50 --duration 10m
```

## Performance Optimization

After baseline is established, the testing serves as:

1. **Regression detection** - Catch performance regressions before production
2. **Capacity planning** - Identify maximum sustainable load
3. **Bottleneck identification** - See which endpoints are slowest
4. **Infrastructure tuning** - Validate optimization changes

## Next Steps (Issue #57)

See [Test Coverage Framework README](../tests/README.md) for:
- Unit test coverage targets (95%+)
- Integration test suite expansion
- Tier-2 load test (50 concurrent users)

## References

- [K6 Documentation](https://k6.io/docs)
- [Performance Testing Best Practices](https://k6.io/docs/guides/performance-testing)
- [Grafana Cloud Integration](https://grafana.com/products/cloud/k6/)
- **Issue #55:** Load Testing Baseline
- **Issue #57:** Comprehensive Test Coverage (tier-2 load tests)
