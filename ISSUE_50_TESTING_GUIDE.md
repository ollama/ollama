# Issue #50: Comprehensive Test Coverage & Resilience Testing Implementation Guide

**Issue**: [#50 - Comprehensive Test Coverage](https://github.com/kushin77/ollama/issues/50)  
**Status**: OPEN - Ready for Assignment  
**Priority**: MEDIUM  
**Estimated Hours**: 60h (8.6 days)  
**Timeline**: Week 1-3 (Feb 3-21, 2026)  
**Dependencies**: #42 (Federation code)  
**Parallel Work**: #43, #44, #45, #46, #47, #48, #49  

## Overview

Implement comprehensive test coverage with **unit tests** (95%+ coverage), **integration tests** (100% critical paths), **chaos engineering** (Chaos Toolkit), **property-based testing** (Hypothesis), and **mutation testing** (Mutmut).

## Test Strategy Matrix

| Type | Coverage | Tool | Target | Timeline |
|------|----------|------|--------|----------|
| Unit | 95% | pytest | All modules | Week 1-2 |
| Integration | 100% critical | pytest | APIs, DB, external | Week 2 |
| Property-Based | 90% | Hypothesis | Core logic | Week 2-3 |
| Mutation | 80% | Mutmut | Validation logic | Week 3 |
| Chaos | 70% | Chaos Toolkit | Resilience | Week 3 |
| E2E | 100% critical | Playwright | User workflows | Week 3 |

## Phase 1: Unit Test Expansion (Week 1, 20 hours)

### 1.1 Unit Test Coverage Analysis
- Current coverage: ~90%
- Target: 95%+ coverage
- Focus gaps: Error handling, edge cases

**Coverage Report**:
```
ollama/
├── api/                   95.2%
├── auth/                  98.5%
├── config/                99.1%
├── services/              87.3%  ← Focus here
├── repositories/          91.2%
├── models/                89.5%
└── utils/                 96.8%
```

### 1.2 New Test Writing
- Error path testing
- Edge case coverage
- Boundary value testing
- Exception handling

**Code Structure** (100+ tests, 2000+ lines):
```python
# tests/unit/services/test_inference_edge_cases.py
import pytest
from ollama.services.inference import InferenceEngine

class TestInferenceEngineEdgeCases:
    """Test edge cases and error paths."""

    def test_generate_with_empty_prompt(self):
        """Empty prompt should raise validation error."""
        engine = InferenceEngine()
        with pytest.raises(ValidationError):
            engine.generate(prompt="")

    def test_generate_with_max_tokens_exceeded(self):
        """Exceeding max tokens should be rejected."""
        engine = InferenceEngine(max_tokens=1000)
        with pytest.raises(ValueError):
            engine.generate(prompt="test", max_tokens=2000)

    def test_generate_with_invalid_model(self):
        """Invalid model should raise error."""
        engine = InferenceEngine()
        with pytest.raises(ModelNotFoundError):
            engine.generate(prompt="test", model="invalid-model")

    def test_concurrent_requests(self):
        """Concurrent requests should be handled safely."""
        engine = InferenceEngine(max_concurrent=10)
        # Test concurrent handling...
```

### 1.3 Coverage Metrics Tracking
- Automated coverage reporting (pytest-cov)
- Branch coverage tracking
- Line coverage targets by module
- CI/CD gating (fail if <95%)

## Phase 2: Integration & Property-Based Testing (Week 2, 20 hours)

### 2.1 Integration Tests
- API endpoint integration
- Database transaction handling
- Cache behavior
- External service integration (mocked)

**Example** (50 tests, 1500+ lines):
```python
# tests/integration/test_api_workflows.py
class TestAPIWorkflows:
    """Test realistic API workflows."""

    @pytest.mark.asyncio
    async def test_generate_with_cache_hit(self, redis_client):
        """Generate should use cache on repeated requests."""
        prompt = "What is AI?"
        
        # First request (cache miss)
        result1 = await api.generate(prompt)
        
        # Second request (cache hit)
        result2 = await api.generate(prompt)
        
        assert result1 == result2
        assert redis_client.get_hit_count() > 0
```

### 2.2 Property-Based Testing with Hypothesis
- Input validation properties
- Data transformation invariants
- Marshalling/unmarshalling consistency

**Example**:
```python
# tests/unit/test_properties.py
from hypothesis import given, strategies as st
from ollama.utils.validators import validate_prompt

class TestPromptValidation:
    @given(st.text(min_size=1, max_size=10000))
    def test_prompt_always_accepts_valid_text(self, text):
        """All non-empty text should be accepted as prompt."""
        result = validate_prompt(text)
        assert isinstance(result, str)
        assert len(result) > 0
```

### 2.3 Mutation Testing with Mutmut
- Test quality validation
- Code quality enforcement
- Bug detection

**Commands**:
```bash
# Run mutation testing
mutmut run --paths-to-mutate=ollama/

# Results show how many mutations were caught
# Target: >80% mutations killed (detected by tests)
```

## Phase 3: Chaos & Resilience Testing (Week 2-3, 20 hours)

### 3.1 Chaos Toolkit Setup
- Service disruption simulation
- Network latency injection
- Resource exhaustion testing
- Failure scenario testing

**Chaos Experiments** (8 experiments):

1. **Pod Crash**: Kill inference service pod
2. **Network Latency**: Add 500ms latency to database
3. **CPU Exhaustion**: Run CPU burn on one pod
4. **Memory Leak**: Simulate memory leak in cache
5. **Disk Full**: Fill disk to 99%
6. **Network Partition**: Simulate region disconnection
7. **Cascading Failure**: Multiple service failures
8. **Recovery**: Verify recovery after failures

**Example Experiment** (YAML):
```yaml
# chaos/experiments/service-pod-crash.yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-crash-experiment
spec:
  containers:
  - name: chaos-monkey
    image: chaostoolkit/chaostoolkit:latest
    volumeMounts:
    - name: experiments
      mountPath: /tmp/experiments
  volumes:
  - name: experiments
    configMap:
      name: chaos-experiments
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: chaos-experiments
data:
  pod-crash.json: |
    {
      "title": "Pod Crash",
      "description": "Kill inference service pod",
      "steady-state-hypothesis": {
        "title": "Service should respond",
        "probes": [{
          "type": "http",
          "url": "http://ollama-api:8000/health",
          "expected_status": 200
        }]
      },
      "method": [{
        "type": "action",
        "name": "Kill pod",
        "provider": {
          "type": "python",
          "module": "chaosk8s.pod.actions",
          "func": "kill_pod",
          "arguments": {
            "pod_pattern": "ollama-api-.*",
            "ns": "ollama"
          }
        }
      }],
      "rollback": [{
        "type": "action",
        "name": "Ensure pod recovered",
        "provider": {
          "type": "python",
          "module": "chaosk8s.pod.probes",
          "func": "read_pod_logs",
          "arguments": {"ns": "ollama"}
        }
      }]
    }
```

### 3.2 Resilience Testing
- Circuit breaker testing
- Retry logic validation
- Timeout handling
- Graceful degradation

**Code** (250 lines - `tests/chaos/test_resilience.py`):
```python
class TestResilience:
    """Test system resilience to failures."""

    @pytest.mark.chaos
    async def test_service_recovery_after_crash(self):
        """Service should recover after pod crash."""
        # Get baseline metrics
        baseline = await get_health_metrics()
        
        # Inject failure (pod crash)
        await chaos.kill_pod('ollama-api')
        
        # Wait for recovery
        await asyncio.sleep(30)
        
        # Verify recovery
        current = await get_health_metrics()
        assert current['availability'] >= 0.99
        assert current['p99_latency'] < baseline['p99_latency'] * 1.2
```

### 3.3 Load Testing with Chaos
- Introduce failures during load tests
- Verify graceful degradation
- Validate auto-scaling triggers
- Recovery time validation

## Acceptance Criteria

- [ ] Unit test coverage: 95%+ (all modules)
- [ ] Integration tests: 100% of critical APIs
- [ ] Property-based tests: All core logic covered
- [ ] Mutation testing: >80% mutations killed
- [ ] Chaos experiments: All 8 passing
- [ ] E2E tests: 100% of golden paths
- [ ] CI/CD gating: All tests required to pass
- [ ] Documentation: Test strategy + examples

## Success Metrics

- **Unit Test Coverage**: 95%+ (enforce in CI)
- **Test Execution Time**: Unit <5min, Integration <10min, Chaos <15min
- **Mutation Score**: >80% (mutations killed by tests)
- **Chaos Resilience**: >99% availability during experiments
- **Regression Prevention**: Zero test-escaping bugs per quarter

## Testing Automation

**CI/CD Integration**:
- PR gating: Run unit + smoke integration tests
- Nightly: Full test suite (unit + integration + mutation)
- Weekly: Chaos testing
- Monthly: Full E2E + load + chaos combined

## Tools & Frameworks

- **pytest**: Unit, integration, E2E tests
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **Hypothesis**: Property-based testing
- **Mutmut**: Mutation testing
- **Chaos Toolkit**: Chaos engineering
- **Locust/K6**: Load testing during chaos

---

**Next Steps**: Assign to QA/test lead, begin Week 1
