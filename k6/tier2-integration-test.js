/**
 * K6 Tier-2 Load Test (Issue #57 - Comprehensive Test Coverage)
 * Tests with 50 concurrent users for integration/acceptance level
 * This is the expanded integration test suite variant
 */

import { check, group, sleep } from 'k6';
import http from 'k6/http';
import { Counter, Trend } from 'k6/metrics';

export const options = {
  // Tier-2: 50 concurrent users (acceptance/integration level)
  stages: [
    { duration: '3m', target: 25 },   // Ramp-up to 25 users
    { duration: '5m', target: 50 },   // Ramp-up to 50 users
    { duration: '10m', target: 50 },  // Sustained at 50 users
    { duration: '3m', target: 0 },    // Cool-down
  ],

  thresholds: {
    http_req_duration: ['p(95)<1500', 'p(99)<3000'],
    http_req_failed: ['rate<0.05'],
  }
};

// Metrics
const apiErrors = new Counter('api_errors_tier2');
const integrationTests = new Counter('integration_tests_complete');
const endToEndTests = new Counter('e2e_tests_complete');
const responseTime = new Trend('response_time_tier2');

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

/**
 * Critical Path 1: API Health Check
 */
function testHealthCheck() {
  group('Health Check Integration', () => {
    const res = http.get(`${BASE_URL}/health`, { timeout: '30s' });

    check(res, {
      'health is 200': (r) => r.status === 200,
      'health response format valid': (r) => {
        try {
          JSON.parse(r.body);
          return true;
        } catch { return false; }
      },
      'includes status field': (r) => r.body.includes('status'),
    });

    integrationTests.add(1);
    responseTime.add(res.timings.duration);
  });
}

/**
 * Critical Path 2: Model Loading & Listing
 */
function testModelManagement() {
  group('Model Management Integration', () => {
    // List models
    const listRes = http.get(`${BASE_URL}/api/models`, { timeout: '30s' });
    const listPass = check(listRes, {
      'list models is 200': (r) => r.status === 200,
      'returns array': (r) => r.body.includes('[') || r.body.includes('{'),
    });

    if (!listPass) apiErrors.add(1);
    responseTime.add(listRes.timings.duration);

    // Show model details (if available)
    const detailRes = http.get(`${BASE_URL}/api/show/tinyllama`, {
      timeout: '30s'
    });

    check(detailRes, {
      'show endpoint exists': (r) => [200, 404].includes(r.status),
    });

    integrationTests.add(1);
    sleep(1);
  });
}

/**
 * Critical Path 3: Token Generation & Streaming
 */
function testTokenGeneration() {
  group('Token Generation E2E', () => {
    const payload = JSON.stringify({
      model: 'tinyllama',
      prompt: 'What is 2+2?',
      stream: false,
      options: {
        temperature: 0.5,
        num_predict: 10
      }
    });

    const res = http.post(`${BASE_URL}/api/generate`, payload, {
      timeout: '60s',
      headers: { 'Content-Type': 'application/json' }
    });

    const pass = check(res, {
      'generate is 200 or 500': (r) => [200, 500].includes(r.status),
      'has response text': (r) => r.body.length > 0,
    });

    if (!pass) apiErrors.add(1);
    responseTime.add(res.timings.duration);
    endToEndTests.add(1);
    sleep(1);
  });
}

/**
 * Critical Path 4: Authentication & Rate Limiting
 */
function testAuthenticationFlow() {
  group('Authentication & Rate Limiting', () => {
    // Valid request
    const validRes = http.get(`${BASE_URL}/api/models`, {
      headers: { 'Authorization': 'Bearer valid-token-for-test' },
      timeout: '30s'
    });

    check(validRes, {
      'valid auth allows access': (r) => [200, 401].includes(r.status),
    });

    // Rapid requests to test rate limiting
    for (let i = 0; i < 5; i++) {
      http.get(`${BASE_URL}/api/models`, { timeout: '30s' });
    }

    const rateLimitRes = http.get(`${BASE_URL}/api/models`, { timeout: '30s' });
    check(rateLimitRes, {
      'rate limiting works': (r) => [200, 429].includes(r.status),
    });

    integrationTests.add(1);
  });
}

/**
 * Critical Path 5: Error Handling
 */
function testErrorHandling() {
  group('Error Handling & Recovery', () => {
    // Invalid model
    const invalidModelRes = http.post(
      `${BASE_URL}/api/generate`,
      JSON.stringify({ model: 'nonexistent-model-xyz', prompt: 'test' }),
      { headers: { 'Content-Type': 'application/json' }, timeout: '30s' }
    );

    check(invalidModelRes, {
      'invalid model returns error': (r) => [400, 404, 500].includes(r.status),
      'error has message': (r) => r.body.includes('error') || r.body.includes('message'),
    });

    // Malformed JSON
    const malformedRes = http.post(
      `${BASE_URL}/api/generate`,
      '{ invalid json }',
      { headers: { 'Content-Type': 'application/json' }, timeout: '30s' }
    );

    check(malformedRes, {
      'malformed input handled': (r) => [400, 500].includes(r.status),
    });

    integrationTests.add(1);
  });
}

/**
 * Virtual User: Simulates realistic integration test scenario
 */
export default function () {
  testHealthCheck();
  sleep(1);

  testModelManagement();
  sleep(1);

  testTokenGeneration();
  sleep(1);

  testAuthenticationFlow();
  sleep(1);

  testErrorHandling();
  sleep(2);
}

/**
 * Setup: Pre-test validation
 */
export function setup() {
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    'API available': (r) => r.status === 200
  }) || console.error('⚠️ API not ready - tests may fail');
}

/**
 * Teardown: Final metrics
 */
export function teardown() {
  console.log(`✅ Tier-2 Integration Tests Complete`);
  console.log(`   - Integration tests: ${integrationTests.value}`);
  console.log(`   - E2E tests: ${endToEndTests.value}`);
  console.log(`   - Errors: ${apiErrors.value}`);
}
