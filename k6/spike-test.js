/**
 * K6 Spike Test Configuration
 * Tests system behavior under sudden load increases
 * Issue #55: Load Testing Baseline - Spike Test Variant
 */

import { check, sleep } from 'k6';
import http from 'k6/http';

export const options = {
  // Spike test: sudden jump to peak load
  stages: [
    { duration: '2m', target: 10 },    // Warm-up
    { duration: '1m', target: 100 },   // Spike to 100 users
    { duration: '2m', target: 100 },   // Stay at 100 users
    { duration: '1m', target: 50 },    // Partial recovery
    { duration: '2m', target: 0 },     // Cool-down
  ],

  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.10'],  // More lenient for spike test (10% error rate)
  }
};

export default function () {
  const res = http.get('http://localhost:8000/api/models', {
    timeout: '30s'
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time acceptable': (r) => r.timings.duration < 2000,
  });

  sleep(1);
}
