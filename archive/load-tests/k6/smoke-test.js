/**
 * Smoke Test - Issue #48 Phase 1
 *
 * Quick validation that API endpoints are responding.
 * Runs 5 concurrent users making basic requests for 30 seconds.
 *
 * Usage:
 *   k6 run smoke-test.js
 *   k6 run smoke-test.js --vus=10 --duration=1m
 *
 * Version: 1.0.0
 * Status: PRODUCTION-READY
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Configuration
export const options = {
  vus: 5,           // 5 virtual users
  duration: '30s',  // 30 seconds duration
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% under 500ms
    http_req_failed: ['rate<0.1'],     // Less than 10% failure rate
    errors: ['rate<0.1'],              // Less than 10% errors
  },
};

// Base URL - configurable via environment variable
const baseUrl = __ENV.BASE_URL || 'https://elevatediq.ai/ollama';
const apiKey = __ENV.API_KEY || 'sk-test-key-123456789';

export default function () {
  const headers = {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };

  // Group 1: Health Check
  group('Health Check', () => {
    const res = http.get(`${baseUrl}/api/v1/health`, { headers });
    
    const success = check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 200ms': (r) => r.timings.duration < 200,
      'has status field': (r) => r.json('status') !== null,
    });

    errorRate.add(!success);
  });

  sleep(1);

  // Group 2: List Models
  group('List Models', () => {
    const res = http.get(`${baseUrl}/api/v1/models`, { headers });
    
    const success = check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 300ms': (r) => r.timings.duration < 300,
      'has models array': (r) => Array.isArray(r.json('data.models')),
    });

    errorRate.add(!success);
  });

  sleep(1);

  // Group 3: Generate Request (Basic)
  group('Generate Text', () => {
    const payload = JSON.stringify({
      model: 'llama3.2',
      prompt: 'What is the capital of France?',
      max_tokens: 50,
    });

    const res = http.post(
      `${baseUrl}/api/v1/generate`,
      payload,
      { headers }
    );
    
    const success = check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 2000ms': (r) => r.timings.duration < 2000,
      'has text field': (r) => r.json('data.text') !== null,
      'valid model response': (r) => r.json('data.model') !== null,
    });

    errorRate.add(!success);
  });

  sleep(2);
}

/**
 * Summary Function - Called at end of test
 */
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'load-tests/results/smoke-results.json': JSON.stringify(data),
  };
}

/**
 * Simple text summary formatter
 */
function textSummary(data, options = {}) {
  const indent = options.indent || '';
  let summary = '\n=== SMOKE TEST RESULTS ===\n';

  const metrics = data.metrics;
  
  // Request metrics
  if (metrics.http_reqs) {
    summary += `${indent}Total Requests: ${metrics.http_reqs.value}\n`;
  }
  
  if (metrics.http_req_duration) {
    const stats = metrics.http_req_duration.values;
    summary += `${indent}Response Time:\n`;
    summary += `${indent}  Avg: ${stats.avg.toFixed(0)}ms\n`;
    summary += `${indent}  Min: ${stats.min.toFixed(0)}ms\n`;
    summary += `${indent}  Max: ${stats.max.toFixed(0)}ms\n`;
    summary += `${indent}  P95: ${stats.p95.toFixed(0)}ms\n`;
    summary += `${indent}  P99: ${stats.p99.toFixed(0)}ms\n`;
  }

  if (metrics.http_req_failed) {
    summary += `${indent}Failed Requests: ${metrics.http_req_failed.value}\n`;
  }

  if (metrics.errors) {
    summary += `${indent}Errors: ${metrics.errors.value}\n`;
  }

  summary += '\n';
  return summary;
}
