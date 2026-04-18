/**
 * K6 Load Testing Script - Ollama API Baseline
 * Tests core API endpoints with realistic traffic patterns
 * Monitors: response times, error rates, throughput
 * Issue #55: Load Testing Baseline
 */

import { check, group, sleep } from 'k6';
import http from 'k6/http';
import { Counter, Gauge, Rate, Trend } from 'k6/metrics';
import { options } from './options.js';

// Custom metrics
const apiErrors = new Counter('api_errors');
const apiSuccess = new Counter('api_success');
const responseTime = new Trend('response_time');
const activeUsers = new Gauge('active_users');
const throughput = new Rate('successful_requests');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

export { options };

/**
 * Setup: Initialize test environment
 */
export function setup() {
  console.log(`Starting load test against: ${BASE_URL}`);

  // Health check
  const res = http.get(`${BASE_URL}/health`, { timeout: '30s' });
  check(res, {
    'health check passed': (r) => r.status === 200,
  }) || console.error('Health check failed - API may not be ready');

  return { startTime: new Date() };
}

/**
 * Main Test Function: VU (Virtual User) lifecycle
 */
export default function (data) {
  activeUsers.add(__VU); // Track active VU count

  // Simulate realistic user behavior with think time
  group('List Models', () => {
    const res = http.get(`${BASE_URL}/api/models`, {
      timeout: `${API_TIMEOUT}ms`,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'k6-load-test/1.0'
      }
    });

    const isSuccess = check(res, {
      'list models status 200': (r) => r.status === 200,
      'list models response time < 1s': (r) => r.timings.duration < 1000,
      'list models has content': (r) => r.body.length > 0,
    });

    responseTime.add(res.timings.duration);
    isSuccess ? apiSuccess.add(1) : apiErrors.add(1);
    throughput.add(isSuccess);

    sleep(1);
  });

  // Generate model call (if any models available)
  group('Generate Completion', () => {
    const payload = JSON.stringify({
      model: 'llama2',
      prompt: 'What is machine learning?',
      stream: false,
      options: {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9
      }
    });

    const res = http.post(`${BASE_URL}/api/generate`, payload, {
      timeout: `${API_TIMEOUT}ms`,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'k6-load-test/1.0'
      }
    });

    const isSuccess = check(res, {
      'generate status 200 or 500': (r) => [200, 500].includes(r.status),
      'generate response time < 5s': (r) => r.timings.duration < 5000,
      'generate has response': (r) => r.body.length > 0,
    });

    responseTime.add(res.timings.duration);
    isSuccess ? apiSuccess.add(1) : apiErrors.add(1);
    throughput.add(isSuccess);

    sleep(2);
  });

  // Pull/Create model
  group('Model Management', () => {
    const res = http.post(
      `${BASE_URL}/api/pull`,
      JSON.stringify({ name: 'tinyllama' }),
      {
        timeout: `${API_TIMEOUT}ms`,
        headers: { 'Content-Type': 'application/json' }
      }
    );

    // Pull may fail if model exists or network issues
    check(res, {
      'pull request accepted': (r) => [200, 202, 409].includes(r.status),
    });

    sleep(1);
  });

  // OpenAI-compatible endpoint
  group('OpenAI Endpoint', () => {
    const res = http.post(
      `${BASE_URL}/v1/chat/completions`,
      JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: 'Hello!' }],
        temperature: 0.7
      }),
      {
        timeout: `${API_TIMEOUT}ms`,
        headers: { 'Content-Type': 'application/json' }
      }
    );

    check(res, {
      'openai endpoint responds': (r) => [200, 400, 500].includes(r.status),
    });

    sleep(1);
  });
}

/**
 * Teardown: Cleanup and final metrics
 */
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Total users reached: ${options.stages.find(s => s.target === Math.max(...options.stages.map(s => s.target))).target}`);
}
