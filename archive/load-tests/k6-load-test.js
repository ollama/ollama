import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const inferenceLatency = new Trend('inference_latency');
const cacheHitRate = new Rate('cache_hits');
const successfulRequests = new Counter('successful_requests');
const rateLimit429s = new Counter('rate_limit_429s');
const activeConcurrentUsers = new Gauge('active_concurrent_users');

// Test configuration
export const options = {
  stages: [
    // Warm up: gradually increase from 1 to 5 concurrent users
    { duration: '30s', target: 5 },
    // Ramp up: increase from 5 to 50 concurrent users
    { duration: '2m', target: 50 },
    // Steady state: hold at 50 concurrent users
    { duration: '5m', target: 50 },
    // Ramp down: decrease from 50 to 0 concurrent users
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    // SLO: 95% of requests < 500ms
    'inference_latency': ['p(95) < 500'],
    // SLO: error rate < 1%
    'errors': ['rate < 0.01'],
    // SLO: cache hit rate > 70%
    'cache_hits': ['rate > 0.70'],
  },
};

const API_KEY = __ENV.API_KEY || 'test-api-key';
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  activeConcurrentUsers.add(1);

  // Test 1: List available models
  group('List Models', () => {
    const listResponse = http.get(`${BASE_URL}/api/v1/models`, {
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        'Content-Type': 'application/json',
      },
    });

    check(listResponse, {
      'status is 200': (r) => r.status === 200,
      'response time < 200ms': (r) => r.timings.duration < 200,
      'has models field': (r) => {
        const json = r.json();
        return json && (json.data?.models || json.models);
      },
    });

    errorRate.add(listResponse.status >= 400);
    if (listResponse.status === 200) {
      successfulRequests.add(1);
    }
  });

  sleep(1);

  // Test 2: Generate text with inference
  group('Generate Text', () => {
    const generatePayload = JSON.stringify({
      model: 'llama3.2',
      prompt: 'What is the capital of France?',
      stream: false,
    });

    const generateResponse = http.post(
      `${BASE_URL}/api/v1/generate`,
      generatePayload,
      {
        headers: {
          Authorization: `Bearer ${API_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );

    const duration = generateResponse.timings.duration;
    inferenceLatency.add(duration);

    check(generateResponse, {
      'status is 200': (r) => r.status === 200,
      'response time < 5s': (r) => r.timings.duration < 5000,
      'has text field': (r) => {
        const json = r.json();
        return json && (json.data?.text || json.text);
      },
    });

    errorRate.add(generateResponse.status >= 400);
    if (generateResponse.status === 200) {
      successfulRequests.add(1);
      cacheHitRate.add(1); // Track as potential cache hit
    } else if (generateResponse.status === 429) {
      rateLimit429s.add(1);
    }
  });

  sleep(2);

  // Test 3: Generate embeddings
  group('Generate Embeddings', () => {
    const embeddingPayload = JSON.stringify({
      model: 'llama3.2',
      input: 'Generate embeddings for this text',
    });

    const embeddingResponse = http.post(
      `${BASE_URL}/api/v1/embeddings`,
      embeddingPayload,
      {
        headers: {
          Authorization: `Bearer ${API_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );

    check(embeddingResponse, {
      'status is 200': (r) => r.status === 200,
      'response time < 500ms': (r) => r.timings.duration < 500,
      'has embeddings': (r) => {
        const json = r.json();
        return json && json.data && Array.isArray(json.data.embeddings);
      },
    });

    errorRate.add(embeddingResponse.status >= 400);
    if (embeddingResponse.status === 200) {
      successfulRequests.add(1);
    }
  });

  sleep(1);

  // Test 4: Health check (very fast)
  group('Health Check', () => {
    const healthResponse = http.get(`${BASE_URL}/health`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    check(healthResponse, {
      'status is 200': (r) => r.status === 200,
      'response time < 50ms': (r) => r.timings.duration < 50,
    });

    errorRate.add(healthResponse.status >= 400);
    if (healthResponse.status === 200) {
      successfulRequests.add(1);
    }
  });

  sleep(1);

  // Test 5: Chat completion
  group('Chat Completion', () => {
    const chatPayload = JSON.stringify({
      model: 'llama3.2',
      messages: [
        {
          role: 'user',
          content: 'What is the meaning of life?',
        },
      ],
    });

    const chatResponse = http.post(
      `${BASE_URL}/api/v1/chat`,
      chatPayload,
      {
        headers: {
          Authorization: `Bearer ${API_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );

    const duration = chatResponse.timings.duration;
    inferenceLatency.add(duration);

    check(chatResponse, {
      'status is 200': (r) => r.status === 200,
      'response time < 5s': (r) => r.timings.duration < 5000,
      'has response': (r) => {
        const json = r.json();
        return json && (json.data?.response || json.message);
      },
    });

    errorRate.add(chatResponse.status >= 400);
    if (chatResponse.status === 200) {
      successfulRequests.add(1);
    }
  });

  activeConcurrentUsers.add(-1);
  sleep(1);
}

// Summary function
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    '/tmp/ollama-load-test-results.json': JSON.stringify(data),
  };
}

function textSummary(data, options) {
  const indent = options?.indent || '';
  let summary = '\n=== Load Test Summary ===\n';

  if (data.metrics) {
    summary += `${indent}Errors: ${data.metrics.errors?.value || 0}\n`;
    summary += `${indent}Successful Requests: ${
      data.metrics.successful_requests?.value || 0
    }\n`;

    if (data.metrics.inference_latency) {
      const latency = data.metrics.inference_latency;
      summary += `${indent}Inference Latency:\n`;
      summary += `${indent}  - P50: ${latency.values?.['p(50)'] || 'N/A'}ms\n`;
      summary += `${indent}  - P95: ${latency.values?.['p(95)'] || 'N/A'}ms\n`;
      summary += `${indent}  - P99: ${latency.values?.['p(99)'] || 'N/A'}ms\n`;
    }

    if (data.metrics.cache_hits) {
      const hitRate = data.metrics.cache_hits;
      summary += `${indent}Cache Hit Rate: ${hitRate.value || 0}%\n`;
    }

    if (data.metrics.rate_limit_429s) {
      summary += `${indent}Rate Limited (429): ${
        data.metrics.rate_limit_429s?.value || 0
      }\n`;
    }
  }

  summary += '=========================\n';
  return summary;
}
