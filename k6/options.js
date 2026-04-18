/**
 * K6 Load Testing Options Configuration
 * Baseline configuration for API performance testing
 * Acceptance: <5% regression tolerance, 100+ concurrent users
 */

export const options = {
  // Baseline test: gradual ramp-up to 100 concurrent users
  stages: [
    { duration: '2m', target: 10 },   // Ramp-up to 10 users over 2 minutes
    { duration: '5m', target: 50 },   // Ramp-up to 50 users over 5 minutes
    { duration: '10m', target: 100 }, // Ramp-up to 100 users over 10 minutes
    { duration: '5m', target: 0 },    // Ramp-down to 0 users over 5 minutes
  ],

  // Thresholds for regression detection (<5% tolerance)
  thresholds: {
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],      // 95th percentile < 1s, 99th percentile < 2s
    http_req_failed: ['rate<0.05'],                        // Error rate < 5%
    http_reqs: ['count>0'],                                // At least some requests completed
  },

  ext: {
    loadimpact: {
      projectID: 3452,
      name: 'Ollama API Baseline Load Test'
    }
  }
};
