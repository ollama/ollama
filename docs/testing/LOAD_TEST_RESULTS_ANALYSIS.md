# 📊 Load Test Results Analysis - January 14, 2026

## Executive Summary

⚠️ **FOLLOW-UP TESTING COMPLETED – MIXED RESULTS**

- ElevatedIQ Cloud Run endpoint still returns 404 for `/api/v1/models` (pre-fix image).
- GovAI-Scout Cloud Run endpoint returns 200 for `/api/v1/models` on direct checks, but under sustained load we observed instability (500s on `/api/v1/models` and 500/503 on `/api/v1/generate`) due to degraded startup without infrastructure.

Action items: redeploy ElevatedIQ Cloud Run with the fixed image; provision or mock infrastructure (Ollama/Qdrant/DB/Redis) or harden degraded mode; re-run Tier 2.

---

## Test Configuration

| Parameter            | Tier 2 (Before Fix)                                       | Tier 2 (After Fix – GovAI-Scout)                          | Notes                               |
| -------------------- | --------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------- |
| **Duration**         | 10 minutes                                                | 5 minutes                                                 | Sanity run to verify fix under load |
| **Concurrent Users** | 50                                                        | 50                                                        | Same user model (Locust)            |
| **Spawn Rate**       | 5 users/sec                                               | 5 users/sec                                               | Same traffic pattern                |
| **Endpoints**        | health, models, generate                                  | health, models, generate                                  | Same weights                        |
| **Host**             | `https://ollama-service-794896362693.us-central1.run.app` | `https://ollama-service-131055855980.us-central1.run.app` | ElevatedIQ vs GovAI-Scout           |

---

## Tier 2 – Before Fix (ElevatedIQ Cloud Run)

Source: `load_test_tier2_results_stats.csv` and `load_test_tier2_results_failures.csv`

- GET `/api/v1/models`: 2,447 requests, 2,447 failures (100% 404)
- POST `/api/v1/generate`: 7,249 requests, 0 failures
- GET `/health`: 4,749 requests, 0 failures

Summary: Core endpoints healthy; models endpoint missing → 16.9% overall error rate driven entirely by `/api/v1/models`.

---

## Tier 2 – After Fix (GovAI-Scout Cloud Run)

Source: `load_test_tier2_after_fix_stats.csv` and `load_test_tier2_after_fix_failures.csv`

- GET `/api/v1/models`: 1,169 requests, 1,078 failures (≈92% 500)
- POST `/api/v1/generate`: 3,555 requests, 3,555 failures (500/503)
- GET `/health`: 2,403 requests, 0 failures
- Aggregated: 7,127 requests, 4,633 failures (≈65% error rate)

Direct checks:

- `curl https://ollama-service-131055855980.us-central1.run.app/api/v1/models` → 200 OK
- `curl https://ollama-service-794896362693.us-central1.run.app/api/v1/models` → 404 NOT FOUND

Interpretation:

- The models route is present and returns 200 when lightly exercised, but under load the service returns 500. The generate route returns 500/503 consistently without the Ollama inference engine provisioned.
- Health endpoint remains stable (0 failures), indicating the app is up but feature-degraded.

---

## Root Causes and Hypotheses

- ElevatedIQ Cloud Run uses an older image without the auth module rename → missing `/api/v1/models` route (404).
- GovAI-Scout Cloud Run is running the fixed image with optionalized services, but:
  - `/api/v1/generate` depends on the Ollama model server; without it, requests return 500/503 under load.
  - `/api/v1/models` likely attempts to query the inference backend or shared state, which fails under load without the model service.
  - Concurrency and resource limits (Cloud Run, 1 vCPU, concurrency ~80) may amplify error rates when calls cascade into unavailable backends.

---

## Recommendations (Immediate)

1. Redeploy ElevatedIQ Cloud Run (`…794896362693`) with the fixed image so `/api/v1/models` is registered and returns 200.
2. For GovAI-Scout (`…131055855980`), choose one:
   - Provision infrastructure: Ollama model server, Qdrant, Redis, Cloud SQL; set env vars; re-run Tier 2.
   - Harden degraded mode: ensure `/api/v1/models` always returns a stub list and `/api/v1/generate` returns a graceful 503 with clear error (or stub) without throwing 500.
   - Reduce concurrency and set Cloud Run min instances > 1 to mitigate cold-start or saturation.
3. Align testing with architecture mandates: run through the GCP Load Balancer path `https://elevatediq.ai/ollama` once the fixed image is deployed and LB routing is confirmed.

---

## Proposed Re-Test Plan

- Step 1: Push fixed image to ElevatedIQ registry and deploy Cloud Run revision.
- Step 2: Verify `/api/v1/models` returns 200 on ElevatedIQ endpoint.
- Step 3: If degraded mode persists, adjust `/api/v1/models` to stub without backend dependency; make `/api/v1/generate` reject quickly with 503 (no 500).
- Step 4: Run Tier 2 (50 users, 10 minutes) via LB URL; capture CSVs and update this analysis.

---

## Evidence Files (Added)

- `load_test_tier2_after_fix_stats.csv`
- `load_test_tier2_after_fix_failures.csv`

Existing (Before Fix):

- `load_test_tier2_results_stats.csv`
- `load_test_tier2_results_failures.csv`

---

## Conclusion

- The original 404 on `/api/v1/models` is resolved in the fixed image (verified via 200 on GovAI-Scout endpoint) but not yet deployed to ElevatedIQ Cloud Run.
- Under load, the service in degraded mode is unstable for model listing and generation (500/503). Stabilization requires either backend provisioning or graceful fallbacks.
- Next actions are deployment alignment and infrastructure decisions per the architecture mandate, followed by re-running Tier 2.

**Updated**: January 14, 2026
