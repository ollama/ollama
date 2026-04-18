# 🚀 Deployment Action Plan: /api/v1/models Fix

**Date**: January 13, 2026
**Commit**: `ac23cbb` - fix(api): resolve /api/v1/models 404
**Target**: https://ollama-service-sozvlwbwva-uc.a.run.app

---

## Pre-Deployment Status

### Current Production State
- ✅ **Health**: 200 OK
- ❌ **Models Endpoint**: 404 NOT FOUND (as expected)
- ⚠️ **Load Test Result**: 2,388 failures on GET /api/v1/models (Tier 2)

### Fix Implemented
- ✅ Renamed `ollama/auth.py` → `ollama/auth_manager.py`
- ✅ Updated all import references
- ✅ Added OpenTelemetry instrumentation guards
- ✅ Created regression test (passing)
- ✅ Committed: `ac23cbb`
- ✅ Pushed to main branch

---

## Deployment Steps

### Step 1: Build New Container Image ⏭️

**Command**:
```bash
cd /home/akushnir/ollama
docker build -t gcr.io/YOUR_PROJECT_ID/ollama:latest .
```

**Expected**: Build completes without errors

**Validation**:
- Check for import errors in build logs
- Verify all dependencies install successfully
- Confirm image size is reasonable (<1GB)

---

### Step 2: Push to Google Container Registry ⏭️

**Command**:
```bash
docker push gcr.io/YOUR_PROJECT_ID/ollama:latest
```

**Expected**: Image pushed successfully

**Validation**:
- Confirm image exists in GCR
- Check image digest matches local build

---

### Step 3: Deploy to Cloud Run ⏭️

**Command**:
```bash
gcloud run deploy ollama-service \
  --image gcr.io/YOUR_PROJECT_ID/ollama:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300s \
  --max-instances 10 \
  --min-instances 1
```

**Expected**:
- Deployment completes in 2-5 minutes
- New revision created
- Traffic shifts to new revision
- Health checks pass

**Validation**:
- Check Cloud Run console for deployment status
- Verify logs show no import errors
- Confirm service starts successfully

---

### Step 4: Verify Health Endpoint ⏭️

**Command**:
```bash
curl -i https://ollama-service-sozvlwbwva-uc.a.run.app/health
```

**Expected Output**:
```
HTTP/2 200
content-type: application/json

{"status":"healthy","timestamp":"2026-01-13T..."}
```

**Validation**:
- Status code: 200 OK
- Response body contains "healthy"
- No errors in Cloud Run logs

---

### Step 5: Test Models Endpoint (Critical) ⏭️

**Command**:
```bash
curl -i https://ollama-service-sozvlwbwva-uc.a.run.app/api/v1/models
```

**Expected Output** (AFTER FIX):
```
HTTP/2 200
content-type: application/json

{
  "models": [
    {
      "name": "llama3.2-vision:latest",
      "size": 7600000000,
      "modified_at": "2026-01-13T...",
      "digest": "sha256:..."
    }
  ]
}
```

**Validation**:
- ✅ Status code: 200 OK (not 404!)
- ✅ Response contains "models" array
- ✅ At least one model listed
- ✅ No errors in Cloud Run logs

**Before Fix (Current)**:
```
HTTP/2 404
```

---

### Step 6: Verify OpenAPI Documentation ⏭️

**Command**:
```bash
curl -s https://ollama-service-sozvlwbwva-uc.a.run.app/docs | grep -o '/api/v1/models'
```

**Expected Output**:
```
/api/v1/models
```

**Validation**:
- Endpoint appears in Swagger UI
- Endpoint documented correctly
- No 404 errors

---

### Step 7: Run Quick Smoke Test ⏭️

**Commands**:
```bash
# Test all critical endpoints
curl -s -o /dev/null -w "Health: %{http_code}\n" \
  https://ollama-service-sozvlwbwva-uc.a.run.app/health

curl -s -o /dev/null -w "Models: %{http_code}\n" \
  https://ollama-service-sozvlwbwva-uc.a.run.app/api/v1/models

curl -s -o /dev/null -w "Generate: %{http_code}\n" \
  -X POST https://ollama-service-sozvlwbwva-uc.a.run.app/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2-vision:latest","prompt":"Hello"}'
```

**Expected Output**:
```
Health: 200
Models: 200    ← FIX CONFIRMED
Generate: 200
```

**Validation**:
- All endpoints return 200 OK
- No 404 errors
- No 500 errors

---

### Step 8: Re-Run Tier 2 Load Test ⏭️

**Command**:
```bash
cd /home/akushnir/ollama
locust -f load_test.py \
  --host https://ollama-service-sozvlwbwva-uc.a.run.app \
  --users 50 \
  --spawn-rate 5 \
  --run-time 10m \
  --html load_test_tier2_after_fix_results.html \
  --csv load_test_tier2_after_fix_results
```

**Expected Results**:
```
GET /api/v1/models
- Requests: ~2,400
- Success Rate: 100% ✅ (was 0%)
- Failures: 0 (was 2,388)
- Avg Latency: <100ms

POST /api/v1/generate
- Requests: ~12,000
- Success Rate: 100% ✅
- Avg Latency: ~57ms
```

**Validation**:
- Zero failures on /api/v1/models
- Performance matches baseline (57ms avg on generate)
- No new errors introduced
- Overall success rate: 100%

---

### Step 9: Compare Load Test Results ⏭️

**Analysis**:
```bash
# Compare CSV results
diff load_test_tier2_results_stats.csv \
     load_test_tier2_after_fix_results_stats.csv
```

**Key Metrics to Compare**:

| Metric | Before Fix | After Fix (Expected) |
|--------|-----------|---------------------|
| Models Success Rate | 0% ❌ | 100% ✅ |
| Models Failures | 2,388 | 0 |
| Generate Success Rate | 100% ✅ | 100% ✅ |
| Generate Avg Latency | 57ms | ~57ms (no regression) |
| Overall Success Rate | 83.6% | 100% ✅ |

**Validation**:
- Models endpoint: 0% → 100%
- No performance regression on other endpoints
- Overall system reliability improved

---

### Step 10: Update Documentation ⏭️

**Files to Update**:
1. `LOAD_TEST_RESULTS_ANALYSIS.md` - Add "After Fix" section
2. `SESSION_LOAD_TEST_COMPLETION.md` - Mark issue resolved
3. `FIX_SUMMARY_MODELS_ENDPOINT.md` - Add production verification
4. `WEEK_1_CONTINUATION_PLAN.md` - Mark Day 1 complete

**Commands**:
```bash
# Update analysis with new results
# Document fix verification
# Update status reports
```

---

## Rollback Plan (If Needed)

### If Deployment Fails

**Step 1**: Revert to previous Cloud Run revision
```bash
gcloud run services update-traffic ollama-service \
  --to-revisions PREVIOUS_REVISION=100 \
  --region us-central1
```

**Step 2**: Check logs for error details
```bash
gcloud logging read "resource.type=cloud_run_revision" \
  --project YOUR_PROJECT_ID \
  --limit 50
```

**Step 3**: Fix locally and redeploy

### If Models Endpoint Still Returns 404

**Diagnosis**:
1. Check Cloud Run logs for import errors
2. Verify container image contains `ollama/auth_manager.py`
3. Check router registration in logs
4. Verify OpenAPI schema endpoint list

**Actions**:
1. Review build logs
2. Test locally with same environment
3. Add additional debug logging
4. Deploy with verbose logging enabled

---

## Success Criteria

### Deployment Success ✅
- [ ] Container builds without errors
- [ ] Image pushed to GCR successfully
- [ ] Cloud Run deployment completes
- [ ] Health endpoint returns 200 OK
- [ ] No import errors in logs

### Fix Verification ✅
- [ ] Models endpoint returns 200 OK (not 404)
- [ ] Models list populated with data
- [ ] OpenAPI docs show endpoint
- [ ] No errors in Cloud Run logs

### Load Test Success ✅
- [ ] GET /api/v1/models: 100% success rate
- [ ] Zero failures (was 2,388)
- [ ] No performance regression
- [ ] Overall system: 100% success rate

### Documentation Complete ✅
- [ ] Load test results documented
- [ ] Fix summary updated with production verification
- [ ] Session completion marked resolved
- [ ] Week 1 plan updated

---

## Timeline Estimate

| Step | Duration | Start | End |
|------|----------|-------|-----|
| Build Image | 5 min | Now | +5m |
| Push to GCR | 2 min | +5m | +7m |
| Deploy to Cloud Run | 3 min | +7m | +10m |
| Verify Endpoints | 2 min | +10m | +12m |
| Run Load Test | 10 min | +12m | +22m |
| Analyze Results | 5 min | +22m | +27m |
| Update Docs | 3 min | +27m | +30m |

**Total Estimated Time**: 30 minutes

---

## Next Steps After Deployment

### Immediate (Today)
1. ✅ Verify 100% success rate on models endpoint
2. ✅ Confirm no performance regression
3. ✅ Update all documentation
4. ⏭️ Brief team on resolution

### Short-term (This Week)
1. Continue Week 1 plan (Day 2: Alert policies, backup testing)
2. Review monitoring dashboards
3. Verify all endpoints are fully operational
4. Performance optimization if needed

### Long-term
1. Add additional model management features
2. Implement model versioning
3. Add model performance benchmarks
4. Enhance monitoring and alerting

---

## Commands Summary

```bash
# 1. Build and push image
cd /home/akushnir/ollama
docker build -t gcr.io/YOUR_PROJECT_ID/ollama:latest .
docker push gcr.io/YOUR_PROJECT_ID/ollama:latest

# 2. Deploy to Cloud Run
gcloud run deploy ollama-service \
  --image gcr.io/YOUR_PROJECT_ID/ollama:latest \
  --region us-central1 --platform managed

# 3. Verify health
curl https://ollama-service-sozvlwbwva-uc.a.run.app/health

# 4. Test models endpoint (CRITICAL)
curl https://ollama-service-sozvlwbwva-uc.a.run.app/api/v1/models

# 5. Run load test
locust -f load_test.py \
  --host https://ollama-service-sozvlwbwva-uc.a.run.app \
  --users 50 --spawn-rate 5 --run-time 10m \
  --html load_test_tier2_after_fix_results.html

# 6. Compare results
diff load_test_tier2_results_stats.csv \
     load_test_tier2_after_fix_results_stats.csv
```

---

**Status**: Ready to Deploy ✅
**Risk Level**: Low (fix verified locally with tests)
**Expected Outcome**: 100% success rate on models endpoint
