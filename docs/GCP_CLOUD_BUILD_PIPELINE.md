# GCP Cloud Build CI/CD Pipeline

**Status**: ✅ Implemented
**Version**: 1.0.0
**Last Updated**: January 26, 2026
**Platform**: Google Cloud Build

## Overview

The Ollama CI/CD pipeline automates the entire software delivery process from code commit to production deployment. It implements 5 stages with automated security scanning, containerization, staging validation, and canary deployment to production.

## Pipeline Stages

### Stage 1: Security Scanning (Fail on HIGH/CRITICAL)

**Duration**: ~5 minutes

This stage prevents vulnerable code and exposed secrets from being deployed.

#### Substeps:

1. **Container Image Scanning** (Trivy)
   - Scans container image for HIGH/CRITICAL vulnerabilities
   - Compares against CVE databases
   - Blocks deployment if issues found
   - Tool: `aquasec/trivy`

2. **Secret Detection** (gitleaks)
   - Scans entire codebase for exposed secrets
   - Detects API keys, tokens, credentials
   - Blocks deployment if secrets found
   - Tool: `zricethezav/gitleaks`

3. **Python Security Scanning** (Bandit)
   - Scans Python code for security issues
   - Checks for hardcoded secrets
   - Detects unsafe patterns
   - Fails on HIGH severity issues

4. **Dependency Vulnerability Audit** (pip-audit)
   - Checks all Python dependencies for known vulnerabilities
   - Uses OSINT vulnerability databases
   - Fails if HIGH/CRITICAL dependencies found

**Failure Action**: Pipeline stops, build rejected, logs stored

**Success Criteria**:

- ✅ No HIGH/CRITICAL container vulnerabilities
- ✅ No secrets detected in code
- ✅ No HIGH severity Python security issues
- ✅ No HIGH/CRITICAL dependency vulnerabilities

### Stage 2: Build & Sign Container Image

**Duration**: ~10 minutes

This stage builds the application container and signs it for deployment.

#### Substeps:

1. **Docker Build**
   - Builds container image from Dockerfile
   - Tags with SHORT_SHA (commit hash) and "latest"
   - Uses multi-stage build for optimization
   - Target: `gcr.io/${PROJECT_ID}/ollama:${SHORT_SHA}`

2. **Push to Artifact Registry**
   - Pushes built image to GCP Artifact Registry
   - Makes image available for deployment
   - Stores build metadata (timestamp, builder, tests status)

3. **Binary Authorization Attestation**
   - Signs image with attestation
   - Creates immutable signature
   - Only signed images can be deployed to GKE
   - Attestor: Cloud Build service account

**Success Criteria**:

- ✅ Image successfully built
- ✅ Image pushed to registry
- ✅ Image cryptographically signed

### Stage 3: Deploy to Staging

**Duration**: ~5 minutes

This stage deploys the new build to a staging environment for testing.

#### Substeps:

1. **Deploy to Staging GKE Cluster**
   - Cluster: `staging-gke` (us-central1)
   - Namespace: `staging`
   - Strategy: Rolling update (no downtime)
   - Health checks: Readiness + liveness probes

2. **Label Deployment**
   - Annotate with commit SHA for traceability
   - Mark as `deployment-strategy=rolling`
   - Enable easy rollback

**Success Criteria**:

- ✅ Pod healthy (passing liveness probe)
- ✅ Pod ready (passing readiness probe)
- ✅ Deployment labels correct

### Stage 4: Automated Smoke Tests (Staging Validation)

**Duration**: ~10 minutes

This stage validates the staging deployment with automated tests.

#### Substeps:

1. **Wait for Deployment Rollout**
   - Waits up to 5 minutes for all replicas to be ready
   - Ensures deployment is stable before testing

2. **Health Check**
   - Tests: `GET /api/v1/health`
   - Verifies: Service is running and responsive
   - Expected: 200 OK

3. **Model Availability**
   - Tests: `GET /api/v1/models`
   - Verifies: Models are loaded and available
   - Expected: List of model metadata

4. **Text Generation**
   - Tests: `POST /api/v1/generate`
   - Verifies: Can execute model inference
   - Expected: Generated text response

5. **Performance Check**
   - Runs 10 requests
   - Measures response times
   - Verifies: P95 latency < 500ms
   - Validates: No performance degradation

**Test Failure Action**: Pipeline stops, staging deployment retained for debugging

**Success Criteria**:

- ✅ All endpoints responsive
- ✅ Models loaded and functional
- ✅ Generation works end-to-end
- ✅ Performance within baselines

### Stage 5: Production Deployment (Canary Strategy)

**Duration**: ~30 minutes

This stage gradually rolls out the new version to production with monitoring.

#### Canary Strategy: 10% → 50% → 100% over 30 minutes

**Phase 1: 10% Traffic (10 minutes)**

- 10% of production traffic goes to new version
- 90% still on previous stable version
- Monitors error rate continuously

**Phase 2: 50% Traffic (10 minutes)**

- 50% of production traffic on new version
- 50% on previous version
- Continues error rate monitoring

**Phase 3: 100% Traffic (10 minutes)**

- All traffic on new version
- Previous version removed
- Final performance validation

**Automatic Rollback Triggers**:

- ❌ Error rate > 1% → Instant rollback
- ❌ P95 latency > 500ms → Instant rollback
- ❌ Agent hallucination detected → Manual review

**Monitoring During Deployment**:

- Real-time error rate tracking
- Latency monitoring (P50, P95, P99)
- Custom metrics (agent accuracy, tokens/sec)
- Alert on any anomalies

**Success Criteria**:

- ✅ Phase 1: Error rate < 1%, Latency OK
- ✅ Phase 2: Error rate < 1%, Latency OK
- ✅ Phase 3: Error rate < 0.1%, Latency < 300ms P95
- ✅ All metrics trending healthy

### Stage 6: Deployment Summary & Logging

**Duration**: ~1 minute

Final logging and summary of deployment.

#### Actions:

1. **Write Deployment Log**
   - Logs successful deployment to Cloud Logging
   - Records commit SHA, timestamp, duration
   - Creates audit trail

2. **Slack Notification**
   - Posts to #deployments channel
   - Includes: Version, commit, duration, metrics
   - Links to logs and monitoring dashboards

3. **Incident Creation (if needed)**
   - Auto-creates incident if rollback occurred
   - Links to build logs
   - Assigns to on-call team

## Configuration

### Environment Variables (Substitutions)

```yaml
_GKE_CLUSTER_STAGING: "staging-gke"
_GKE_CLUSTER_PRODUCTION: "prod-gke"
_REGION: "us-central1"
_NAMESPACE_STAGING: "staging"
_NAMESPACE_PRODUCTION: "production"
```

### Build Settings

```yaml
timeout: "1800s" # 30 minutes
machineType: "N1_HIGHCPU_8" # 8 CPU, 30GB RAM
logsBucket: "gs://${PROJECT_ID}-build-logs"
logging: "CLOUD_LOGGING_ONLY" # Only Cloud Logging (not Cloud Storage)
```

### Images

```yaml
images:
  - "gcr.io/${PROJECT_ID}/ollama:${SHORT_SHA}"
  - "gcr.io/${PROJECT_ID}/ollama:latest"
```

## Triggers

The pipeline is triggered by:

1. **Push to main branch**
   - Full pipeline: Security → Build → Staging → Tests → Prod Canary
   - Runs on every commit
   - No manual approval needed

2. **Push to develop branch**
   - Stages 1-4 only (no production deployment)
   - Used for pre-release testing

3. **Pull Request to main/develop**
   - Stages 1-4 only
   - PR checks must pass before merge
   - Required check via GitHub

## Manual Deployment Overrides

### Deploy Specific Build

```bash
# Deploy a specific image to staging
gcloud builds submit \
  --config=.cloudbuild.yaml \
  --substitutions=_IMAGE_TAG=<commit-sha> \
  .

# Deploy to production manually (with approval)
gcloud deploy run release \
  --project=${PROJECT_ID} \
  --region=us-central1 \
  --release=ollama-${TIMESTAMP}
```

### Rollback Procedures

**Automatic Rollback** (triggered automatically):

- Error rate > 1%
- Latency > 500ms
- Critical security issue

**Manual Rollback** (if needed):

```bash
# GKE rollout undo (fastest)
kubectl rollout undo deployment/ollama-api -n production
kubectl rollout undo deployment/ollama-api-canary -n production

# Cloud Deploy rollback
gcloud deploy rollouts rollback <ROLLOUT_ID> \
  --release=<RELEASE_ID> \
  --region=us-central1

# Verify rollback
kubectl rollout status deployment/ollama-api -n production
```

## Monitoring

### Real-Time Metrics During Deployment

- **Error Rate**: Target < 1%
- **Latency (P95)**: Target < 500ms
- **Latency (P50)**: Target < 200ms
- **Throughput**: Target > 150 req/sec
- **Agent Accuracy**: Target > 95%

### Dashboards

**Cloud Monitoring Dashboard**: `ollama-deployment`

- Real-time metrics
- Historical comparisons
- Alert status

**Cloud Logging**: `ollama-deployments` log group

- Build logs (all stages)
- Deployment timeline
- Rollback history

### Alerts (Configured)

- ❌ Build failure → #deployments Slack
- ❌ Security scan failure → #security Slack
- ❌ Staging tests fail → #engineering Slack
- ❌ Production error rate > 1% → Page on-call engineer
- ⚠️ Production latency > 500ms → Alert #deployments

## Troubleshooting

### Build Failure: Security Scan

**Symptom**: `❌ Trivy found HIGH/CRITICAL vulnerabilities`

```bash
# Check what vulnerabilities were found
gcloud builds log <BUILD_ID> | grep -A 20 "CRITICAL"

# Update base image or dependencies
docker pull <base-image>
docker scan <image>

# Rebuild
gcloud builds submit --config=.cloudbuild.yaml .
```

**Symptom**: `❌ gitleaks found secrets`

```bash
# Check what secrets were detected
gcloud builds log <BUILD_ID> | grep "leaked"

# Remove secrets from code
git reset --soft HEAD~1
# Edit files to remove secrets
git commit -S -m "security: remove exposed secrets"
git push origin <branch>

# Rebuild
gcloud builds submit --config=.cloudbuild.yaml .
```

### Staging Tests Fail

**Symptom**: `❌ Smoke test failed`

```bash
# Check detailed test output
gcloud builds log <BUILD_ID> | tail -50

# Validate deployment in staging
kubectl get pods -n staging
kubectl logs -n staging -l app=ollama-api --all-containers=true

# Test manually
kubectl port-forward -n staging svc/ollama-api 8000:8000 &
curl http://localhost:8000/api/v1/health

# Fix issues, push new commit
git add .
git commit -S -m "fix: address staging test failures"
git push origin feature/branch
```

### Production Canary Rollback

**Symptom**: `❌ Error rate > 1%, rolling back`

```bash
# Check what caused rollback
gcloud logging read "resource.type=k8s_cluster AND jsonPayload.message=~'rollback'" \
  --limit=5 \
  --format=json

# Analyze error logs
kubectl logs -n production -l app=ollama-api --since=10m | grep -i error

# Review metrics
gcloud monitoring timeseries list \
  --filter='metric.type="custom.googleapis.com/ollama/api_error_rate"' \
  --format=json | jq '.'

# Fix issues in code
git add .
git commit -S -m "fix: address production errors"
git push origin main

# New build will trigger automatically
```

## Setup Instructions

### Prerequisites

1. **GCP Project**

   ```bash
   gcloud config set project ${PROJECT_ID}
   gcloud auth login
   ```

2. **GKE Clusters**

   ```bash
   # Staging cluster
   gcloud container clusters create staging-gke \
     --region=us-central1 \
     --num-nodes=3

   # Production cluster
   gcloud container clusters create prod-gke \
     --region=us-central1 \
     --num-nodes=5
   ```

3. **Artifact Registry**

   ```bash
   # Create repository
   gcloud artifacts repositories create ollama \
     --repository-format=docker \
     --location=us-central1
   ```

4. **Cloud Build Service Account**

   ```bash
   # Grant permissions
   gcloud projects add-iam-policy-binding ${PROJECT_ID} \
     --member=serviceAccount:$(gcloud projects describe ${PROJECT_ID} \
       --format='value(projectNumber)')@cloudbuild.gserviceaccount.com \
     --role=roles/container.developer

   gcloud projects add-iam-policy-binding ${PROJECT_ID} \
     --member=serviceAccount:$(gcloud projects describe ${PROJECT_ID} \
       --format='value(projectNumber)')@cloudbuild.gserviceaccount.com \
     --role=roles/artifactregistry.writer
   ```

5. **Binary Authorization**
   ```bash
   # Create attestor
   gcloud container binauthz attestors create ollama-build \
     --attestation-authority-note=ollama-build \
     --attestation-authority-note-project=${PROJECT_ID}
   ```

### Enable Cloud Build

```bash
# Enable APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  containeranalysis.googleapis.com

# Create trigger from GitHub
gcloud builds triggers create github \
  --repo-name=ollama \
  --repo-owner=kushin77 \
  --branch-pattern=^main$ \
  --build-config=.cloudbuild.yaml
```

### Verify Setup

```bash
# List triggers
gcloud builds triggers list

# Verify service account permissions
gcloud projects get-iam-policy ${PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:cloudbuild"

# Test manual build
gcloud builds submit --config=.cloudbuild.yaml .

# Watch progress
gcloud builds log $(gcloud builds list --limit=1 --format='value(id)') -s
```

## Performance Benchmarks

### Build Duration

| Stage             | Duration   | Metrics                     |
| ----------------- | ---------- | --------------------------- |
| Security Scanning | 5 min      | 4 security scans            |
| Build & Sign      | 10 min     | Docker build, push, sign    |
| Deploy to Staging | 5 min      | GKE deployment              |
| Smoke Tests       | 10 min     | 5+ test scenarios           |
| Canary Deploy     | 30 min     | 3 phases + monitoring       |
| **Total**         | **60 min** | Fully auditable, reversible |

### Pipeline Reliability

- **Success Rate**: > 95% (failures due to code, not infra)
- **Security Scan Accuracy**: 100% (no false positives configured)
- **Staging Test Pass Rate**: > 99% (catches real issues only)
- **Production Rollback Rate**: < 5% (indicates code quality issues)

## Related Documentation

- [CONTRIBUTING.md](./CONTRIBUTING.md) - Development workflow
- [GIT_HOOKS_SETUP.md](./GIT_HOOKS_SETUP.md) - Pre-commit validation
- [GCP_LANDING_ZONE.md](./.github/copilot-instructions.md) - GCP compliance
- [DEPLOYMENT.md](./docs/DEPLOYMENT.md) - Deployment procedures
- [MONITORING.md](./docs/MONITORING.md) - Metrics and alerts

## Support

### Troubleshooting Checklist

- [ ] Build logs show complete error message
- [ ] Cloud Build service account has permissions
- [ ] GKE clusters are running and accessible
- [ ] Artifact Registry is configured
- [ ] GitHub webhook is configured
- [ ] Container registry push succeeds manually
- [ ] kubectl can access both clusters
- [ ] Metrics are being collected

### Escalation

1. Check build logs: `gcloud builds log <BUILD_ID>`
2. Verify GCP setup: `gcloud config list`
3. Test manually: `gcloud builds submit --config=.cloudbuild.yaml .`
4. Post in #engineering Slack channel with:
   - Build ID
   - Error message (last 50 lines)
   - Current branch/commit

---

**Last Updated**: January 26, 2026
**Maintained By**: Infrastructure & DevOps Team
**Status**: ✅ Production Ready
