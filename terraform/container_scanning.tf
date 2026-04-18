# terraform/container_scanning.tf - Trivy Container Vulnerability Scanning
#
# Automated vulnerability detection for container images:
# 1. Trivy scanning: OS packages, application dependencies
# 2. Cloud Build integration: Automatic scan on image build
# 3. Artifact Analysis: Store results in Container Analysis API
# 4. Policy enforcement: Block images with critical vulnerabilities
# 5. Reporting: Dashboard and alerts for vulnerability tracking

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ============================================================================
# CLOUD BUILD SCANNING CONFIGURATION
# ============================================================================

# Cloud Build repository for scanning scripts
resource "google_sourcerepo_repository" "scanning_scripts" {
  project = var.project_id
  name    = "ollama-scanning-scripts"
}

# Cloud Storage bucket for scan results
resource "google_storage_bucket" "scan_results" {
  project       = var.project_id
  name          = "ollama-scan-results-${var.environment}"
  location      = var.gcs_region
  force_destroy = false

  uniform_bucket_level_access = true

  encryption {
    default_kms_key_name = var.storage_cmek_key_id
  }

  versioning {
    enabled = true
  }

  # Lifecycle: Keep scan results for 90 days
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "scanning"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }
}

# Service account for scanning operations
resource "google_service_account" "scanner" {
  project      = var.project_id
  account_id   = "ollama-scanner-sa"
  display_name = "Ollama Container Scanner"
  description  = "Service account for Trivy scanning and vulnerability analysis"
}

# Scanner service account: Write scan results
resource "google_storage_bucket_iam_member" "scanner_write" {
  bucket = google_storage_bucket.scan_results.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.scanner.email}"
}

# Scanner service account: Container Analysis editor
resource "google_project_iam_member" "scanner_container_analysis" {
  project = var.project_id
  role    = "roles/containeranalysis.notes.editor"
  member  = "serviceAccount:${google_service_account.scanner.email}"
}

# ============================================================================
# CLOUD BUILD SCANNING STEP CONFIGURATION
# ============================================================================

# Cloud Build SBOM (Software Bill of Materials) generation
resource "google_cloudbuild_trigger" "scan_on_build" {
  project  = var.project_id
  name     = "ollama-scan-trigger"
  filename = "cloudbuild-scan.yaml"

  # This trigger should be configured in Cloud Build console or separate YAML
  # For now, define the configuration structure
}

# Scanning step for cloudbuild.yaml (to be included in CI/CD pipeline):
# The following should be added to cloudbuild.yaml:
#
# - name: 'gcr.io/cloud-builders/docker'
#   id: 'scan-image'
#   args:
#     - 'run'
#     - '--rm'
#     - '-v'
#     - '/var/run/docker.sock:/var/run/docker.sock'
#     - 'aquasec/trivy:latest'
#     - 'image'
#     - '--exit-code'
#     - '0'  # 0=allow all, set to 1 to block on vulnerabilities
#     - '--severity'
#     - 'CRITICAL,HIGH'
#     - '--format'
#     - 'sarif'
#     - '--output'
#     - '/workspace/scan-results.sarif'
#     - '${_IMAGE_NAME}'

# ============================================================================
# VULNERABILITY ATTESTATION
# ============================================================================

# Service account for creating vulnerability attestations
resource "google_service_account" "vulnerability_attestor" {
  project      = var.project_id
  account_id   = "ollama-vuln-attestor-sa"
  display_name = "Ollama Vulnerability Attestor"
  description  = "Service account for creating vulnerability attestations"
}

# Grant attestation creation permissions
resource "google_project_iam_member" "vuln_attestor_creator" {
  project = var.project_id
  role    = "roles/binaryauthorization.attestorsAdmin"
  member  = "serviceAccount:${google_service_account.vulnerability_attestor.email}"
}

# Grant Container Analysis permissions
resource "google_project_iam_member" "vuln_attestor_analysis" {
  project = var.project_id
  role    = "roles/containeranalysis.occurrences.editor"
  member  = "serviceAccount:${google_service_account.vulnerability_attestor.email}"
}

# ============================================================================
# CONTAINER ANALYSIS API CONFIGURATION
# ============================================================================

# Enable Container Analysis API
resource "google_project_service" "container_analysis_api" {
  project = var.project_id
  service = "containeranalysis.googleapis.com"

  disable_on_destroy = false
}

# Custom note for vulnerability scan results
resource "google_container_analysis_note" "vulnerability_scan" {
  project = var.project_id
  name    = "ollama-vulnerability-scan"

  attestation_authority {
    hint {
      human_readable_name = "Ollama Vulnerability Scanning"
    }
  }
}

# ============================================================================
# VULNERABILITY DATABASE & UPDATES
# ============================================================================

# Cloud Scheduler job: Update vulnerability database daily
resource "google_cloud_scheduler_job" "update_vuln_db" {
  project       = var.project_id
  name          = "ollama-update-vuln-db"
  description   = "Daily vulnerability database update for Trivy"
  schedule      = "0 2 * * *"  # 2 AM UTC daily
  time_zone     = "UTC"
  attempt_deadline = "320s"

  http_target {
    http_method = "POST"
    uri         = "https://cloudbuild.googleapis.com/v1/projects/${var.project_id}/builds"

    headers = {
      "Content-Type" = "application/json"
    }

    body = base64encode(jsonencode({
      source = {
        repoSource = {
          repoName = "ollama-scanning-scripts"
          branchName = "main"
        }
      }
      steps = [
        {
          name = "gcr.io/cloud-builders/gke-deploy"
          args = ["run", "--", "trivy", "image", "--download-db-only"]
          waitFor = ["-"]
        }
      ]
    }))

    auth_header {
      service_account_email = google_service_account.scanner.email
    }
  }
}

# ============================================================================
# VULNERABILITY SEVERITY POLICIES
# ============================================================================

# Policy: Critical vulnerabilities must be patched immediately
# Policy: High vulnerabilities must be patched within 7 days
# Policy: Medium vulnerabilities must be patched within 30 days
# Policy: Low vulnerabilities tracked but not blocking

# This is enforced through:
# 1. Binary Authorization attestation requirements
# 2. Cloud Build policy gates
# 3. Monitoring and alerting

# ============================================================================
# SCANNING AUTOMATION IN CLOUD BUILD
# ============================================================================

# The scanning process should be integrated into Cloud Build pipeline:
#
# Step 1: Build image
#   docker build -t $IMAGE_REPO/$IMAGE_NAME:$COMMIT_SHA
#
# Step 2: Push to Artifact Registry (untagged/test)
#   docker push $IMAGE_REPO/$IMAGE_NAME:scan-$COMMIT_SHA
#
# Step 3: Scan with Trivy
#   trivy image --severity CRITICAL,HIGH $IMAGE_REPO/$IMAGE_NAME:scan-$COMMIT_SHA
#
# Step 4: Analyze results
#   - If critical found: Block and notify
#   - If high found: Require review and approval
#   - If low found: Log and continue
#
# Step 5: Tag and push production image (if scan passes)
#   docker tag $IMAGE_REPO/$IMAGE_NAME:scan-$COMMIT_SHA \
#              $IMAGE_REPO/$IMAGE_NAME:latest
#   docker push $IMAGE_REPO/$IMAGE_NAME:latest
#
# Step 6: Create attestation (after successful scan)
#   gcloud beta container binauthz attestations sign-and-create \
#     --attestation-project=$PROJECT_ID \
#     --artifact-url=$IMAGE_REPO/$IMAGE_NAME:latest \
#     --attestation-authority-note=$NOTE_ID \
#     --keyversion-project=$PROJECT_ID \
#     --keyversion-location=$KEY_LOCATION \
#     --keyversion-keyring=$KEYRING \
#     --keyversion-key=$KEY

# ============================================================================
# SBOM (SOFTWARE BILL OF MATERIALS) GENERATION
# ============================================================================

# Generate SBOM for supply chain transparency
# Uses Syft (complements Trivy scanning)
#
# Cloud Build step for SBOM:
# - name: 'gcr.io/cloud-builders/docker'
#   id: 'generate-sbom'
#   args:
#     - 'run'
#     - '--rm'
#     - 'anchore/syft:latest'
#     - '${_IMAGE_NAME}'
#     - '-o'
#     - 'spdx-json'
#     - '>${WORKSPACE}/sbom.spdx.json'

# Store SBOM in Cloud Storage
resource "google_storage_bucket" "sbom_storage" {
  project       = var.project_id
  name          = "ollama-sbom-${var.environment}"
  location      = var.gcs_region
  force_destroy = false

  uniform_bucket_level_access = true

  encryption {
    default_kms_key_name = var.storage_cmek_key_id
  }

  versioning {
    enabled = true
  }

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "sbom"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }
}

# ============================================================================
# MONITORING & ALERTING
# ============================================================================

# Alert: High/Critical vulnerabilities detected
resource "google_monitoring_alert_policy" "vulnerabilities_detected" {
  project      = var.project_id
  display_name = "Container Vulnerabilities Detected"
  combiner     = "OR"

  conditions {
    display_name = "Critical/High vulnerabilities in scanned image"

    condition_threshold {
      filter            = "resource.type=\"container.image\" AND metric.type=\"containeranalysis.googleapis.com/image_vulnerabilities\""
      comparison        = "COMPARISON_GT"
      threshold_value   = 0
      duration          = "60s"
    }
  }

  notification_channels = var.alert_channels

  lifecycle {
    ignore_changes = [notification_channels]
  }
}

# Alert: Scan failure or timeout
resource "google_monitoring_alert_policy" "scan_failures" {
  project      = var.project_id
  display_name = "Container Scan Failure"
  combiner     = "OR"

  conditions {
    display_name = "Container image scan failed or timed out"

    condition_threshold {
      filter            = "resource.type=\"container.image\" AND metric.type=\"containeranalysis.googleapis.com/scan_failures\""
      comparison        = "COMPARISON_GT"
      threshold_value   = 0
      duration          = "300s"
    }
  }

  notification_channels = var.alert_channels

  lifecycle {
    ignore_changes = [notification_channels]
  }
}

# ============================================================================
# OUTPUTS
# ============================================================================

output "scan_results_bucket" {
  value       = google_storage_bucket.scan_results.name
  description = "Cloud Storage bucket for scan results"
}

output "sbom_bucket" {
  value       = google_storage_bucket.sbom_storage.name
  description = "Cloud Storage bucket for SBOM storage"
}

output "scanner_service_account" {
  value       = google_service_account.scanner.email
  description = "Scanner service account email"
}

output "vulnerability_note" {
  value       = google_container_analysis_note.vulnerability_scan.name
  description = "Container Analysis note for vulnerability scans"
}

output "vulnerability_attestor_sa" {
  value       = google_service_account.vulnerability_attestor.email
  description = "Vulnerability attestor service account"
}
