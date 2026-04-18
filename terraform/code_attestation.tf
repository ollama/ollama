# terraform/code_attestation.tf - Cryptographic Deployment Signing
#
# Code Attestation ensures deployments are cryptographically signed:
# 1. Build artifact signing: Sign container images after successful build
# 2. Deployment attestations: Prove image came from approved CI/CD pipeline
# 3. Audit trail: Record who signed what and when
# 4. Verification: GKE verifies signatures before deployment
#
# Integration with Binary Authorization:
# - Only signed, scanned images deploy to production
# - Unsigned/unapproved images automatically blocked
# - Full chain of custody from source code to running container

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
# ATTESTATION KEY MANAGEMENT (KMS)
# ============================================================================

# Create KMS key for attestation signing
resource "google_kms_crypto_key" "attestation_signing" {
  project           = var.project_id
  name              = "ollama-attestation-signing-key"
  key_ring          = var.kms_key_ring_id
  rotation_period   = "7776000s"  # 90 days
  version_template {
    algorithm       = "RSA_SIGN_PKCS1_4096_SHA512"
    protection_level = "HSM"  # Hardware Security Module
  }

  lifecycle {
    prevent_destroy = true
  }
}

# ============================================================================
# ATTESTATION AUTHORITY SETUP
# ============================================================================

# Attestation authority for image signing
resource "google_binary_authorization_attestor" "build_attestor" {
  project = var.project_id
  name    = "ollama-build-attestor"

  user_owned_grafeas_note {
    note_reference = google_container_analysis_note.build_attestation.name
  }

  # Define valid signing keys
  attestation_authority_note {
    note_reference = google_container_analysis_note.build_attestation.name

    public_keys {
      id = "ollama-ci-key-1"
      pkix_public_key {
        public_key_pem = tls_self_signed_cert.ci_signing_cert.cert_pem
        signature_algorithm = "RSA_SIGN_PKCS1_4096_SHA512"
      }
    }
  }
}

# Grafeas note for build attestations
resource "google_container_analysis_note" "build_attestation" {
  project = var.project_id
  name    = "ollama-build-attestation"

  attestation_authority {
    hint {
      human_readable_name = "Ollama CI/CD Build Attestation"
    }
  }
}

# ============================================================================
# CI/CD SIGNING KEY INFRASTRUCTURE
# ============================================================================

# Generate RSA key pair for CI/CD signing
resource "tls_private_key" "ci_signing_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Create self-signed certificate for CI/CD
resource "tls_self_signed_cert" "ci_signing_cert" {
  private_key_pem = tls_private_key.ci_signing_key.private_key_pem

  subject {
    common_name       = "Ollama CI/CD Build Signer"
    organization      = var.organization_name
    organizational_unit = "Platform Engineering"
    country           = "US"
    province          = "California"
    locality          = "San Francisco"
  }

  validity_period_hours = 87600  # 10 years
  allowed_uses = [
    "digital_signature",
    "key_encipherment",
  ]
}

# Store CI/CD signing key in Secret Manager
resource "google_secret_manager_secret" "ci_signing_key" {
  project   = var.project_id
  secret_id = "ollama-ci-signing-key-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "ci_signing_key" {
  secret      = google_secret_manager_secret.ci_signing_key.id
  secret_data = tls_private_key.ci_signing_key.private_key_pem
}

# Store CI/CD signing certificate in Secret Manager
resource "google_secret_manager_secret" "ci_signing_cert" {
  project   = var.project_id
  secret_id = "ollama-ci-signing-cert-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "ci_signing_cert" {
  secret      = google_secret_manager_secret.ci_signing_cert.id
  secret_data = tls_self_signed_cert.ci_signing_cert.cert_pem
}

# ============================================================================
# SERVICE ACCOUNTS FOR ATTESTATION
# ============================================================================

# Service account for Cloud Build to create attestations
resource "google_service_account" "build_attestor" {
  project      = var.project_id
  account_id   = "ollama-build-attestor-sa"
  display_name = "Ollama Build Attestor"
  description  = "Service account for Cloud Build to create image attestations"
}

# Grant Cloud Build attestation creation permissions
resource "google_project_iam_member" "build_attestation_creator" {
  project = var.project_id
  role    = "roles/binaryauthorization.attestorsAdmin"
  member  = "serviceAccount:${google_service_account.build_attestor.email}"
}

# Grant access to signing key
resource "google_secret_manager_iam_member" "build_attestor_key_access" {
  secret_id = google_secret_manager_secret.ci_signing_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.build_attestor.email}"
}

# Service account for attestation verification in GKE
resource "google_service_account" "attestation_verifier_gke" {
  project      = var.project_id
  account_id   = "ollama-attestation-verifier-gke-sa"
  display_name = "Ollama Attestation Verifier (GKE)"
  description  = "Service account for GKE to verify image attestations"
}

# Grant GKE attestation viewing permissions
resource "google_project_iam_member" "gke_attestation_viewer" {
  project = var.project_id
  role    = "roles/binaryauthorization.attestorsViewer"
  member  = "serviceAccount:${google_service_account.attestation_verifier_gke.email}"
}

# ============================================================================
# AUDIT LOGGING FOR ATTESTATION
# ============================================================================

# Enable audit logging for Container Analysis
resource "google_project_iam_audit_config" "container_analysis_audit" {
  project = var.project_id
  service = "containeranalysis.googleapis.com"

  audit_log_config {
    log_type = "ADMIN_WRITE"
  }

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# Enable audit logging for Binary Authorization
resource "google_project_iam_audit_config" "binary_auth_audit" {
  project = var.project_id
  service = "binaryauthorization.googleapis.com"

  audit_log_config {
    log_type = "ADMIN_WRITE"
  }

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# ============================================================================
# ATTESTATION WORKFLOW (Cloud Build Integration)
# ============================================================================

# The following steps should be added to Cloud Build pipeline (cloudbuild.yaml):
#
# STEP 1: Build and scan image
# - name: 'gcr.io/cloud-builders/docker'
#   id: 'build'
#   args:
#     - 'build'
#     - '-t'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:${SHORT_SHA}'
#     - '-f'
#     - 'docker/Dockerfile'
#     - '.'
#
# STEP 2: Push to staging (untagged)
# - name: 'gcr.io/cloud-builders/docker'
#   id: 'push-staging'
#   args:
#     - 'push'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:scan-${SHORT_SHA}'
#
# STEP 3: Scan image with Trivy
# - name: 'gcr.io/cloud-builders/gke-deploy'
#   id: 'scan'
#   args:
#     - 'run'
#     - '--'
#     - 'trivy'
#     - 'image'
#     - '--severity'
#     - 'CRITICAL,HIGH'
#     - '--exit-code'
#     - '1'  # Fail build on high severity
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:scan-${SHORT_SHA}'
#
# STEP 4: Tag production image
# - name: 'gcr.io/cloud-builders/docker'
#   id: 'tag-prod'
#   args:
#     - 'tag'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:scan-${SHORT_SHA}'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:${SHORT_SHA}'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:latest'
#
# STEP 5: Push to production
# - name: 'gcr.io/cloud-builders/docker'
#   id: 'push-prod'
#   args:
#     - 'push'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:${SHORT_SHA}'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:latest'
#
# STEP 6: Create attestation signature
# - name: 'gcr.io/cloud-builders/gke-deploy'
#   id: 'create-attestation'
#   args:
#     - 'run'
#     - '--'
#     - 'gcloud'
#     - 'beta'
#     - 'container'
#     - 'binauthz'
#     - 'attestations'
#     - 'sign-and-create'
#     - '--project'
#     - '${PROJECT_ID}'
#     - '--artifact-url'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:${SHORT_SHA}'
#     - '--attestation-project'
#     - '${PROJECT_ID}'
#     - '--attestation-authority-note'
#     - 'ollama-build-attestation'
#     - '--attestation-authority-note-project'
#     - '${PROJECT_ID}'
#     - '--signing-key-secret-name'
#     - 'ollama-ci-signing-key-${_ENV}'
#     - '--signing-key-secret-project'
#     - '${PROJECT_ID}'
#
# STEP 7: Verify attestation
# - name: 'gcr.io/cloud-builders/gke-deploy'
#   id: 'verify-attestation'
#   args:
#     - 'run'
#     - '--'
#     - 'gcloud'
#     - 'container'
#     - 'binauthz'
#     - 'attestations'
#     - 'describe'
#     - '--artifact-url'
#     - '${_IMAGE_REPO}/${_IMAGE_NAME}:${SHORT_SHA}'
#     - '--attestation-project'
#     - '${PROJECT_ID}'

# ============================================================================
# ATTESTATION CHAIN OF CUSTODY
# ============================================================================

# The complete attestation chain includes:
#
# 1. Source Code Commitment
#    - Git commit hash (immutable)
#    - Commit message and author
#    - GPG signature on commit
#
# 2. Build Execution
#    - Cloud Build build ID
#    - Build log (Cloud Logging)
#    - Build time and duration
#    - Build environment variables
#
# 3. Image Creation
#    - Image digest (immutable, content-based)
#    - Build steps and commands
#    - Base image used
#    - Dockerfile content
#
# 4. Vulnerability Scanning
#    - Trivy scan results (SARIF format)
#    - Vulnerability count by severity
#    - Scan timestamp
#    - Scanner version
#
# 5. Image Signing
#    - Attestation created and signed
#    - Signing key and certificate
#    - Timestamp of signature
#    - Signer identity (service account)
#
# 6. Policy Enforcement
#    - Binary Authorization policy applied
#    - Attestation verified by GKE
#    - Deployment allowed/denied
#    - Audit log entry created
#
# All steps logged in Cloud Audit Logs (7-year retention)

# ============================================================================
# DEPLOYMENT VERIFICATION POLICY
# ============================================================================

# Only images with valid attestations deploy to production:
#
# ✓ Image must be from approved Artifact Registry
# ✓ Image must have passed vulnerability scan
# ✓ Image must have valid cryptographic signature
# ✓ Signature must be from approved Cloud Build service account
# ✓ Attestation timestamp must be recent (< 7 days)
# ✗ Unsigned images are blocked
# ✗ Unscanned images are blocked
# ✗ Images with critical vulnerabilities are blocked

# ============================================================================
# MONITORING & ALERTING
# ============================================================================

# Alert: Attestation creation failed
resource "google_monitoring_alert_policy" "attestation_creation_failed" {
  project      = var.project_id
  display_name = "Image Attestation Creation Failed"
  combiner     = "OR"

  conditions {
    display_name = "Attestation signing failed in Cloud Build"

    condition_threshold {
      filter            = "resource.type=\"build\" AND metric.type=\"build/attestation_failures\""
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

# Alert: Unsigned image deployment attempt
resource "google_monitoring_alert_policy" "unsigned_image_deployment" {
  project      = var.project_id
  display_name = "Unsigned Image Deployment Attempted"
  combiner     = "OR"

  conditions {
    display_name = "Deployment attempted without valid attestation"

    condition_threshold {
      filter            = "resource.type=\"k8s_cluster\" AND metric.type=\"deployment/unsigned_image_attempts\""
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

# ============================================================================
# OUTPUTS
# ============================================================================

output "build_attestor_name" {
  value       = google_binary_authorization_attestor.build_attestor.name
  description = "Build attestor resource name"
}

output "build_attestation_note" {
  value       = google_container_analysis_note.build_attestation.name
  description = "Container Analysis note for build attestations"
}

output "ci_signing_key_secret" {
  value       = google_secret_manager_secret.ci_signing_key.id
  description = "Secret Manager ID for CI/CD signing key"
  sensitive   = true
}

output "ci_signing_cert_secret" {
  value       = google_secret_manager_secret.ci_signing_cert.id
  description = "Secret Manager ID for CI/CD signing certificate"
}

output "build_attestor_service_account" {
  value       = google_service_account.build_attestor.email
  description = "Service account for Cloud Build attestation creation"
}

output "attestation_signing_key_id" {
  value       = google_kms_crypto_key.attestation_signing.id
  description = "KMS key ID for attestation signing"
  sensitive   = true
}
