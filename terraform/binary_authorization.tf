# terraform/binary_authorization.tf - Google Binary Authorization
#
# Binary Authorization ensures only verified, approved container images
# are deployed to GKE clusters. Prevents unauthorized or vulnerable images.
#
# Components:
# 1. Artifact Registry: Centralized container image storage (CMEK encrypted)
# 2. Attestation: Cryptographic proof of image approval
# 3. Policy: Enforcement rules (block unapproved images)
# 4. Verification: Service accounts and IAM for attestation checking

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
# ARTIFACT REGISTRY (Centralized Container Image Storage)
# ============================================================================

# Docker repository in Artifact Registry
resource "google_artifact_registry_repository" "ollama_docker" {
  project       = var.project_id
  location      = var.artifact_registry_region
  repository_id = "ollama-docker"
  description   = "Ollama container images with CMEK encryption"
  format        = "DOCKER"
  mode          = "STANDARD_REPOSITORY"

  docker_config {
    immutable_tags = true  # Tags cannot be overwritten (prevent tampering)
  }

  # CMEK Encryption
  kms_key_name = var.storage_cmek_key_id

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "artifact-registry"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }
}

# Helm charts repository
resource "google_artifact_registry_repository" "ollama_helm" {
  project       = var.project_id
  location      = var.artifact_registry_region
  repository_id = "ollama-helm"
  description   = "Ollama Helm charts with CMEK encryption"
  format        = "HELM"

  kms_key_name = var.storage_cmek_key_id

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "helm-charts"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }
}

# Service account for pushing images to Artifact Registry
resource "google_service_account" "artifact_registry_pusher" {
  project      = var.project_id
  account_id   = "ollama-artifact-pusher-sa"
  display_name = "Ollama Artifact Registry Pusher"
  description  = "Service account for Cloud Build to push images to Artifact Registry"
}

# Grant Cloud Build service account access to push images
resource "google_artifact_registry_repository_iam_member" "cloud_build_push" {
  project    = var.project_id
  location   = google_artifact_registry_repository.ollama_docker.location
  repository = google_artifact_registry_repository.ollama_docker.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${var.cloud_build_service_account}"
}

# Grant developers read access
resource "google_artifact_registry_repository_iam_member" "developers_read" {
  project    = var.project_id
  location   = google_artifact_registry_repository.ollama_docker.location
  repository = google_artifact_registry_repository.ollama_docker.name
  role       = "roles/artifactregistry.reader"
  member     = "group:developers@${var.organization_domain}"
}

# ============================================================================
# BINARY AUTHORIZATION POLICY
# ============================================================================

# Binary Authorization Policy
resource "google_binary_authorization_policy" "ollama" {
  project            = var.project_id
  admission_whitelist_patterns = []
  default_admission_rule {
    require_attestations_by = [google_binary_authorization_attestor.ollama.name]
    enforcement_mode        = "ENFORCED_BLOCK_AND_AUDIT_LOG"
  }

  global_policy_evaluation_enabled = true
  enforce_on_latest_images        = true

  # Kubernetes system images exemption
  kubernetes_namespace_admissionrules = {
    "istio-system" = {
      require_attestations_by = [google_binary_authorization_attestor.ollama.name]
      enforcement_mode        = "DRYRUN_AUDIT_LOG_ONLY"  # Warn but don't block system pods
    }
    "kube-system" = {
      require_attestations_by = []
      enforcement_mode        = "ALWAYS_ALLOW"  # Allow system components
    }
    "kube-public" = {
      require_attestations_by = []
      enforcement_mode        = "ALWAYS_ALLOW"
    }
  }
}

# ============================================================================
# ATTESTATION AUTHORITY (Digital Signature Verification)
# ============================================================================

# Attestor: Entity that signs and verifies image attestations
resource "google_binary_authorization_attestor" "ollama" {
  project = var.project_id
  name    = "ollama-attestor"

  user_owned_grafeas_note {
    note_reference = google_container_analysis_note.ollama_attestation.name
  }
}

# Grafeas Note: Records image attestation information
resource "google_container_analysis_note" "ollama_attestation" {
  project = var.project_id
  name    = "ollama-attestation-note"

  attestation_authority {
    hint {
      human_readable_name = "Ollama Image Attestation"
    }
  }
}

# Service account for attestation verification
resource "google_service_account" "attestation_verifier" {
  project      = var.project_id
  account_id   = "ollama-attestation-verifier-sa"
  display_name = "Ollama Attestation Verifier"
  description  = "Service account for verifying image attestations"
}

# Grant GKE cluster access to attestation verifier
resource "google_project_iam_member" "gke_attestation_verifier" {
  project = var.project_id
  role    = "roles/binaryauthorization.attestorsAdmin"
  member  = "serviceAccount:${google_service_account.attestation_verifier.email}"
}

# ============================================================================
# PUBLIC KEY INFRASTRUCTURE FOR ATTESTATION
# ============================================================================

# Generate RSA key pair for signing attestations
resource "tls_private_key" "attestation_signing_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Certificate for public key (for verification)
resource "tls_self_signed_cert" "attestation_cert" {
  private_key_pem = tls_private_key.attestation_signing_key.private_key_pem

  subject {
    common_name       = "Ollama Binary Authorization"
    organization      = var.organization_name
    organizational_unit = "Platform Engineering"
    country           = "US"
    province          = "California"
    locality          = "San Francisco"
  }

  validity_period_hours = 87600  # 10 years
  allowed_uses = [
    "digital_signature",
    "cert_signing",
  ]
}

# Store attestation signing key in Secret Manager
resource "google_secret_manager_secret" "attestation_signing_key" {
  project   = var.project_id
  secret_id = "ollama-attestation-signing-key-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "attestation_signing_key" {
  secret      = google_secret_manager_secret.attestation_signing_key.id
  secret_data = tls_private_key.attestation_signing_key.private_key_pem
}

# Store public certificate in Secret Manager
resource "google_secret_manager_secret" "attestation_public_cert" {
  project   = var.project_id
  secret_id = "ollama-attestation-public-cert-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "attestation_public_cert" {
  secret      = google_secret_manager_secret.attestation_public_cert.id
  secret_data = tls_self_signed_cert.attestation_cert.cert_pem
}

# ============================================================================
# POLICY ENFORCEMENT ON GKE CLUSTER
# ============================================================================

# Enable Binary Authorization on GKE cluster
resource "google_container_cluster" "binary_authorization_enabled" {
  project = var.project_id

  binary_authorization {
    enabled = true
  }

  # This should be applied to existing cluster
  depends_on = [
    google_binary_authorization_policy.ollama,
    google_binary_authorization_attestor.ollama
  ]
}

# ============================================================================
# IAM: SERVICE ACCOUNT PERMISSIONS (Least Privilege)
# ============================================================================

# Cloud Build can create attestations
resource "google_project_iam_member" "cloud_build_attestation_creator" {
  project = var.project_id
  role    = "roles/binaryauthorization.attestorsAdmin"
  member  = "serviceAccount:${var.cloud_build_service_account}"
}

# Cloud Build can read attestation notes
resource "google_project_iam_member" "cloud_build_note_reader" {
  project = var.project_id
  role    = "roles/containeranalysis.notes.editor"
  member  = "serviceAccount:${var.cloud_build_service_account}"
}

# GKE nodes can read attestations
resource "google_project_iam_member" "gke_attestation_reader" {
  project = var.project_id
  role    = "roles/binaryauthorization.attestorsViewer"
  member  = "serviceAccount:${var.gke_node_service_account}"
}

# ============================================================================
# ARTIFACT REGISTRY CLEANUP POLICIES
# ============================================================================

# Cleanup old images (retention: 30 days)
resource "google_artifact_registry_cleanup_policies" "ollama_docker_cleanup" {
  project        = var.project_id
  location       = google_artifact_registry_repository.ollama_docker.location
  repository     = google_artifact_registry_repository.ollama_docker.name

  cleanup_policies {
    id     = "delete-old-images"
    action = "DELETE"

    most_recent_versions {
      keep_count = 10  # Keep 10 most recent versions
    }
  }

  cleanup_policies {
    id     = "delete-untagged-images"
    action = "DELETE"

    condition {
      tag_state             = "UNTAGGED"
      older_than            = "7776000s"  # 90 days
    }
  }
}

# ============================================================================
# MONITORING & ALERTS
# ============================================================================

# Alert: Binary Authorization policy violation
resource "google_monitoring_alert_policy" "binary_auth_violation" {
  project      = var.project_id
  display_name = "Binary Authorization Policy Violation"
  combiner     = "OR"

  conditions {
    display_name = "Unapproved image deployment attempted"

    condition_threshold {
      filter            = "resource.type=\"k8s_cluster\" AND metric.type=\"binaryauthorization.googleapis.com/policy_violations\""
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

# Alert: Attestation verification failures
resource "google_monitoring_alert_policy" "attestation_failures" {
  project      = var.project_id
  display_name = "Attestation Verification Failure"
  combiner     = "OR"

  conditions {
    display_name = "Image attestation verification failed"

    condition_threshold {
      filter            = "resource.type=\"k8s_cluster\" AND metric.type=\"binaryauthorization.googleapis.com/attestation_failures\""
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

output "artifact_registry_repository" {
  value       = google_artifact_registry_repository.ollama_docker.repository_id
  description = "Artifact Registry repository for Docker images"
}

output "artifact_registry_location" {
  value       = google_artifact_registry_repository.ollama_docker.location
  description = "Artifact Registry repository location"
}

output "binary_authorization_policy" {
  value       = google_binary_authorization_policy.ollama.name
  description = "Binary Authorization policy name"
}

output "attestor_name" {
  value       = google_binary_authorization_attestor.ollama.name
  description = "Attestor resource name"
}

output "attestation_note" {
  value       = google_container_analysis_note.ollama_attestation.name
  description = "Grafeas note for attestations"
}

output "attestation_signing_key_secret" {
  value       = google_secret_manager_secret.attestation_signing_key.id
  description = "Secret Manager ID for attestation signing key"
  sensitive   = true
}

output "attestation_public_cert_secret" {
  value       = google_secret_manager_secret.attestation_public_cert.id
  description = "Secret Manager ID for public attestation certificate"
}

output "artifact_registry_docker_image_path" {
  value       = "${var.artifact_registry_region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ollama_docker.repository_id}"
  description = "Full path for pushing Docker images to Artifact Registry"
}
