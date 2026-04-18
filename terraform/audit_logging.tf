# ============================================================================
# GCP Cloud Logging & 7-Year Retention Infrastructure
# ============================================================================
# Purpose: Enable 7-year audit logging for Landing Zone compliance
# Mandate: Landing Zone Mandate #7 (Audit Logging)
# Created: 2026-01-19
# Owner: Infrastructure Team
# ============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ============================================================================
# Variables
# ============================================================================

variable "project_id" {
  description = "GCP project ID for Ollama service"
  type        = string
  default     = "prod-ollama-api"
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "audit_retention_days" {
  description = "Audit log retention in days (7 years = 2557 days)"
  type        = number
  default     = 2557  # 7 years as per Landing Zone mandate
}

variable "log_bucket_location" {
  description = "GCS bucket location for audit logs"
  type        = string
  default     = "US"
}

variable "service_account_email" {
  description = "Service account for Cloud Logging writes"
  type        = string
}

# ============================================================================
# GCS Bucket for 7-Year Log Retention
# ============================================================================

resource "google_storage_bucket" "audit_logs" {
  project       = var.project_id
  name          = "${var.project_id}-audit-logs-${var.environment}"
  location      = var.log_bucket_location
  storage_class = "COLDLINE"  # Cost-optimized for long-term retention

  # Force destroy disabled - protect audit logs from accidental deletion
  force_destroy = false

  # Uniform bucket-level access (required for security)
  uniform_bucket_level_access = true

  # Versioning enabled for audit trail
  versioning {
    enabled = true
  }

  # Lifecycle rule: Delete after 7 years + 30 days grace period
  lifecycle_rule {
    condition {
      age = var.audit_retention_days + 30
    }
    action {
      type = "Delete"
    }
  }

  # Lifecycle rule: Move to ARCHIVE after 6 years (cost optimization)
  lifecycle_rule {
    condition {
      age = 2192  # 6 years
    }
    action {
      type          = "SetStorageClass"
      storage_class = "ARCHIVE"
    }
  }

  # Encryption with CMEK (Customer-Managed Encryption Key)
  encryption {
    default_kms_key_name = google_kms_crypto_key.audit_logs_key.id
  }

  # Mandatory labels from Landing Zone compliance
  labels = {
    environment      = var.environment
    team             = "ollama-platform"
    application      = "ollama"
    component        = "audit-logging"
    cost-center      = "ai-infrastructure"
    managed-by       = "terraform"
    git_repo         = "github-com-kushin77-ollama"
    lifecycle_status = "active"
    compliance       = "landing-zone-audit"
    retention_years  = "7"
    data_sensitivity = "audit"
  }
}

# ============================================================================
# Cloud Logging Sink (Export to GCS)
# ============================================================================

resource "google_logging_project_sink" "audit_logs_sink" {
  project     = var.project_id
  name        = "${var.environment}-ollama-audit-sink"
  description = "Export Ollama API audit logs to GCS for 7-year retention"

  # Destination: GCS bucket
  destination = "storage.googleapis.com/${google_storage_bucket.audit_logs.name}"

  # Filter: Capture all Ollama API logs
  filter = <<-EOT
    resource.type="cloud_run_revision" OR
    resource.type="k8s_container" OR
    resource.type="gce_instance"
    AND
    (
      logName="projects/${var.project_id}/logs/ollama-api-audit" OR
      labels.service="ollama-api"
    )
  EOT

  # Include children resources
  include_children = true

  # Unique writer identity for IAM binding
  unique_writer_identity = true
}

# ============================================================================
# IAM Permissions for Cloud Logging to Write to GCS
# ============================================================================

resource "google_storage_bucket_iam_member" "audit_logs_writer" {
  bucket = google_storage_bucket.audit_logs.name
  role   = "roles/storage.objectCreator"
  member = google_logging_project_sink.audit_logs_sink.writer_identity
}

# ============================================================================
# Cloud KMS for CMEK Encryption
# ============================================================================

# KMS Keyring
resource "google_kms_key_ring" "audit_logs_keyring" {
  project  = var.project_id
  name     = "${var.environment}-audit-logs-keyring"
  location = var.log_bucket_location
}

# KMS Crypto Key
resource "google_kms_crypto_key" "audit_logs_key" {
  name            = "${var.environment}-audit-logs-key"
  key_ring        = google_kms_key_ring.audit_logs_keyring.id
  rotation_period = "7776000s"  # 90 days

  # Prevent deletion of encrypted data
  lifecycle {
    prevent_destroy = true
  }

  # Version template
  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = "SOFTWARE"
  }

  # Mandatory labels
  labels = {
    environment = var.environment
    application = "ollama"
    component   = "encryption"
    managed-by  = "terraform"
  }
}

# Grant Cloud Storage service account access to KMS key
resource "google_kms_crypto_key_iam_member" "audit_logs_key_user" {
  crypto_key_id = google_kms_crypto_key.audit_logs_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.project.number}@gs-project-accounts.iam.gserviceaccount.com"
}

# ============================================================================
# Data Sources
# ============================================================================

data "google_project" "project" {
  project_id = var.project_id
}

# ============================================================================
# Service Account for Application Logging
# ============================================================================

resource "google_service_account" "cloud_logging_writer" {
  project      = var.project_id
  account_id   = "${var.environment}-ollama-logging-sa"
  display_name = "Cloud Logging Writer for Ollama API"
  description  = "Service account for writing audit logs to Cloud Logging"
}

# Grant logging.logWriter role
resource "google_project_iam_member" "logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloud_logging_writer.email}"
}

# Grant logging.viewer role (for debugging)
resource "google_project_iam_member" "logging_viewer" {
  project = var.project_id
  role    = "roles/logging.viewer"
  member  = "serviceAccount:${google_service_account.cloud_logging_writer.email}"
}

# ============================================================================
# Monitoring Alert (Log Export Failures)
# ============================================================================

resource "google_monitoring_alert_policy" "log_export_failure" {
  project      = var.project_id
  display_name = "${var.environment} - Ollama Audit Log Export Failure"
  combiner     = "OR"

  conditions {
    display_name = "Log sink export failures detected"

    condition_threshold {
      filter          = "resource.type=\"logging_sink\" AND resource.labels.name=\"${google_logging_project_sink.audit_logs_sink.name}\" AND metric.type=\"logging.googleapis.com/exports/error_count\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [var.notification_channel_id]

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content = <<-EOT
      Audit log export to GCS is failing. This violates Landing Zone 7-year retention mandate.

      Immediate Actions:
      1. Check Cloud Logging sink status
      2. Verify GCS bucket permissions
      3. Verify KMS key access
      4. Escalate to infrastructure team if unresolved within 15 minutes

      Escalation: L2 → L3 → CTO
    EOT
  }
}

# ============================================================================
# Variables for Monitoring
# ============================================================================

variable "notification_channel_id" {
  description = "Notification channel ID for alerts"
  type        = string
}

# ============================================================================
# Outputs
# ============================================================================

output "audit_logs_bucket" {
  description = "GCS bucket for 7-year audit log retention"
  value       = google_storage_bucket.audit_logs.name
}

output "audit_logs_sink" {
  description = "Cloud Logging sink for audit logs"
  value       = google_logging_project_sink.audit_logs_sink.name
}

output "service_account_email" {
  description = "Service account email for Cloud Logging writes"
  value       = google_service_account.cloud_logging_writer.email
}

output "kms_key_id" {
  description = "KMS key ID for CMEK encryption"
  value       = google_kms_crypto_key.audit_logs_key.id
}

output "retention_days" {
  description = "Audit log retention period in days"
  value       = var.audit_retention_days
}

output "compliance_status" {
  description = "Landing Zone audit logging compliance status"
  value       = "✅ 7-year retention configured"
}
