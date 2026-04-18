# Cloud Logging Configuration for Centralized Security Event Collection
# 7-year retention for compliance, structured logging, audit trail

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Locals for configuration
locals {
  logging_bucket_name = "ollama-security-logs-${var.environment}"
  log_retention_days  = 2555  # 7 years

  # Security event log filters
  security_events = [
    "resource.type=\"k8s_cluster\"",
    "resource.type=\"gke_cluster\"",
    "severity=\"CRITICAL\" OR severity=\"ERROR\" OR severity=\"WARNING\""
  ]

  # Audit trail events
  audit_events = [
    "protoPayload.methodName=~\".*\"",
    "protoPayload.status.code != 0"  # Only failed operations
  ]
}

# Cloud Logging Bucket (destination for logs)
resource "google_logging_project_bucket_config" "security_logs" {
  project      = var.project_id
  location     = var.logging_region
  bucket_id    = replace(local.logging_bucket_name, "-", "_")
  retention_days = local.log_retention_days

  # CMEK encryption for logs at rest
  cmek_config {
    kms_key_name = var.logging_cmek_key
  }

  # Enable analytics on logs
  enable_analytics = true

  labels = {
    environment   = var.environment
    team          = "security"
    application   = "ollama"
    component     = "logging"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git_repo      = "github.com/kushin77/ollama"
    lifecycle_status = "active"
  }

  depends_on = [
    google_project_iam_member.logging_cmek_key_user
  ]
}

# Log Router Sink 1: Security Events
resource "google_logging_project_sink" "security_events_sink" {
  name   = "ollama-security-events-sink"
  destination = "logging.googleapis.com/projects/${var.project_id}/locations/${var.logging_region}/buckets/${google_logging_project_bucket_config.security_logs.bucket_id}"

  # Filter for security-relevant events
  filter = <<-EOT
    resource.type="k8s_cluster"
    AND (
      protoPayload.methodName=~".*Mutate.*"
      OR protoPayload.methodName=~".*Create.*"
      OR protoPayload.methodName=~".*Delete.*"
      OR protoPayload.methodName=~".*Update.*"
      OR severity="ERROR"
      OR severity="CRITICAL"
    )
  EOT

  # Include any existing log entries
  include_children = true

  depends_on = [
    google_logging_project_bucket_config.security_logs
  ]
}

# Log Router Sink 2: Binary Authorization Events
resource "google_logging_project_sink" "binary_auth_events_sink" {
  name   = "ollama-binary-auth-sink"
  destination = "logging.googleapis.com/projects/${var.project_id}/locations/${var.logging_region}/buckets/${google_logging_project_bucket_config.security_logs.bucket_id}"

  # Binary Authorization policy decisions
  filter = <<-EOT
    resource.type="k8s_cluster"
    AND protoPayload.methodName=~"io.k8s.core.v1.pods.*"
    AND protoPayload.request.metadata.annotations."binaryauthorization.grafeas.io/attestation"
  EOT

  include_children = true
}

# Log Router Sink 3: Container Analysis (Vulnerability Scans)
resource "google_logging_project_sink" "container_analysis_sink" {
  name   = "ollama-container-analysis-sink"
  destination = "logging.googleapis.com/projects/${var.project_id}/locations/${var.logging_region}/buckets/${google_logging_project_bucket_config.security_logs.bucket_id}"

  # Container Analysis API calls
  filter = <<-EOT
    protoPayload.serviceName="containeranalysis.googleapis.com"
  EOT

  include_children = true
}

# Log Router Sink 4: IAM Changes (Access Control)
resource "google_logging_project_sink" "iam_changes_sink" {
  name   = "ollama-iam-changes-sink"
  destination = "logging.googleapis.com/projects/${var.project_id}/locations/${var.logging_region}/buckets/${google_logging_project_bucket_config.security_logs.bucket_id}"

  # IAM policy changes
  filter = <<-EOT
    protoPayload.methodName=~"SetIamPolicy"
    OR protoPayload.methodName=~"AddBinding"
  EOT

  include_children = true
}

# Log Router Sink 5: Network Changes
resource "google_logging_project_sink" "network_changes_sink" {
  name   = "ollama-network-changes-sink"
  destination = "logging.googleapis.com/projects/${var.project_id}/locations/${var.logging_region}/buckets/${google_logging_project_bucket_config.security_logs.bucket_id}"

  # VPC, firewall, and network policy changes
  filter = <<-EOT
    resource.type="gce_firewall_rule"
    OR resource.type="gce_network"
    OR protoPayload.methodName=~".*Firewall.*"
  EOT

  include_children = true
}

# Service Account for Log Analysis
resource "google_service_account" "log_analyzer" {
  account_id   = "ollama-log-analyzer-${var.environment}"
  display_name = "Ollama Log Analyzer (${var.environment})"
  description  = "Service account for analyzing security logs and generating insights"

  labels = {
    environment   = var.environment
    team          = "security"
    application   = "ollama"
  }
}

# IAM: Log Analyzer can read logs
resource "google_project_iam_member" "log_analyzer_viewer" {
  project = var.project_id
  role    = "roles/logging.viewer"
  member  = "serviceAccount:${google_service_account.log_analyzer.email}"
}

# IAM: Log Analyzer can create queries
resource "google_project_iam_member" "log_analyzer_private_logs_viewer" {
  project = var.project_id
  role    = "roles/logging.privateLogViewer"
  member  = "serviceAccount:${google_service_account.log_analyzer.email}"
}

# Cloud Logging Query 1: Failed Authentication Attempts
resource "google_logging_project_bucket_config" "failed_auth_query" {
  project      = var.project_id
  location     = var.logging_region
  bucket_id    = "failed_auth_query"
  retention_days = local.log_retention_days

  labels = {
    query_type = "authentication"
  }

  depends_on = [
    google_logging_project_bucket_config.security_logs
  ]
}

# Save query as a saved search (for dashboard use)
resource "google_logging_log_view" "failed_auth_view" {
  project       = var.project_id
  bucket        = google_logging_project_bucket_config.security_logs.bucket_id
  location      = var.logging_region
  name          = "ollama-failed-auth-attempts"
  display_name  = "Failed Authentication Attempts"

  filter = <<-EOT
    severity="ERROR"
    AND (
      protoPayload.status.message=~".*authentication.*"
      OR protoPayload.status.message=~".*unauthorized.*"
      OR protoPayload.status.code=401
      OR protoPayload.status.code=403
    )
  EOT

  depends_on = [
    google_logging_project_bucket_config.security_logs
  ]
}

# CMEK Key User IAM binding for logging service
resource "google_project_iam_member" "logging_cmek_key_user" {
  project = var.project_id
  role    = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member  = "serviceAccount:logging@${var.project_id}.iam.gserviceaccount.com"
}

# Output for logging configuration
output "security_logs_bucket" {
  value       = google_logging_project_bucket_config.security_logs.name
  description = "Cloud Logging bucket name for security events"
}

output "log_analyzer_service_account" {
  value       = google_service_account.log_analyzer.email
  description = "Service account email for log analysis"
}

output "security_logs_sink_names" {
  value = {
    security_events = google_logging_project_sink.security_events_sink.name
    binary_auth     = google_logging_project_sink.binary_auth_events_sink.name
    container_analysis = google_logging_project_sink.container_analysis_sink.name
    iam_changes     = google_logging_project_sink.iam_changes_sink.name
    network_changes = google_logging_project_sink.network_changes_sink.name
  }
  description = "Names of log router sinks configured"
}
