# Security Command Center Integration and Threat Detection
# Centralized security posture management and compliance monitoring

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Enable Security Command Center API
resource "google_project_service" "scc_api" {
  project = var.project_id
  service = "securitycenter.googleapis.com"
  disable_on_destroy = false
}

# Enable Asset Inventory API (required for SCC)
resource "google_project_service" "asset_api" {
  project = var.project_id
  service = "cloudasset.googleapis.com"
  disable_on_destroy = false
}

# Security Command Center Service Account
resource "google_service_account" "scc_service" {
  account_id   = "ollama-scc-service-${var.environment}"
  display_name = "Ollama SCC Service Account (${var.environment})"
  description  = "Service account for Security Command Center integration"

  labels = {
    environment   = var.environment
    team          = "security"
    application   = "ollama"
  }

  depends_on = [google_project_service.scc_api]
}

# IAM: SCC can read organization resources
resource "google_project_iam_member" "scc_viewer" {
  project = var.project_id
  role    = "roles/securitycenter.securityCenterViewer"
  member  = "serviceAccount:${google_service_account.scc_service.email}"
}

# IAM: SCC can write findings
resource "google_project_iam_member" "scc_findings_editor" {
  project = var.project_id
  role    = "roles/securitycenter.findingsEditor"
  member  = "serviceAccount:${google_service_account.scc_service.email}"
}

# Custom SCC Finding Category 1: Supply Chain Security
resource "google_scc_custom_module" "supply_chain_findings" {
  display_name = "Ollama Supply Chain Security Findings (${var.environment})"
  description  = "Detects supply chain security violations (unsigned images, vulnerable containers)"
  scope        = "PROJECT"
  enablement_state = "ENABLED"
  severity     = "HIGH"

  custom_config {
    recommendation = "Ensure all container images are signed and vulnerability-free before deployment"
    description    = <<-EOT
      This finding detects:
      1. Unsigned container images attempting deployment
      2. Images with critical or high vulnerabilities
      3. Images not from approved Artifact Registry
      4. Missing Software Bill of Materials (SBOM)
      5. Binary Authorization policy violations
    EOT
  }

  depends_on = [google_project_service.scc_api]
}

# Custom SCC Finding Category 2: Network Security
resource "google_scc_custom_module" "network_security_findings" {
  display_name = "Ollama Network Security Findings (${var.environment})"
  description  = "Detects network security violations and misconfigurations"
  scope        = "PROJECT"
  enablement_state = "ENABLED"
  severity     = "HIGH"

  custom_config {
    recommendation = "Review and remediate network security violations immediately"
    description    = <<-EOT
      This finding detects:
      1. Firewall rules allowing public access to internal services
      2. Network policies with excessive permissions
      3. Unencrypted communication channels
      4. Service exposure to internet
      5. VPC peering with untrusted networks
    EOT
  }

  depends_on = [google_project_service.scc_api]
}

# Custom SCC Finding Category 3: Data Protection
resource "google_scc_custom_module" "data_protection_findings" {
  display_name = "Ollama Data Protection Findings (${var.environment})"
  description  = "Detects encryption and data protection violations"
  scope        = "PROJECT"
  enablement_state = "ENABLED"
  severity     = "CRITICAL"

  custom_config {
    recommendation = "Enable CMEK encryption and TLS for all data"
    description    = <<-EOT
      This finding detects:
      1. Data stores without CMEK encryption
      2. Backups without encryption
      3. Unencrypted inter-service communication
      4. Missing TLS enforcement
      5. Weak encryption algorithms
    EOT
  }

  depends_on = [google_project_service.scc_api]
}

# Custom SCC Finding Category 4: Access Control
resource "google_scc_custom_module" "access_control_findings" {
  display_name = "Ollama Access Control Findings (${var.environment})"
  description  = "Detects IAM and access control violations"
  scope        = "PROJECT"
  enablement_state = "ENABLED"
  severity     = "HIGH"

  custom_config {
    recommendation = "Apply least privilege principles to all service accounts"
    description    = <<-EOT
      This finding detects:
      1. Overly permissive IAM roles
      2. Missing Workload Identity
      3. Service accounts with unnecessary permissions
      4. Outdated service account credentials
      5. Missing audit logging for sensitive operations
    EOT
  }

  depends_on = [google_project_service.scc_api]
}

# Custom SCC Finding Category 5: Compliance
resource "google_scc_custom_module" "compliance_findings" {
  display_name = "Ollama Compliance Findings (${var.environment})"
  description  = "Detects compliance violations and missing controls"
  scope        = "PROJECT"
  enablement_state = "ENABLED"
  severity     = "MEDIUM"

  custom_config {
    recommendation = "Enable required controls for compliance frameworks"
    description    = <<-EOT
      This finding detects:
      1. Missing audit logging
      2. Insufficient log retention
      3. Missing vulnerability scanning
      4. Incomplete inventory tracking
      5. Missing incident response procedures
    EOT
  }

  depends_on = [google_project_service.scc_api]
}

# Threat Detection: Workload Vulnerability Scanner Integration
resource "google_project_service" "workload_vulnerability_scanner" {
  project = var.project_id
  service = "workloads.googleapis.com"
  disable_on_destroy = false
}

# Threat Detection: Binary Authorization Attestor Monitoring
resource "google_monitoring_alert_policy" "attestor_failures" {
  display_name = "Binary Authorization Attestor Failures (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Attestor operation failure"
    condition_threshold = {
      filter          = "resource.type=\"k8s_cluster\" AND protoPayload.methodName=~\".*attestor.*\" AND protoPayload.status.code != 0"
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  documentation = {
    content   = "Binary Authorization attestor operation failed. Check attestor configuration and key accessibility."
    mime_type = "text/markdown"
  }
}

# Threat Detection: Privilege Escalation Monitoring
resource "google_monitoring_alert_policy" "privilege_escalation" {
  display_name = "Potential Privilege Escalation Attempts (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Privilege escalation detected"
    condition_threshold = {
      filter          = "protoPayload.request.metadata.annotations=~\".*privileged.*\" OR protoPayload.request.spec.containers.securityContext.privileged=true"
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  alert_strategy = {
    auto_close = "86400s"  # 24 hours
  }

  documentation = {
    content   = "Privileged container or privilege escalation detected. Immediate security review required."
    mime_type = "text/markdown"
  }
}

# Threat Detection: Anomalous Network Activity
resource "google_monitoring_alert_policy" "anomalous_network_activity" {
  display_name = "Anomalous Network Activity Detected (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Unusual network traffic pattern"
    condition_threshold = {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"compute.googleapis.com/instance/network/sent_bytes_count\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.network_anomaly_threshold  # Bytes per 5 minutes

      aggregations = [{
        alignment_period    = "300s"
        per_series_aligner  = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_STDDEV_POP"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  documentation = {
    content   = "Anomalous network traffic detected. Investigate for data exfiltration or lateral movement."
    mime_type = "text/markdown"
  }
}

# Threat Detection: Failed Deployment Attempts
resource "google_monitoring_alert_policy" "failed_deployments" {
  display_name = "Failed Deployment Attempts (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Multiple failed deployment attempts"
    condition_threshold = {
      filter          = "resource.type=\"k8s_cluster\" AND protoPayload.methodName=~\"create\" AND protoPayload.status.code != 0"
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
        group_by_fields    = ["protoPayload.status.message"]
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  documentation = {
    content   = "Multiple deployment failures detected. Check Binary Authorization policy, image signatures, and vulnerabilities."
    mime_type = "text/markdown"
  }
}

# Threat Detection: Configuration Drift
resource "google_monitoring_alert_policy" "configuration_drift" {
  display_name = "Security Configuration Drift Detected (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Unauthorized configuration change"
    condition_threshold = {
      filter          = "protoPayload.methodName=~\".*Update.*\" AND resource.type=\"k8s_cluster\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  alert_strategy = {
    auto_close = "604800s"  # 7 days
  }

  documentation = {
    content   = "Rapid configuration changes detected. Verify all changes are authorized and documented."
    mime_type = "text/markdown"
  }
}

# Output
output "scc_custom_modules" {
  value = {
    supply_chain       = google_scc_custom_module.supply_chain_findings.id
    network_security   = google_scc_custom_module.network_security_findings.id
    data_protection    = google_scc_custom_module.data_protection_findings.id
    access_control     = google_scc_custom_module.access_control_findings.id
    compliance         = google_scc_custom_module.compliance_findings.id
  }
  description = "SCC custom finding modules for Ollama"
}

output "threat_detection_alerts" {
  value = {
    attestor_failures           = google_monitoring_alert_policy.attestor_failures.id
    privilege_escalation        = google_monitoring_alert_policy.privilege_escalation.id
    anomalous_network_activity  = google_monitoring_alert_policy.anomalous_network_activity.id
    failed_deployments          = google_monitoring_alert_policy.failed_deployments.id
    configuration_drift         = google_monitoring_alert_policy.configuration_drift.id
  }
  description = "Threat detection alert policies"
}
