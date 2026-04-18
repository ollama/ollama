# ==============================================================================
# Agentic Infrastructure Variables
# ==============================================================================
# Landing Zone 24-Label Mandate Compliance
# ==============================================================================

# =============================================================================
# GCP Project Configuration
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_project" {
  description = "GCP Project identifier for namespace"
  type        = string
}

variable "region" {
  description = "GCP Region (e.g., us-central1, us-east1)"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (production, staging, development, sandbox)"
  type        = string
  validation {
    condition     = contains(["production", "prod", "staging", "development", "dev", "sandbox"], var.environment)
    error_message = "Environment must be: production, staging, development, or sandbox"
  }
}

variable "application" {
  description = "Application name (ollama)"
  type        = string
  default     = "ollama"
}

variable "component" {
  description = "Component name (agents, orchestrator, etc.)"
  type        = string
  default     = "agents"
}

# =============================================================================
# Cloud Run Configuration
# =============================================================================

variable "agent_image_uri" {
  description = "Artifact Registry URI for agent service image"
  type        = string
}

variable "orchestrator_image_uri" {
  description = "Artifact Registry URI for orchestrator service image"
  type        = string
}

variable "cpu_limit" {
  description = "CPU limit per agent service instance (e.g., '2')"
  type        = string
  default     = "2"
}

variable "memory_limit" {
  description = "Memory limit per agent service instance (e.g., '2Gi')"
  type        = string
  default     = "2Gi"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "orchestrator_min_instances" {
  description = "Minimum orchestrator instances"
  type        = number
  default     = 0
}

variable "orchestrator_max_instances" {
  description = "Maximum orchestrator instances"
  type        = number
  default     = 5
}

variable "log_level" {
  description = "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
  type        = string
  default     = "INFO"
}

variable "ollama_service_url" {
  description = "Internal Ollama service URL"
  type        = string
}

# =============================================================================
# Database Configuration
# =============================================================================

variable "firestore_location" {
  description = "Firestore location (us-central1, eu-west1, etc.)"
  type        = string
  default     = "us-central1"
}

variable "bq_location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

# =============================================================================
# Encryption Configuration (CMEK)
# =============================================================================

variable "artifact_kms_key" {
  description = "Cloud KMS key for Artifact Registry encryption (full resource name)"
  type        = string
}

variable "pubsub_kms_key" {
  description = "Cloud KMS key for Pub/Sub encryption"
  type        = string
}

variable "bq_kms_key" {
  description = "Cloud KMS key for BigQuery encryption"
  type        = string
}

# =============================================================================
# Monitoring Configuration
# =============================================================================

variable "slack_channel" {
  description = "Slack channel name for alerts (e.g., #prod-ollama-alerts)"
  type        = string
  default     = ""
}

# =============================================================================
# LANDING ZONE 24-LABEL MANDATORY SCHEMA
# =============================================================================
# ✅ All 24 labels REQUIRED per GCP Landing Zone governance
# These labels enable compliance tracking, cost attribution, and lifecycle management

variable "resource_labels" {
  description = "24 Mandatory labels required by GCP Landing Zone policy"
  type        = map(string)

  validation {
    condition = alltrue([
      for label in [
        "environment",
        "cost_center",
        "team",
        "managed_by",
        "created_by",
        "created_date",
        "lifecycle_state",
        "teardown_date",
        "retention_days",
        "product",
        "component",
        "tier",
        "compliance",
        "version",
        "stack",
        "backup_strategy",
        "monitoring_enabled",
        "budget_owner",
        "project_code",
        "monthly_budget_usd",
        "chargeback_unit",
        "git_repository",
        "git_branch",
        "auto_delete"
      ] : contains(keys(var.resource_labels), label)
    ])
    error_message = "❌ VIOLATION: All 24 mandatory labels must be present in resource_labels."
  }
}

# =============================================================================
# Label Descriptions (Reference)
# =============================================================================
#
# Category 1: Organizational (4 labels)
#   - environment: production|staging|development|sandbox
#   - cost_center: Finance code (e.g., "AI-ENG-001")
#   - team: Team name (e.g., "ai-infrastructure")
#   - managed_by: terraform (mandatory)
#
# Category 2: Lifecycle & Retention (5 labels)
#   - created_by: Email address
#   - created_date: YYYY-MM-DD
#   - lifecycle_state: active|maintenance|sunset
#   - teardown_date: YYYY-MM-DD or "none"
#   - retention_days: 0-3650
#
# Category 3: Business & Risk (4 labels)
#   - product: ollama
#   - component: agents|orchestrator|api|database|etc
#   - tier: critical|high|medium|low
#   - compliance: pci-dss|hipaa|sox|fedramp|none
#
# Category 4: Technical (4 labels)
#   - version: Application version (e.g., "0.1.0")
#   - stack: Tech stack (e.g., "python-3.11-fastapi-gcp")
#   - backup_strategy: continuous|hourly|daily|weekly|none
#   - monitoring_enabled: true|false
#
# Category 5: Financial (4 labels)
#   - budget_owner: Email or team name
#   - project_code: Finance code (e.g., "OLLAMA-2026-001")
#   - monthly_budget_usd: Expected monthly cost
#   - chargeback_unit: Department or cost center
#
# Category 6: Git Attribution & Mapping (3 labels)
#   - git_repository: github.com/kushin77/ollama
#   - git_branch: main|develop|feature-*
#   - auto_delete: true|false
#
# =============================================================================
