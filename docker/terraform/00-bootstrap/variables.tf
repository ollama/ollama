# ==============================================================================
# Variables: Bootstrap Phase
# ==============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

# ==============================================================================
# 24-LABEL MANDATORY SCHEMA
# ==============================================================================

variable "project_labels" {
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
      ] : contains(keys(var.project_labels), label)
    ])
    error_message = "❌ VIOLATION: All 24 mandatory labels must be present in project_labels."
  }
}
