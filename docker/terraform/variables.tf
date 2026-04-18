# Budget Alerts Configuration Variables

variable "project_id" {
  type        = string
  description = "GCP project ID for cost tracking"
}

variable "billing_account_name" {
  type        = string
  description = "Name of the GCP billing account (display_name from google_billing_account)"
  default     = "My Billing Account"
}

variable "monthly_budget_usd" {
  type        = number
  description = "Monthly budget limit in USD for Ollama project"
  default     = 500

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "Monthly budget must be positive number"
  }
}

variable "budget_alert_email" {
  type        = string
  description = "Email address for warning alerts (50% and 80% thresholds)"

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.budget_alert_email))
    error_message = "Must be a valid email address"
  }
}

variable "budget_alert_email_critical" {
  type        = string
  description = "Email address for critical alerts (100% threshold)"

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.budget_alert_email_critical))
    error_message = "Must be a valid email address"
  }
}

variable "environment" {
  type        = string
  description = "Environment name (production, staging, development)"
  default     = "production"

  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be production, staging, or development"
  }
}

variable "team" {
  type        = string
  description = "Team responsible for cost governance"
  default     = "platform"
}

variable "cost_center" {
  type        = string
  description = "Finance cost center code for mandatory labeling (e.g., FIN-1234)"
}

variable "lifecycle_status" {
  type        = string
  description = "Resource lifecycle status for labeling (active|maintenance|sunset)"
  default     = "active"
  validation {
    condition     = contains(["active", "maintenance", "sunset"], var.lifecycle_status)
    error_message = "lifecycle_status must be one of: active, maintenance, sunset"
  }
}

variable "enable_pubsub_alerts" {
  type        = bool
  description = "Enable Pub/Sub topic for programmatic budget alerts"
  default     = false
}
