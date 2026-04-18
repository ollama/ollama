# CDN Infrastructure Variables
# Variables for Cloud CDN, Storage, and related configuration
#
# Note: gcp_project_id, gcp_region, environment, team, cost_center, lifecycle_status
# are defined in variables.tf. This file contains CDN-specific variables only.

variable "cdn_domains" {
  type        = list(string)
  description = "Domains for CDN SSL certificate (e.g., cdn.elevatediq.ai, assets.elevatediq.ai)"
  validation {
    condition     = length(var.cdn_domains) > 0
    error_message = "At least one domain must be specified."
  }
}

variable "cdn_cache_modes" {
  type = object({
    documentation_ttl    = number
    images_ttl           = number
    model_artifacts_ttl  = number
  })
  description = "Cache TTL configuration for different asset types"
  default = {
    documentation_ttl   = 3600      # 1 hour
    images_ttl          = 86400     # 1 day
    model_artifacts_ttl = 604800    # 7 days
  }
}

variable "cdn_compression" {
  type        = bool
  description = "Enable automatic compression for CDN"
  default     = true
}

variable "cdn_rate_limit" {
  type = object({
    requests_per_minute = number
    ban_duration_sec    = number
  })
  description = "Rate limiting configuration"
  default = {
    requests_per_minute = 100
    ban_duration_sec    = 600
  }
}

variable "enable_cdn_logs" {
  type        = bool
  description = "Enable logging for CDN requests"
  default     = true
}

variable "cdn_log_retention_days" {
  type        = number
  description = "Days to retain CDN logs"
  default     = 90
}

variable "allow_countries" {
  type        = list(string)
  description = "ISO country codes to allow (empty = all allowed)"
  default     = []
}

variable "deny_countries" {
  type        = list(string)
  description = "ISO country codes to deny"
  default     = ["KP", "IR", "CU"]  # North Korea, Iran, Cuba
}

variable "cdn_budget_alert_threshold" {
  type        = number
  description = "Monthly budget threshold for CDN costs in USD"
  default     = 500
}

variable "tags" {
  type        = map(string)
  description = "Additional tags to apply to CDN resources"
  default     = {}
}
