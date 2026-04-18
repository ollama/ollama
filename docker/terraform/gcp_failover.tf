# GCP Automated Failover for Ollama API
# Purpose: Configure multi-region active-passive failover using a global HTTP(S) Load Balancer
# - Primary backend: active (capacity 100%)
# - Secondary backend: passive (failover=true)
# Health checks: HTTP to /api/v1/health on port 8000 (internal service)
# Labels/Naming follow GCP Landing Zone conventions

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.20.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.primary_region
}

# Note: Terraform block and provider are partially duplicated in failover module.
# For module consolidation, move provider config to main.tf if not already present.
# ----------------------------
# Variables
# ----------------------------
# Note: Terraform block and provider are partially duplicated in failover module.
# For module consolidation, move provider config to main.tf if not already present.
# ----------------------------
# Variables (duplicates removed; see variables.tf for centralized definitions)
# ----------------------------
# project_id, environment, application, component already defined in variables.tf

variable "primary_region" {
  description = "Primary region (e.g., us-central1)"
  type        = string
}

variable "secondary_region" {
  description = "Secondary region (e.g., us-east1)"
  type        = string
}

variable "enable_failover" {
  description = "Enable creation of failover backend bindings"
  type        = bool
  default     = false
}

variable "primary_instance_group" {
  description = "Self-link of the primary managed instance group (MIG) serving the API"
  type        = string
}

variable "secondary_instance_group" {
  description = "Self-link of the secondary managed instance group (MIG) serving the API"
  type        = string
}

variable "health_check_path" {
  description = "HTTP health check path"
  type        = string
  default     = "/api/v1/health"
}

variable "labels" {
  description = "Mandatory Landing Zone labels"
  type        = map(string)
  default = {
    environment      = "production"
    team             = "platform"
    application      = "ollama"
    component        = "api"
    cost-center      = "0000"
    managed-by       = "terraform"
    git_repo         = "github.com/kushin77/ollama"
    lifecycle_status = "active"
  }
}

# ----------------------------
# Health Check (Global)
# ----------------------------
resource "google_compute_health_check" "ollama_api" {
  count       = var.enable_failover ? 1 : 0
  name        = "${var.environment}-ollama-network-hc"
  description = "HTTP health check for Ollama API"

  http_health_check {
    port         = 8000
    request_path = var.health_check_path
  }

  timeout_sec        = 5
  check_interval_sec = 10
  healthy_threshold  = 2
  unhealthy_threshold = 3
}

# ----------------------------
# Backend Service with Failover
# Note: The backend block 'failover = true' marks a backend as backup.
# ----------------------------
resource "google_compute_backend_service" "ollama_api_backend" {
  count       = var.enable_failover ? 1 : 0
  name        = "${var.environment}-ollama-network-backend"
  description = "Global backend service for Ollama API with active-passive failover"
  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 30
  connection_draining_timeout_sec = 10

  # Enable logging (useful for observability)
  log_config {
    enable = true
  }

  health_checks = [google_compute_health_check.ollama_api[0].self_link]

  backend {
    group           = var.primary_instance_group
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
    max_utilization = 0.8
    # Primary (active)
    failover        = false
  }

  backend {
    group           = var.secondary_instance_group
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
    max_utilization = 0.8
    # Secondary (passive)
    failover        = true
  }

  # Optional: outlier detection and circuit breakers (provider >= v5)
  dynamic "outlier_detection" {
    for_each = var.enable_failover ? [1] : []
    content {
      max_ejection_percent    = 50
      consecutive_5xx_errors  = 3
      interval                = "10s"
      base_ejection_time      = "30s"
    }
  }

  dynamic "circuit_breakers" {
    for_each = var.enable_failover ? [1] : []
    content {
      max_connections       = 1024
      max_pending_requests  = 1024
      max_requests          = 2048
      max_retries           = 3
    }
  }
}

# ----------------------------
# URL Map / HTTPS Proxy / Forwarding Rule
# IMPORTANT: If an existing LB is already configured, attach this backend
# service to the existing URL map instead of creating a new one.
# ----------------------------
resource "google_compute_url_map" "ollama_url_map" {
  count       = var.enable_failover ? 1 : 0
  name        = "${var.environment}-ollama-network-url-map"
  default_service = google_compute_backend_service.ollama_api_backend[0].self_link
}

resource "google_compute_target_https_proxy" "ollama_https_proxy" {
  count = var.enable_failover ? 1 : 0
  name  = "${var.environment}-ollama-network-https-proxy"
  url_map = google_compute_url_map.ollama_url_map[0].self_link
  # SSL certs assumed pre-provisioned via Certificate Manager; attach here
  ssl_certificates = []
}

resource "google_compute_global_forwarding_rule" "ollama_https_rule" {
  count                 = var.enable_failover ? 1 : 0
  name                  = "${var.environment}-ollama-network-https-fr"
  load_balancing_scheme = "EXTERNAL"
  port_range            = "443"
  target                = google_compute_target_https_proxy.ollama_https_proxy[0].self_link
  ip_protocol           = "TCP"
}
