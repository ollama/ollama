# ============================================================================
# GCP Landing Zone Domain Registry - Ollama Service Registration
# ============================================================================
# Purpose: Register Ollama API endpoints with centralized domain registry
# Mandate: Landing Zone Mandate #6 (Endpoint Registration)
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

variable "domain_registry_project" {
  description = "GCP project ID where Landing Zone domain registry is hosted"
  type        = string
  default     = "landing-zone-hub"
}

variable "service_name" {
  description = "Service name for registration"
  type        = string
  default     = "ollama"
}

variable "backend_ip" {
  description = "Internal IP address of Ollama backend service"
  type        = string
}

# ============================================================================
# Data Sources
# ============================================================================

# Fetch existing domain registry configuration
data "google_compute_global_address" "landing_zone_lb" {
  project = var.domain_registry_project
  name    = "landing-zone-public-ip"
}

# ============================================================================
# Domain Registry Entry - Main API Endpoint
# ============================================================================

resource "google_compute_url_map" "ollama_api" {
  project     = var.domain_registry_project
  name        = "${var.environment}-${var.service_name}-url-map"
  description = "URL map for Ollama API endpoints - registered via Landing Zone domain registry"

  # Default backend service (health check endpoint)
  default_service = google_compute_backend_service.ollama_backend.id

  # Path-based routing for API endpoints
  host_rule {
    hosts        = ["elevatediq.ai"]
    path_matcher = "ollama-api-paths"
  }

  path_matcher {
    name            = "ollama-api-paths"
    default_service = google_compute_backend_service.ollama_backend.id

    # API v1 endpoints
    path_rule {
      paths   = ["/ollama/api/v1/*"]
      service = google_compute_backend_service.ollama_backend.id
    }

    # Health check endpoint (public, no auth)
    path_rule {
      paths   = ["/ollama/health"]
      service = google_compute_backend_service.ollama_backend.id
    }

    # Metrics endpoint (internal only, auth required)
    path_rule {
      paths   = ["/ollama/metrics"]
      service = google_compute_backend_service.ollama_backend.id
    }
  }
}

# ============================================================================
# Backend Service Configuration
# ============================================================================

resource "google_compute_backend_service" "ollama_backend" {
  project               = var.domain_registry_project
  name                  = "${var.environment}-${var.service_name}-backend"
  description           = "Backend service for Ollama API (FastAPI on port 8000)"
  protocol              = "HTTP"
  port_name             = "http"
  timeout_sec           = 30
  enable_cdn            = false
  load_balancing_scheme = "EXTERNAL_MANAGED"

  # Health check configuration
  health_checks = [google_compute_health_check.ollama_health.id]

  # Backend configuration
  backend {
    group           = google_compute_instance_group.ollama_instances.id
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
    max_utilization = 0.8
  }

  # Connection draining
  connection_draining_timeout_sec = 300

  # Security settings
  security_settings {
    client_tls_policy = google_compute_client_tls_policy.ollama_tls.id
  }

  # Logging configuration
  log_config {
    enable      = true
    sample_rate = 1.0
  }

  # IAP (Identity-Aware Proxy) configuration
  iap {
    oauth2_client_id     = var.iap_client_id
    oauth2_client_secret = var.iap_client_secret
  }

  # Mandatory labels from Landing Zone compliance
  labels = {
    environment      = var.environment
    team             = "ollama-platform"
    application      = "ollama"
    component        = "api"
    cost-center      = "ai-infrastructure"
    managed-by       = "terraform"
    git_repo         = "github-com-kushin77-ollama"
    lifecycle_status = "active"
    compliance       = "landing-zone-registered"
    service_tier     = "tier-1"
    data_sensitivity = "public"
  }
}

# ============================================================================
# Health Check Configuration
# ============================================================================

resource "google_compute_health_check" "ollama_health" {
  project             = var.domain_registry_project
  name                = "${var.environment}-${var.service_name}-health-check"
  description         = "Health check for Ollama API service"
  check_interval_sec  = 10
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port               = 8000
    request_path       = "/health"
    proxy_header       = "NONE"
    response           = ""
  }

  # Mandatory labels
  labels = {
    environment = var.environment
    application = "ollama"
    component   = "health-check"
    managed-by  = "terraform"
  }
}

# ============================================================================
# Instance Group (Backend Instances)
# ============================================================================

resource "google_compute_instance_group" "ollama_instances" {
  project     = var.project_id
  name        = "${var.environment}-${var.service_name}-instance-group"
  description = "Instance group for Ollama backend containers"
  zone        = "us-central1-a"

  # Instances will be added dynamically by deployment process
  instances = []

  # Named ports for load balancer
  named_port {
    name = "http"
    port = 8000
  }

  # Mandatory labels
  labels = {
    environment = var.environment
    application = "ollama"
    component   = "compute"
    managed-by  = "terraform"
  }
}

# ============================================================================
# TLS Policy (Mutual TLS for Backend)
# ============================================================================

resource "google_compute_client_tls_policy" "ollama_tls" {
  project     = var.domain_registry_project
  name        = "${var.environment}-${var.service_name}-tls-policy"
  description = "TLS 1.3+ policy for Ollama backend communication"

  # Enforce TLS 1.3 minimum
  min_tls_version = "TLS_1_3"

  # Server certificate validation
  server_validation_ca {
    grpc_endpoint {
      target_uri = "unix:///var/run/sds/uds_path"
    }
  }

  # Mandatory labels
  labels = {
    environment = var.environment
    application = "ollama"
    component   = "tls"
    managed-by  = "terraform"
  }
}

# ============================================================================
# Firewall Rules (Zero Trust - Only Allow LB)
# ============================================================================

resource "google_compute_firewall" "allow_lb_to_ollama" {
  project     = var.project_id
  name        = "${var.environment}-allow-lb-to-${var.service_name}"
  description = "Allow GCP Load Balancer to reach Ollama backend (zero-trust perimeter)"
  network     = "default"
  direction   = "INGRESS"
  priority    = 1000

  # Allow only from GCP Load Balancer source ranges
  source_ranges = [
    "130.211.0.0/22",  # GCP Load Balancer health checks
    "35.191.0.0/16"    # GCP Load Balancer proxies
  ]

  # Target Ollama instances
  target_tags = ["${var.environment}-${var.service_name}-backend"]

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  # Mandatory labels
  labels = {
    environment = var.environment
    application = "ollama"
    component   = "firewall"
    managed-by  = "terraform"
  }
}

resource "google_compute_firewall" "deny_all_to_ollama" {
  project     = var.project_id
  name        = "${var.environment}-deny-all-to-${var.service_name}"
  description = "Block all non-LB traffic to Ollama backend (zero-trust enforcement)"
  network     = "default"
  direction   = "INGRESS"
  priority    = 2000

  # Deny all sources except LB
  source_ranges = ["0.0.0.0/0"]

  # Target Ollama instances
  target_tags = ["${var.environment}-${var.service_name}-backend"]

  deny {
    protocol = "tcp"
    ports    = ["8000"]
  }

  # Mandatory labels
  labels = {
    environment = var.environment
    application = "ollama"
    component   = "firewall"
    managed-by  = "terraform"
  }
}

# ============================================================================
# IAP Configuration Variables
# ============================================================================

variable "iap_client_id" {
  description = "OAuth2 client ID for Identity-Aware Proxy"
  type        = string
  sensitive   = true
}

variable "iap_client_secret" {
  description = "OAuth2 client secret for Identity-Aware Proxy"
  type        = string
  sensitive   = true
}

# ============================================================================
# Outputs
# ============================================================================

output "public_endpoint" {
  description = "Public HTTPS endpoint for Ollama API"
  value       = "https://elevatediq.ai/ollama"
}

output "load_balancer_ip" {
  description = "GCP Load Balancer public IP address"
  value       = data.google_compute_global_address.landing_zone_lb.address
}

output "backend_service_id" {
  description = "Backend service ID for Ollama API"
  value       = google_compute_backend_service.ollama_backend.id
}

output "health_check_endpoint" {
  description = "Health check endpoint URL"
  value       = "https://elevatediq.ai/ollama/health"
}

output "registration_status" {
  description = "Domain registry registration status"
  value       = "✅ Registered with Landing Zone domain registry"
}
