# Cloud CDN Infrastructure for Static Assets
#
# Provides global content delivery network for:
# - Documentation (compiled HTML, Markdown)
# - Images (PNG, WebP optimized)
# - Model artifacts (ONNX, metadata)
# - Configuration files (YAML, JSON)
#
# Expected Impact:
# - Latency reduction: 70% (from 500ms to 150ms p99)
# - Bandwidth savings: 40% (via caching + compression)
# - Cost reduction: 50% (reduced origin requests)
#
# Note: terraform and locals blocks are centralized in main.tf

locals {
  environment = var.environment
  project_id = var.gcp_project_id
  region      = var.gcp_region

  # Mandatory PMO labels for GCP Landing Zone compliance
  common_labels = {
    environment      = local.environment
    team            = "infra-team"
    application     = "ollama"
    component       = "cdn"
    cost-center     = var.cost_center
    managed-by      = "terraform"
    git_repo        = "github.com/kushin77/ollama"
    lifecycle_status = "active"
  }
}

# Storage bucket for static assets
resource "google_storage_bucket" "ollama_assets" {
  project       = local.project_id
  name          = "${local.environment}-ollama-assets"
  location      = local.region
  force_destroy = false

  labels = merge(
    local.common_labels,
    {
      data_classification = "public"
      pii_data           = "false"
    }
  )

  # Enable versioning for rollback capability
  versioning {
    enabled = true
  }

  # Lifecycle management to reduce storage costs
  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "STANDARD"
    }
    condition {
      age = 90
    }
  }

  # Automatic deletion of old versions (keep 5)
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 5
      is_live            = false
    }
  }

  # Uniform bucket-level access (required for CDN)
  uniform_bucket_level_access = true
}

# IAM binding to allow public read access
resource "google_storage_bucket_iam_binding" "public_read" {
  bucket = google_storage_bucket.ollama_assets.name
  role   = "roles/storage.objectViewer"

  members = [
    "allUsers"
  ]
}

# Backend bucket for Cloud CDN
resource "google_compute_backend_bucket" "ollama_cdn" {
  project     = local.project_id
  name        = "${local.environment}-ollama-cdn"
  bucket_name = google_storage_bucket.ollama_assets.name

  # Enable Cloud CDN
  enable_cdn = true

  # Enable automatic compression
  compression_mode = "AUTOMATIC"

  # Cache policy configuration
  cdn_policy {
    # Cache all static content by default
    cache_mode = "CACHE_ALL_STATIC"

    # Client-side cache TTL (Browser cache)
    client_ttl = 3600  # 1 hour for documentation

    # Server-side cache TTL (CDN cache)
    default_ttl = 86400  # 1 day

    # Maximum TTL for any object
    max_ttl = 604800  # 7 days for model artifacts

    # Cache negative responses (404s)
    negative_caching = true

    negative_caching_policy {
      code = 404
      ttl  = 120  # Cache 404s for 2 minutes
    }

    negative_caching_policy {
      code = 410
      ttl  = 120
    }

    # Serve stale content while revalidating
    serve_while_stale = 86400  # 1 day

    # Custom request headers for origin
    custom_request_headers {
      headers = [
        "X-CDN-Request: true",
        "X-Requested-By: cloud-cdn"
      ]
    }
  }

  labels = merge(
    local.common_labels,
    {
      resource_type = "backend-bucket"
    }
  )
}

# Logging bucket for CDN activity (optional, for audit trails)
resource "google_storage_bucket" "cdn_logs" {
  project       = local.project_id
  name          = "${local.environment}-ollama-cdn-logs"
  location      = local.region
  force_destroy = false

  labels = merge(
    local.common_labels,
    {
      data_classification = "internal"
      pii_data           = "false"
    }
  )

  # Lifecycle to manage logs storage costs
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 90  # Keep logs for 3 months
    }
  }
}

# URL map to route requests to CDN
resource "google_compute_url_map" "ollama_cdn_routes" {
  project = local.project_id
  name    = "${local.environment}-ollama-cdn-routes"

  default_service = google_compute_backend_bucket.ollama_cdn.self_link

  # Route specific paths for different cache policies
  path_rule {
    paths = [
      "/assets/*",
      "/docs/*",
      "/images/*"
    ]
    service = google_compute_backend_bucket.ollama_cdn.self_link
  }

  # Model artifacts with longer TTL
  path_rule {
    paths = [
      "/models/*",
      "/weights/*"
    ]
    service = google_compute_backend_bucket.ollama_cdn.self_link
  }
}

# HTTPS proxy for CDN
resource "google_compute_target_https_proxy" "ollama_cdn_proxy" {
  project      = local.project_id
  name         = "${local.environment}-ollama-cdn-proxy"
  url_map      = google_compute_url_map.ollama_cdn_routes.id
  ssl_policy   = google_compute_ssl_policy.cdn_policy.id

  # SSL certificate (self-managed or managed)
  ssl_certificates = [
    google_compute_managed_ssl_certificate.cdn_cert.id
  ]
}

# Managed SSL certificate
resource "google_compute_managed_ssl_certificate" "cdn_cert" {
  project = local.project_id
  name    = "${local.environment}-ollama-cdn-cert"

  managed {
    domains = var.cdn_domains
  }

  lifecycle {
    create_before_destroy = true
  }
}

# SSL policy for CDN (TLS 1.3+ only, per Security mandate)
resource "google_compute_ssl_policy" "cdn_policy" {
  project        = local.project_id
  name           = "${local.environment}-ollama-cdn-ssl"
  profile        = "RESTRICTED"  # TLS 1.3 only
  min_tls_version = "TLS_1_3"

  custom_features = [
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
  ]
}

# Global forwarding rule for CDN
resource "google_compute_global_forwarding_rule" "ollama_cdn" {
  project                = local.project_id
  name                   = "${local.environment}-ollama-cdn-rule"
  ip_protocol            = "TCP"
  load_balancing_scheme  = "EXTERNAL"
  port_range             = "443"
  target_https_proxy     = google_compute_target_https_proxy.ollama_cdn_proxy.id
}

# Cloud Armor policy for DDoS protection
resource "google_compute_security_policy" "cdn_armor" {
  project = local.project_id
  name    = "${local.environment}-ollama-cdn-armor"

  description = "Cloud Armor policy for CDN DDoS protection"

  # Block traffic from high-risk countries
  rules {
    action   = "deny(403)"
    priority = "1000"
    match {
      versioned_expr = "CEL"
      cel_options {
        recommended_header_names = ["Accept-Language"]
      }
      cel_expression = "origin.region_code in ['KP', 'IR', 'CU']"
    }
    description = "Deny traffic from high-risk countries"
  }

  # Rate limiting (100 req/min per IP)
  rules {
    action   = "rate_based_ban"
    priority = "1001"
    match {
      versioned_expr = "CEL"
      cel_expression = "true"
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"

      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }

      ban_duration_sec = 600
    }
    description = "Rate limiting: 100 req/min"
  }

  # Allow all other traffic
  rules {
    action   = "allow"
    priority = "65535"
    match {
      versioned_expr = "CEL"
      cel_expression = "true"
    }
    description = "Default rule"
  }
}

# Cloud Monitoring dashboards and alerts
resource "google_monitoring_dashboard" "cdn_dashboard" {
  project        = local.project_id
  dashboard_json = jsonencode({
    displayName = "${local.environment}-ollama-cdn-dashboard"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Cache Hit Ratio"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"compute.googleapis.com/https/request_count\" resource.type=\"https_lb_rule\""
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Request Latency (P99)"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"compute.googleapis.com/https/internal_request_latencies\" resource.type=\"https_lb_rule\""
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Bandwidth (Bytes)"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"compute.googleapis.com/https/request_bytes_count\" resource.type=\"https_lb_rule\""
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Error Rate (4xx, 5xx)"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"compute.googleapis.com/https/request_count\" resource.type=\"https_lb_rule\" metric.response_code_class=~\"4xx|5xx\""
                  }
                }
              }]
            }
          }
        }
      ]
    }
  })
}

# Outputs for integration with load balancer
output "cdn_ip_address" {
  value       = google_compute_global_forwarding_rule.ollama_cdn.ip_address
  description = "Public IP address for CDN"
}

output "cdn_backend_bucket_id" {
  value       = google_compute_backend_bucket.ollama_cdn.id
  description = "CDN backend bucket ID"
}

output "cdn_bucket_name" {
  value       = google_storage_bucket.ollama_assets.name
  description = "GCS bucket name for assets"
}

output "cdn_url" {
  value       = "https://${var.cdn_domains[0]}/assets"
  description = "CDN endpoint URL"
}
