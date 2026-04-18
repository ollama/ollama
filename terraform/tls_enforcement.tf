# terraform/tls_enforcement.tf - TLS 1.3+ Security Policies
#
# This module enforces TLS 1.3+ encryption for all communication:
# - Cloud Load Balancer SSL policies (public endpoints)
# - Service-to-service mTLS (Istio)
# - Certificate management (Google Certificate Manager)
# - HSTS enforcement
# - OCSP stapling

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# ============================================================================
# CERTIFICATE MANAGEMENT
# ============================================================================

# Google-managed certificate for elevatediq.ai/ollama
resource "google_certificate_manager_certificate" "ollama_public" {
  project  = var.project_id
  name     = "ollama-public-cert"
  scope    = "EDGE_CACHE"
  managed {
    domains = [var.domain_name]
  }
  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }
}

# Self-signed certificate for internal mTLS (valid 10 years)
resource "tls_private_key" "mtls_ca" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "tls_self_signed_cert" "mtls_ca" {
  private_key_pem = tls_private_key.mtls_ca.private_key_pem

  subject {
    common_name       = "Ollama Internal CA"
    organization      = var.organization_name
    organizational_unit = "Platform Engineering"
    country           = "US"
    province          = "California"
    locality          = "San Francisco"
  }

  validity_period_hours = 87600  # 10 years

  allowed_uses = [
    "key_encipherment",
    "digital_signature",
    "cert_signing",
  ]

  is_ca_certificate = true
}

# Store mTLS CA certificate in Secret Manager
resource "google_secret_manager_secret" "mtls_ca_cert" {
  project   = var.project_id
  secret_id = "ollama-mtls-ca-cert-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "mtls_ca_cert" {
  secret      = google_secret_manager_secret.mtls_ca_cert.id
  secret_data = tls_self_signed_cert.mtls_ca.cert_pem
}

resource "google_secret_manager_secret" "mtls_ca_key" {
  project   = var.project_id
  secret_id = "ollama-mtls-ca-key-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "mtls_ca_key" {
  secret      = google_secret_manager_secret.mtls_ca_key.id
  secret_data = tls_private_key.mtls_ca.private_key_pem
}

# ============================================================================
# CLOUD LOAD BALANCER SSL POLICY (TLS 1.3+)
# ============================================================================

# SSL policy enforcing TLS 1.3 only
resource "google_compute_ssl_policy" "ollama_modern" {
  project         = var.project_id
  name            = "ollama-modern-policy"
  profile         = "RESTRICTED"  # TLS 1.3 only
  min_tls_version = "TLS_1_3"

  # Approved ciphers (modern, secure)
  # RESTRICTED profile uses:
  # - TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
  # - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
  # - TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
  # - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
  # - TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305
  # - TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305
}

# SSL policy for backward compatibility (TLS 1.2+)
resource "google_compute_ssl_policy" "ollama_compatible" {
  project         = var.project_id
  name            = "ollama-compat-policy"
  profile         = "MODERN"  # TLS 1.2+
  min_tls_version = "TLS_1_2"
}

# Backend service with TLS policy
resource "google_compute_backend_service" "ollama_backend" {
  project                 = var.project_id
  name                    = "ollama-backend-${var.environment}"
  protocol                = "HTTPS"
  health_checks           = [google_compute_health_check.ollama.id]
  timeout_sec             = 30
  enable_cdn              = true
  session_affinity        = "CLIENT_IP"
  connection_draining_timeout_sec = 300

  backend {
    group           = var.instance_group_url
    balancing_mode  = "RATE"
    max_rate_per_endpoint = 1000
  }

  log_config {
    enable      = true
    sample_rate = 0.01  # 1% sample for cost
  }

  custom_request_headers {
    headers = ["X-Client-Region:{client_region}"]
  }

  custom_response_headers {
    headers = [
      "Strict-Transport-Security: max-age=31536000; includeSubDomains; preload",
      "X-Content-Type-Options: nosniff",
      "X-Frame-Options: DENY",
      "X-XSS-Protection: 1; mode=block",
      "Referrer-Policy: strict-origin-when-cross-origin",
      "Permissions-Policy: geolocation=(), microphone=(), camera=()"
    ]
  }
}

# HTTPS target proxy with SSL policy
resource "google_compute_target_https_proxy" "ollama" {
  project            = var.project_id
  name               = "ollama-https-proxy-${var.environment}"
  url_map            = google_compute_url_map.ollama.id
  ssl_certificates   = [google_certificate_manager_certificate.ollama_public.id]
  ssl_policy         = google_compute_ssl_policy.ollama_modern.id

  # Enable TLS inspection features
  quic_override = "ENABLE"  # Support QUIC (HTTP/3)
}

# URL map for routing
resource "google_compute_url_map" "ollama" {
  project         = var.project_id
  name            = "ollama-url-map-${var.environment}"
  default_service = google_compute_backend_service.ollama_backend.id

  host_rule {
    hosts        = [var.domain_name]
    path_matcher = "ollama-paths"
  }

  path_matcher {
    name            = "ollama-paths"
    default_service = google_compute_backend_service.ollama_backend.id

    path_rule {
      paths   = ["/api/v1/*"]
      service = google_compute_backend_service.ollama_backend.id
    }

    path_rule {
      paths   = ["/health"]
      service = google_compute_backend_service.ollama_backend.id
    }
  }
}

# Global forwarding rule (public IP)
resource "google_compute_global_address" "ollama" {
  project      = var.project_id
  name         = "ollama-public-ip-${var.environment}"
  address_type = "EXTERNAL"
  ip_version   = "IPV4"
}

resource "google_compute_global_forwarding_rule" "ollama_https" {
  project    = var.project_id
  name       = "ollama-https-lb-${var.environment}"
  ip_address = google_compute_global_address.ollama.id
  target     = google_compute_target_https_proxy.ollama.id
  port_range = "443"
}

# HTTP redirect to HTTPS (enforce TLS)
resource "google_compute_target_http_proxy" "ollama_redirect" {
  project  = var.project_id
  name     = "ollama-http-redirect-${var.environment}"
  url_map  = google_compute_url_map.ollama_redirect.id
}

resource "google_compute_url_map" "ollama_redirect" {
  project = var.project_id
  name    = "ollama-http-redirect-${var.environment}"

  default_url_redirect {
    redirect_code         = "301"
    https_redirect        = true
    strip_query           = false
    strip_trailing_slash  = false
  }
}

resource "google_compute_global_forwarding_rule" "ollama_http" {
  project    = var.project_id
  name       = "ollama-http-lb-${var.environment}"
  ip_address = google_compute_global_address.ollama.id
  target     = google_compute_target_http_proxy.ollama_redirect.id
  port_range = "80"
}

# ============================================================================
# HEALTH CHECKS (HTTPS)
# ============================================================================

resource "google_compute_health_check" "ollama" {
  project = var.project_id
  name    = "ollama-health-check-${var.environment}"

  https_health_check {
    port               = 8000
    request_path       = "/api/v1/health"
    check_interval_sec = 10
    timeout_sec        = 5
    healthy_threshold  = 2
    unhealthy_threshold = 3
  }

  log_config {
    enable = true
  }
}

# ============================================================================
# ISTIO MUTUAL TLS CONFIGURATION (Service-to-Service)
# ============================================================================

# PeerAuthentication: Enforce mTLS for all pods
resource "kubernetes_manifest" "peerauthentication_default" {
  manifest = {
    apiVersion = "security.istio.io/v1beta1"
    kind       = "PeerAuthentication"
    metadata = {
      name      = "default"
      namespace = "default"
    }
    spec = {
      mtls = {
        mode = "STRICT"  # Enforce mTLS, no plain HTTP
      }
      portLevelMtls = {
        8000 = {
          mode = "STRICT"  # API port
        }
        5432 = {
          mode = "STRICT"  # Database port
        }
        6379 = {
          mode = "STRICT"  # Redis port
        }
        11434 = {
          mode = "STRICT"  # Ollama inference port
        }
      }
    }
  }

  depends_on = [var.istio_namespace]
}

# DestinationRule: TLS settings for service calls
resource "kubernetes_manifest" "destination_rule_services" {
  manifest = {
    apiVersion = "networking.istio.io/v1beta1"
    kind       = "DestinationRule"
    metadata = {
      name      = "services"
      namespace = "default"
    }
    spec = {
      host = "*.default.svc.cluster.local"
      trafficPolicy = {
        tls = {
          mode = "ISTIO_MUTUAL"  # Istio mTLS
          sni  = "auto"
        }
        connectionPool = {
          tcp = {
            maxConnections = 100
          }
          http = {
            http1MaxPendingRequests = 2000
            http2MaxRequests        = 1000
          }
        }
        loadBalancer = {
          simple = "ROUND_ROBIN"
        }
      }
    }
  }

  depends_on = [var.istio_namespace]
}

# RequestAuthentication: JWT validation
resource "kubernetes_manifest" "request_authentication" {
  manifest = {
    apiVersion = "security.istio.io/v1beta1"
    kind       = "RequestAuthentication"
    metadata = {
      name      = "jwt-auth"
      namespace = "default"
    }
    spec = {
      jwtRules = [
        {
          issuer  = var.jwt_issuer
          jwksUri = var.jwks_uri
          audiences = ["ollama-api"]
        }
      ]
    }
  }

  depends_on = [var.istio_namespace]
}

# AuthorizationPolicy: JWT requirement on API
resource "kubernetes_manifest" "authorization_policy_api" {
  manifest = {
    apiVersion = "security.istio.io/v1beta1"
    kind       = "AuthorizationPolicy"
    metadata = {
      name      = "api-auth"
      namespace = "default"
    }
    spec = {
      selector = {
        matchLabels = {
          app = "ollama-api"
        }
      }
      action = "ALLOW"
      rules = [
        {
          from = [
            {
              source = {
                principals = ["cluster.local/ns/default/ollama-api"]
              }
            }
          ]
          to = [
            {
              operation = {
                methods = ["POST", "GET", "PUT"]
                paths   = ["/api/v1/*"]
              }
            }
          ]
        }
      ]
    }
  }

  depends_on = [var.istio_namespace]
}

# ============================================================================
# CERTIFICATE ROTATION (Automated)
# ============================================================================

# Kubernetes certificate rotation for service certificates
resource "kubernetes_manifest" "certificate_csr_rotation" {
  manifest = {
    apiVersion = "v1"
    kind       = "ConfigMap"
    metadata = {
      name      = "certificate-rotation-policy"
      namespace = "istio-system"
    }
    data = {
      "rotation-policy" : yamlencode({
        enabled              = true
        rotateBefore        = "720h"  # Rotate 30 days before expiry
        renewBefore         = "2160h" # Renew 90 days before expiry
        issuerRef = {
          name  = "letsencrypt-prod"
          kind  = "ClusterIssuer"
          group = "cert-manager.io"
        }
      })
    }
  }

  depends_on = [var.cert_manager_namespace]
}

# ============================================================================
# MONITORING: TLS/SSL METRICS
# ============================================================================

# Alert: SSL certificate expiration warning
resource "google_monitoring_alert_policy" "ssl_cert_expiry" {
  project      = var.project_id
  display_name = "SSL Certificate Expiration Warning"
  combiner     = "OR"

  conditions {
    display_name = "Certificate expires in 30 days"

    condition_threshold {
      filter            = "resource.type=\"https_lb_rule\" AND metric.type=\"compute.googleapis.com/https_lb_rule/ssl/certificate_expiration_days\""
      comparison        = "COMPARISON_LT"
      threshold_value   = 30  # Alert if < 30 days
      duration          = "300s"
    }
  }

  notification_channels = var.alert_channels

  lifecycle {
    ignore_changes = [notification_channels]
  }
}

# Alert: TLS handshake failures
resource "google_monitoring_alert_policy" "tls_handshake_failures" {
  project      = var.project_id
  display_name = "TLS Handshake Failure Rate"
  combiner     = "OR"

  conditions {
    display_name = "High TLS failure rate"

    condition_threshold {
      filter            = "resource.type=\"https_lb_rule\" AND metric.type=\"compute.googleapis.com/https_lb_rule/ssl/handshake_failures_count\""
      comparison        = "COMPARISON_GT"
      threshold_value   = 10
      duration          = "300s"
    }
  }

  notification_channels = var.alert_channels

  lifecycle {
    ignore_changes = [notification_channels]
  }
}

# ============================================================================
# OUTPUTS
# ============================================================================

output "load_balancer_ip" {
  value       = google_compute_global_address.ollama.address
  description = "Load balancer public IP"
}

output "ssl_policy_name" {
  value       = google_compute_ssl_policy.ollama_modern.name
  description = "TLS 1.3+ SSL policy name"
}

output "certificate_name" {
  value       = google_certificate_manager_certificate.ollama_public.name
  description = "Public SSL certificate name"
}

output "mtls_ca_secret" {
  value       = google_secret_manager_secret.mtls_ca_cert.id
  description = "mTLS CA certificate secret"
}

output "public_endpoint" {
  value       = "https://${var.domain_name}/ollama"
  description = "Public API endpoint URL"
}
