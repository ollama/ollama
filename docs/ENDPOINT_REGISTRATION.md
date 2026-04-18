# Ollama Endpoint Registration - Landing Zone Integration

**Status**: REQUIRED - Mandate #6 Compliance
**Date**: January 19, 2026
**Reference**: [Endpoint Onboarding Integration Guide](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/onboarding/ENDPOINT_ONBOARDING_INTEGRATION.md)

---

## Overview

Per GCP Landing Zone **Mandate #6** (added Jan 18, 2026), all public-facing services MUST be registered in the centralized domain registry. This document outlines ollama's endpoint registration requirements.

---

## Required Endpoint Configuration

### Primary Endpoint

```yaml
endpoint: https://elevatediq.ai/ollama
domain: elevatediq.ai
subdomain: ollama
backend_service: ollama-api-backend
health_check_path: /health
```

### Domain Registry Entry (Terraform)

**Location**: `gcp-landing-zone/terraform/modules/networking/lb-domain-registry/variables.tf`

```hcl
"ollama" = {
  domain             = "elevatediq.ai"
  subdomains         = ["ollama"]
  tls_enabled        = true
  oauth_protected    = false  # Public API (API key authenticated)
  cloud_armor_policy = "global-armor"
  backend_service    = "ollama-api-backend"
  health_check_path  = "/health"
  timeout_sec        = 30
  enable_cdn         = true

  # API routing rules
  path_rules = {
    "inference" = {
      paths   = ["/api/v1/generate", "/api/v1/chat", "/api/v1/embeddings"]
      service = "ollama-inference-backend"
    }
    "models" = {
      paths   = ["/api/v1/models/*"]
      service = "ollama-models-backend"
    }
    "conversations" = {
      paths   = ["/api/v1/conversations/*"]
      service = "ollama-conversations-backend"
    }
  }
}
```

---

## Security Requirements (FedRAMP Compliant)

### SC-7: Boundary Protection
- ✅ All backends private (internal only)
- ✅ GCP Load Balancer is ONLY external entry point
- ✅ Firewall blocks direct access to ports 8000, 5432, 6379, 11434

### SC-8: Transmission Confidentiality
- ✅ TLS 1.3 enforced for all external traffic
- ✅ Mutual TLS for internal service communication
- ✅ SSL certificate auto-renewal via Google Managed SSL

### AC-3: Access Enforcement
- ✅ API key authentication required (all endpoints except `/health`)
- ✅ Rate limiting: 100 requests/min per API key
- ✅ CORS restricted to `https://elevatediq.ai` origins only

### AU-2: Audit Events
- ✅ Cloud Logging captures all API requests
- ✅ 7-year retention policy (compliance requirement)
- ✅ Request ID tracking for all endpoints
- ✅ Structured logging with user_id, model, latency_ms

---

## Cloud Armor Configuration

```hcl
resource "google_compute_security_policy" "ollama_armor" {
  name        = "ollama-ddos-protection"
  description = "DDoS protection for ollama API endpoints"

  rule {
    action   = "rate_based_ban"
    priority = "100"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"

      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }

      ban_duration_sec = 600
    }
  }

  rule {
    action   = "deny(403)"
    priority = "200"
    match {
      expr {
        expression = "origin.region_code == 'CN' || origin.region_code == 'KP'"
      }
    }
  }

  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
  }
}
```

---

## Health Check Configuration

```hcl
resource "google_compute_health_check" "ollama_health" {
  name                = "ollama-health-check"
  check_interval_sec  = 10
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 8000
    request_path = "/health"
  }

  log_config {
    enable = true
  }
}
```

---

## Backend Service Configuration

```hcl
resource "google_compute_backend_service" "ollama_api" {
  name                  = "ollama-api-backend"
  protocol              = "HTTP"
  port_name             = "http"
  timeout_sec           = 30
  enable_cdn            = true
  health_checks         = [google_compute_health_check.ollama_health.id]
  security_policy       = google_compute_security_policy.ollama_armor.id

  backend {
    group           = google_compute_instance_group_manager.ollama.instance_group
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }

  iap {
    oauth2_client_id     = var.oauth_client_id
    oauth2_client_secret = var.oauth_client_secret
  }
}
```

---

## Monitoring & Alerting

### Required Metrics

```yaml
metrics:
  - name: ollama_request_count
    type: counter
    labels: [endpoint, status_code, model]

  - name: ollama_request_latency
    type: histogram
    labels: [endpoint, model]
    buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

  - name: ollama_health_check_status
    type: gauge
    labels: [backend, region]
```

### Required Alerts

1. **High Error Rate**
   - Condition: 5xx errors > 1% for 5 minutes
   - Severity: Critical
   - Notification: PagerDuty + Slack

2. **Health Check Failure**
   - Condition: Health check fails for 3 consecutive checks
   - Severity: Critical
   - Notification: PagerDuty

3. **High Latency**
   - Condition: P99 latency > 10s for 5 minutes
   - Severity: Warning
   - Notification: Slack

4. **Rate Limit Exceeded**
   - Condition: 429 responses > 100/min
   - Severity: Warning
   - Notification: Slack

---

## Deployment Checklist

- [ ] Add ollama to domain registry in `terraform/modules/networking/lb-domain-registry/variables.tf`
- [ ] Configure SSL certificate for `ollama.elevatediq.ai`
- [ ] Set up Cloud Armor security policy
- [ ] Configure health check endpoint at `/health`
- [ ] Enable Cloud Logging with 7-year retention
- [ ] Set up monitoring dashboards in Cloud Monitoring
- [ ] Configure alerting rules
- [ ] Test endpoint from external network
- [ ] Verify firewall blocks direct access to internal services
- [ ] Document emergency access procedures
- [ ] Train team on incident response
- [ ] Update runbooks with endpoint details

---

## Testing Procedures

### Pre-Deployment Testing

```bash
# 1. Verify health endpoint
curl https://ollama.elevatediq.ai/health
# Expected: {"status": "healthy"}

# 2. Test API authentication
curl -H "Authorization: Bearer invalid-key" \
     https://ollama.elevatediq.ai/api/v1/models
# Expected: 401 Unauthorized

# 3. Test valid API request
curl -H "Authorization: Bearer <api-key>" \
     https://ollama.elevatediq.ai/api/v1/models
# Expected: 200 OK with models list

# 4. Test rate limiting
for i in {1..150}; do
  curl -H "Authorization: Bearer <api-key>" \
       https://ollama.elevatediq.ai/health
done
# Expected: 429 Too Many Requests after 100 requests

# 5. Verify CORS enforcement
curl -H "Origin: https://malicious-site.com" \
     https://ollama.elevatediq.ai/api/v1/models
# Expected: CORS error (blocked by browser)
```

### Post-Deployment Verification

```bash
# 1. Check backend health
gcloud compute backend-services get-health ollama-api-backend --global

# 2. Review logs
gcloud logging read "resource.type=http_load_balancer AND
                     resource.labels.backend_service_name=ollama-api-backend" \
                    --limit=50 --format=json

# 3. Check Cloud Armor metrics
gcloud compute security-policies describe global-armor --global

# 4. Verify SSL certificate
curl -vI https://ollama.elevatediq.ai 2>&1 | grep -i "SSL certificate"
```

---

## Incident Response

### Endpoint Unavailable

1. Check backend health: `gcloud compute backend-services get-health ollama-api-backend --global`
2. Review logs: `gcloud logging read "severity>=ERROR"`
3. Check Cloud Armor rules for false positives
4. Verify SSL certificate validity
5. Check DNS resolution: `nslookup ollama.elevatediq.ai`

### High Latency

1. Check inference service metrics
2. Review database connection pool
3. Check Redis cache hit rate
4. Verify model cache usage
5. Scale backend instances if needed

### Rate Limit Issues

1. Review Cloud Armor rate limit rules
2. Check for legitimate traffic spikes
3. Adjust rate limits if needed
4. Block malicious IPs if detected

---

## References

- [Endpoint & Domain Management Strategy](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/governance/ENDPOINT_DOMAIN_MANAGEMENT_STRATEGY.md)
- [Endpoint Onboarding Integration Guide](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/onboarding/ENDPOINT_ONBOARDING_INTEGRATION.md)
- [Endpoint Management Operational Guide](https://github.com/kushin77/gcp-landing-zone/blob/main/docs/operations/ENDPOINT_MANAGEMENT_OPERATIONAL_GUIDE.md)
- [Cloud Armor Documentation](https://cloud.google.com/armor/docs)
- [Load Balancing Best Practices](https://cloud.google.com/load-balancing/docs/best-practices)

---

**Owner**: AI Infrastructure Team
**Last Updated**: January 19, 2026
**Next Review**: April 19, 2026 (Quarterly)
**Status**: PENDING IMPLEMENTATION
