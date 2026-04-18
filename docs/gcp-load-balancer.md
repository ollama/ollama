# GCP Load Balancer Configuration for elevatediq.ai/ollama

## Overview

This document outlines how to configure Google Cloud Load Balancer to route traffic from `elevatediq.ai/ollama` to the Ollama inference backend.

---

## Architecture

```
┌─────────────────┐
│ Client Request  │
│ (elevatediq.ai) │
└────────┬────────┘
         │ HTTPS
         ▼
┌─────────────────────────────┐
│  GCP Cloud Load Balancer    │
│  - TLS Termination          │
│  - SSL Certificate          │
│  - Health Checks            │
│  - Routing Rules            │
└────────┬────────────────────┘
         │ HTTP (internal)
         ▼
┌────────────────────────────────────┐
│  Backend Service Group             │
│  - ollama-api:8000                 │
│  - ollama-api:8000 (replica)       │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Ollama Services │
│  (GKE or VMs)    │
└──────────────────┘
```

---

## Prerequisites

- Google Cloud Project with billing enabled
- gcloud CLI installed and authenticated
- Domain ownership for elevatediq.ai (DNS records)
- SSL Certificate (self-signed, LetsEncrypt, or GCP CA)

---

## Step 1: Create Backend Instance Group

```bash
# Create instance group (example with 2 VMs)
gcloud compute instance-groups managed create ollama-ig \
  --base-instance-name=ollama \
  --template=ollama-instance-template \
  --size=2 \
  --region=us-central1

# Set autoscaling
gcloud compute instance-groups managed set-autoscaling ollama-ig \
  --max-num-replicas 5 \
  --min-num-replicas 2 \
  --target-cpu-utilization 0.60 \
  --region=us-central1
```

---

## Step 2: Create Health Check

```bash
# Create HTTP health check
gcloud compute health-checks create http ollama-health-check \
  --port=8000 \
  --request-path=/health \
  --check-interval=10s \
  --timeout=5s \
  --unhealthy-threshold=2 \
  --healthy-threshold=2
```

---

## Step 3: Create Backend Service

```bash
# Create backend service
gcloud compute backend-services create ollama-backend \
  --global \
  --protocol=HTTP \
  --port-name=http \
  --health-checks=ollama-health-check \
  --enable-cdn \
  --session-affinity=CLIENT_IP \
  --affinityCookieTtlSec=3600

# Add instance group to backend service
gcloud compute backend-services add-backend ollama-backend \
  --instance-group=ollama-ig \
  --instance-group-region=us-central1 \
  --global
```

### Backend Service Configuration (terraform/gcp.tf)

```hcl
resource "google_compute_backend_service" "ollama" {
  name            = "ollama-backend"
  protocol        = "HTTP"
  port_name       = "http"
  timeout_sec     = 300
  enable_cdn      = true
  
  health_checks   = [google_compute_health_check.ollama.id]
  
  backend {
    group           = google_compute_instance_group_manager.ollama.instance_group
    balancing_mode  = "RATE"
    max_rate_per_instance = 100
  }
  
  connection_draining_timeout_sec = 30
  
  # Session affinity for stateful connections
  session_affinity = "CLIENT_IP"
  affinity_cookie_ttl_sec = 3600
  
  custom_request_headers {
    headers = ["X-Forwarded-For", "X-Forwarded-Proto"]
  }
}
```

---

## Step 4: Create URL Map

```bash
# Create URL map for routing
gcloud compute url-maps create ollama-url-map \
  --default-service=ollama-backend

# Create path rule for /ollama/* paths (optional)
gcloud compute url-maps add-path-rule ollama-url-map \
  --default-service=ollama-backend \
  --service=ollama-backend \
  --path-rule='*'
```

---

## Step 5: Create SSL Certificate

### Option A: Using Google-Managed Certificate

```bash
gcloud compute ssl-certificates create ollama-cert \
  --domains=elevatediq.ai,api.elevatediq.ai
```

### Option B: Using Custom Certificate

```bash
# Upload existing certificate
gcloud compute ssl-certificates create ollama-cert \
  --certificate=path/to/cert.crt \
  --private-key=path/to/key.pem
```

### Option C: Using Let's Encrypt (Terraform)

```hcl
resource "google_compute_managed_ssl_certificate" "ollama" {
  name = "ollama-cert"
  
  managed {
    domains = ["elevatediq.ai", "api.elevatediq.ai"]
  }
}
```

---

## Step 6: Create HTTPS Proxy

```bash
# Create target HTTPS proxy
gcloud compute target-https-proxies create ollama-https-proxy \
  --url-map=ollama-url-map \
  --ssl-certificates=ollama-cert \
  --ssl-policy=ollama-ssl-policy

# Create SSL policy (if needed)
gcloud compute ssl-policies create ollama-ssl-policy \
  --profile=MODERN \
  --min-tls-version=TLS_1_2
```

---

## Step 7: Create Forwarding Rule

```bash
# Reserve static IP
gcloud compute addresses create ollama-ip --global

# Get IP address
gcloud compute addresses describe ollama-ip --global

# Create forwarding rule
gcloud compute forwarding-rules create ollama-https-rule \
  --global \
  --target-https-proxy=ollama-https-proxy \
  --address=ollama-ip \
  --ports=443

# Create HTTP to HTTPS redirect
gcloud compute forwarding-rules create ollama-http-rule \
  --global \
  --target-http-proxy=ollama-http-proxy \
  --address=ollama-ip \
  --ports=80

gcloud compute target-http-proxies create ollama-http-proxy \
  --url-map=ollama-redirect-url-map

gcloud compute url-maps create ollama-redirect-url-map \
  --default-service=ollama-backend
```

---

## Step 8: DNS Configuration

Add DNS records pointing to the load balancer IP:

```dns
# A record
elevatediq.ai          A    <LOAD_BALANCER_IP>
api.elevatediq.ai      A    <LOAD_BALANCER_IP>

# Or using CNAME
api.elevatediq.ai      CNAME    elevatediq.ai
```

---

## Step 9: Configure Ollama for Public Endpoint

### Environment Variables

```bash
# On Ollama backend VMs
export OLLAMA_PUBLIC_URL=https://elevatediq.ai/ollama
export OLLAMA_DOMAIN=elevatediq.ai
export OLLAMA_ENABLE_TLS=false  # TLS terminated at LB
```

### Configuration File (production.yaml)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  public_url: "https://elevatediq.ai/ollama"
  domain: "elevatediq.ai"
  
  security_headers:
    strict_transport_security: "max-age=31536000; includeSubDomains; preload"
    x_content_type_options: "nosniff"
    x_frame_options: "DENY"

security:
  api_key_auth_enabled: true
  api_key_header: "X-API-Key"
  cors_origins:
    - "https://elevatediq.ai"
    - "https://*.elevatediq.ai"
  tls_enabled: false  # LB handles TLS
```

---

## Step 10: Cloud Armor (DDoS Protection)

```bash
# Create security policy
gcloud compute security-policies create ollama-security-policy \
  --type=CLOUD_ARMOR

# Add rate limiting rule
gcloud compute security-policies rules create 100 \
  --security-policy=ollama-security-policy \
  --action=rate-based-ban \
  --rate-limit-options=enforced-cap-elements=CLIENT_IP \
  --rate-limit-options=rate-limit-threshold-count=100 \
  --rate-limit-options=rate-limit-threshold-interval-sec=60 \
  --ban-duration-sec=600

# Attach to backend service
gcloud compute backend-services update ollama-backend \
  --security-policy=ollama-security-policy \
  --global
```

---

## Step 11: Monitoring & Logging

### Enable Cloud Logging

```bash
gcloud compute backend-services update ollama-backend \
  --global \
  --enable-logging \
  --logging-sample-rate=1.0
```

### Create Monitoring Dashboard

```bash
# View metrics in Cloud Monitoring
# Navigate to Monitoring > Dashboards
# Add widgets for:
# - Backend latency
# - Request count
# - Error rate (4xx, 5xx)
# - CPU utilization
```

---

## Testing

### Test Public Endpoint

```bash
# Health check
curl -v https://elevatediq.ai/ollama/health

# With API key
curl -H "X-API-Key: your-api-key" \
  https://elevatediq.ai/ollama/api/models

# Generate text
curl -X POST https://elevatediq.ai/ollama/api/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "llama2",
    "prompt": "Hello",
    "stream": false
  }'
```

### Monitor Load Balancer Metrics

```bash
# View traffic distribution
gcloud compute backend-services get-health ollama-backend --global

# View latency
gcloud logging read "resource.type=http_load_balancer" \
  --limit 50 \
  --format json
```

---

## Troubleshooting

### Unhealthy Backends

```bash
# Check health check results
gcloud compute backend-services get-health ollama-backend --global

# SSH into VM and test locally
gcloud compute ssh ollama-instance-1

# Check service is running
sudo systemctl status ollama

# Test health endpoint
curl http://localhost:8000/health
```

### Certificate Issues

```bash
# Check certificate status
gcloud compute ssl-certificates list

gcloud compute ssl-certificates describe ollama-cert

# Renew certificate (if using Let's Encrypt)
# Automatic with managed certificates
```

### High Latency

1. Check backend health
2. Review Cloud Monitoring metrics
3. Verify instance specs (CPU, memory)
4. Check network latency
5. Review application logs

---

## Terraform Configuration

Complete infrastructure-as-code example:

```hcl
# Provider
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# Health Check
resource "google_compute_health_check" "ollama" {
  name               = "ollama-health-check"
  check_interval_sec = 10
  timeout_sec        = 5

  http_health_check {
    port         = 8000
    request_path = "/health"
  }
}

# Backend Service
resource "google_compute_backend_service" "ollama" {
  name            = "ollama-backend"
  protocol        = "HTTP"
  port_name       = "http"
  timeout_sec     = 300
  health_checks   = [google_compute_health_check.ollama.id]
  enable_cdn      = true
  session_affinity = "CLIENT_IP"
}

# URL Map
resource "google_compute_url_map" "ollama" {
  name            = "ollama-url-map"
  default_service = google_compute_backend_service.ollama.id
}

# SSL Certificate
resource "google_compute_managed_ssl_certificate" "ollama" {
  name = "ollama-cert"
  managed {
    domains = ["elevatediq.ai"]
  }
}

# HTTPS Proxy
resource "google_compute_target_https_proxy" "ollama" {
  name             = "ollama-https-proxy"
  url_map          = google_compute_url_map.ollama.id
  ssl_certificates = [google_compute_managed_ssl_certificate.ollama.id]
}

# Forwarding Rule
resource "google_compute_global_forwarding_rule" "ollama" {
  name       = "ollama-https-rule"
  ip_version = "IPV4"
  load_balancing_scheme = "EXTERNAL"
  port_range = "443"
  target     = google_compute_target_https_proxy.ollama.id
  ip_address = google_compute_global_address.ollama.address
}

# Static IP
resource "google_compute_global_address" "ollama" {
  name = "ollama-ip"
}
```

---

## Performance Optimization

### CDN Configuration

```bash
# Enable CDN with custom cache settings
gcloud compute backend-services update ollama-backend \
  --global \
  --enable-cdn \
  --cache-mode=CACHE_ALL_STATIC \
  --client-ttl=3600 \
  --default-ttl=3600 \
  --max-ttl=86400
```

### Session Affinity

```bash
# Sticky sessions for stateful inference
gcloud compute backend-services update ollama-backend \
  --global \
  --session-affinity=CLIENT_IP \
  --affinityCookieTtlSec=3600
```

---

## Cost Optimization

1. **Use committed use discounts** for reserved IPs and compute
2. **Enable CDN** to reduce egress costs
3. **Use Cloud Armor** for DDoS protection (cheaper than attacks)
4. **Implement request filtering** to avoid unnecessary backend calls
5. **Monitor and alert** on quota usage

---

## Security Best Practices

1. ✅ Use HTTPS for all traffic (TLS 1.2+)
2. ✅ Implement Cloud Armor for DDoS protection
3. ✅ Enable API key authentication
4. ✅ Use VPC Service Controls for network isolation
5. ✅ Enable Cloud Audit Logs
6. ✅ Restrict IAM permissions
7. ✅ Implement rate limiting
8. ✅ Use Cloud KMS for secrets management

---

**Version**: 1.0.0  
**Last Updated**: January 12, 2026  
**Maintained by**: elevatediq.ai engineering team
