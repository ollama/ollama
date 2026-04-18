# GCP Load Balancer Setup - elevatediq.ai/ollama

## Architecture Overview

```
Internet
    ↓
GCP Load Balancer (elevatediq.ai/ollama)
  - Path-based routing: /ollama/*
  - TLS Termination
  - Internet NEG (Firewalla DDNS)
    ↓
Firewalla DDNS (d8r978f08m4.d.firewalla.org)
    ↓
Local Backend (192.168.168.42:11000)
  - FastAPI Application
  - Real Ollama Integration (port 8000)
  - PostgreSQL, Redis, Qdrant
```

---

## GCP Components

### 1. Static IP Address

**Name:** `ollama-static-ip`
**Address:** `136.110.229.243`
**Region:** Global

```bash
gcloud compute addresses create ollama-static-ip \
    --global \
    --ip-version=IPV4
```

### 2. SSL Certificate

**Certificate Name:** `elevatediq-ssl-cert`
**Type:** Google-managed
**Domain:** `elevatediq.ai`
**Status:** PROVISIONING

```bash
gcloud compute ssl-certificates create elevatediq-ssl-cert \
    --domains=elevatediq.ai \
    --global
```

**Legacy Certificate:** `ollama-ssl-cert` (ollama.elevatediq.ai) - kept for backward compatibility

### 3. Internet Network Endpoint Group (NEG)

**Name:** `ollama-internet-neg`
**Type:** INTERNET_FQDN_PORT
**Endpoint:** `d8r978f08m4.d.firewalla.org:11000`

```bash
gcloud compute network-endpoint-groups create ollama-internet-neg \
    --network-endpoint-type=INTERNET_FQDN_PORT \
    --global

gcloud compute network-endpoint-groups update ollama-internet-neg \
    --add-endpoint="fqdn=d8r978f08m4.d.firewalla.org,port=11000" \
    --global
```

### 4. Backend Service

**Name:** `ollama-backend`
**Protocol:** HTTPS
**Port:** 11000
**Health Checks:** None (not compatible with Internet NEG)

```bash
gcloud compute backend-services create ollama-backend \
    --protocol=HTTPS \
    --port-name=https \
    --global

gcloud compute backend-services add-backend ollama-backend \
    --network-endpoint-group=ollama-internet-neg \
    --global-network-endpoint-group \
    --global
```

### 5. URL Map (Path-Based Routing)

**Name:** `ollama-url-map`
**Host Rule:** `elevatediq.ai`
**Path Matcher:** `ollama-paths`

Routes traffic:
- `/ollama` → ollama-backend
- `/ollama/*` → ollama-backend

```bash
# URL map configuration (imported from YAML)
gcloud compute url-maps import ollama-url-map \
    --source=urlmap.yaml \
    --global
```

**urlmap.yaml:**
```yaml
defaultService: https://www.googleapis.com/compute/v1/projects/govai-scout/global/backendServices/ollama-backend
name: ollama-url-map
hostRules:
- hosts:
  - elevatediq.ai
  pathMatcher: ollama-paths
pathMatchers:
- name: ollama-paths
  defaultService: https://www.googleapis.com/compute/v1/projects/govai-scout/global/backendServices/ollama-backend
  pathRules:
  - paths:
    - /ollama
    - /ollama/*
    service: https://www.googleapis.com/compute/v1/projects/govai-scout/global/backendServices/ollama-backend
```

### 6. Target HTTPS Proxy

**Name:** `ollama-target-proxy`
**URL Map:** `ollama-url-map`
**SSL Certificates:** 
- `elevatediq-ssl-cert` (primary - elevatediq.ai)
- `ollama-ssl-cert` (legacy - ollama.elevatediq.ai)

```bash
gcloud compute target-https-proxies create ollama-target-proxy \
    --url-map=ollama-url-map \
    --ssl-certificates=elevatediq-ssl-cert,ollama-ssl-cert \
    --global
```

### 7. Global Forwarding Rule

**Name:** `ollama-https-forwarding-rule`
**IP Address:** `136.110.229.243`
**Port:** 443
**Target:** `ollama-target-proxy`

```bash
gcloud compute forwarding-rules create ollama-https-forwarding-rule \
    --address=ollama-static-ip \
    --global \
    --target-https-proxy=ollama-target-proxy \
    --ports=443
```

---

## DNS Configuration

**Domain:** `elevatediq.ai/ollama`
**Static IP:** `136.110.229.243`

### DNS Record

Add an A record in your DNS provider:

```
Type: A
Name: @ (or root/apex)
Value: 136.110.229.243
TTL: 3600
```

**Verification:**
```bash
dig elevatediq.ai +short
# Should return: 136.110.229.243
```

---

## Firewalla Configuration

### Required Firewall Rule

**Purpose:** Allow GCP Load Balancer to reach local backend

1. Open Firewalla admin console
2. Navigate to **Rules** → **Port Forwarding** or **Firewall Rules**
3. Add inbound rule:
   - **Port:** 11000
   - **Protocol:** TCP
   - **Source:** 0.0.0.0/0 (or restrict to GCP IP ranges)
   - **Destination:** 192.168.168.42:11000

### DDNS Status

**DDNS Hostname:** `d8r978f08m4.d.firewalla.org`
**Resolved IP:** `71.245.249.168`

```bash
# Verify DDNS resolution
dig d8r978f08m4.d.firewalla.org +short
# Should return: 71.245.249.168
```

---

## Testing

### 1. Check SSL Certificate Status

```bash
gcloud compute ssl-certificates describe elevatediq-ssl-cert --global
```

Wait for status: `ACTIVE` (takes 10-20 minutes after DNS propagation)

### 2. Test Backend Directly (via Firewalla DDNS)

```bash
curl http://d8r978f08m4.d.firewalla.org:11000/health
```

### 3. Test Load Balancer Endpoints

#### Health Endpoint
```bash
curl https://elevatediq.ai/ollama/health
```

#### Models List
```bash
curl https://elevatediq.ai/ollama/v1/models
```

#### Generate Text
```bash
curl -X POST https://elevatediq.ai/ollama/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder:6.7b",
    "prompt": "Write a Python function to calculate fibonacci",
    "stream": false
  }'
```

#### Chat Completion
```bash
curl -X POST https://elevatediq.ai/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder:6.7b",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Explain async/await in Python"}
    ]
  }'
```

---

## Monitoring

### Check Load Balancer Status

```bash
# Backend service health
gcloud compute backend-services get-health ollama-backend --global

# SSL certificate status
gcloud compute ssl-certificates list --format="table(name,domains,managed.status)"

# Forwarding rule
gcloud compute forwarding-rules describe ollama-https-forwarding-rule --global
```

### View Logs

```bash
# Enable logging on backend service
gcloud compute backend-services update ollama-backend \
    --enable-logging \
    --logging-sample-rate=1.0 \
    --global

# View logs in Cloud Console
# Logging → Logs Explorer → Filter: resource.type="http_load_balancer"
```

---

## Cost Estimate

### Monthly Costs (us-central1)

| Component | Cost | Notes |
|-----------|------|-------|
| Forwarding Rule | $18/month | One global forwarding rule |
| Backend Service | Free | No extra charge |
| Static IP | $7.20/month | Reserved global IP |
| SSL Certificate | Free | Google-managed cert |
| **Estimated Total** | **~$25/month** | Excludes traffic egress |

**Traffic Egress:** $0.085/GB for internet egress from us-central1

---

## Troubleshooting

### SSL Certificate Not Provisioning

1. Verify DNS is correctly configured:
   ```bash
   dig elevatediq.ai +short
   ```

2. Wait 10-20 minutes after DNS propagation

3. Check certificate status:
   ```bash
   gcloud compute ssl-certificates describe elevatediq-ssl-cert --global
   ```

### Backend Unreachable

1. Verify Firewalla DDNS resolves correctly:
   ```bash
   dig d8r978f08m4.d.firewalla.org +short
   ```

2. Test direct access to backend:
   ```bash
   curl http://d8r978f08m4.d.firewalla.org:11000/health
   ```

3. Check Firewalla firewall rules allow inbound port 11000

4. Verify local backend is running:
   ```bash
   curl http://192.168.168.42:11000/health
   ```

### 502 Bad Gateway

- Backend service is unreachable from GCP
- Check Firewalla firewall rules
- Verify Internet NEG endpoint is correct
- Ensure local FastAPI app is running

---

## Cleanup (Optional)

To delete all GCP Load Balancer resources:

```bash
# Delete forwarding rule
gcloud compute forwarding-rules delete ollama-https-forwarding-rule --global

# Delete target proxy
gcloud compute target-https-proxies delete ollama-target-proxy --global

# Delete URL map
gcloud compute url-maps delete ollama-url-map --global

# Delete backend service
gcloud compute backend-services delete ollama-backend --global

# Delete NEG
gcloud compute network-endpoint-groups delete ollama-internet-neg --global

# Delete SSL certificates
gcloud compute ssl-certificates delete elevatediq-ssl-cert --global
gcloud compute ssl-certificates delete ollama-ssl-cert --global

# Release static IP
gcloud compute addresses delete ollama-static-ip --global
```

---

## Summary

✅ **Configured Components:**
- Static IP: `136.110.229.243`
- Domain: `elevatediq.ai/ollama` (path-based routing)
- SSL Certificate: `elevatediq-ssl-cert` (PROVISIONING)
- Internet NEG: Firewalla DDNS endpoint
- Backend Service: Connected to Load Balancer
- URL Map: Path matcher for `/ollama/*`

⏳ **Pending:**
- DNS A record: `elevatediq.ai → 136.110.229.243`
- SSL certificate provisioning (10-20 minutes)
- Firewalla firewall: Allow inbound port 11000

🧪 **Test After DNS + SSL:**
```bash
curl https://elevatediq.ai/ollama/health
curl https://elevatediq.ai/ollama/v1/models
```

---

## Day 2 Ops (Runbook)

### Alerting (minimum useful alerts)
- **HTTPS 5xx rate**: trigger if >1% for 5 minutes on `elevatediq.ai/ollama/*`.
- **Latency (p99)**: trigger if >2s for 5 minutes.
- **Cert status/expiry**: alert if Google-managed cert is not `ACTIVE` or <14 days to expiry.
- **Backend reachability**: alert if `/ollama/health` fails 3 consecutive checks.

### Dashboards to watch
- **Load balancer**: Requests, 4xx/5xx rates, backend connect errors.
- **Application**: p95/p99 latency, error rate, rate-limit hits.
- **Infra**: CPU/memory on backend host; Redis/Postgres health.

### Health probes (cron-friendly)
```bash
# External path + cert
curl -fsSL https://elevatediq.ai/ollama/health

# Backend direct (Firewalla DDNS)
curl -fsSL http://d8r978f08m4.d.firewalla.org:11000/health
```

### Failover / break-glass
1) **Bypass LB**: hit `http://d8r978f08m4.d.firewalla.org:11000` directly.
2) **Rotate cert**: recreate Google-managed cert if stuck in PROVISIONING; confirm DNS A record first.
3) **Endpoint swap**: update the Internet NEG endpoint to a standby host if primary is down.
4) **Firewall adjust**: temporarily relax Firewalla source ranges to restore connectivity, then re-tighten.

### Change management
- Apply changes in low-traffic windows; validate `/ollama/health` before/after.
- Keep prior NEG endpoint and cert available for fast rollback.
- Adjust rate limits cautiously and watch 5xx/latency while doing so.

### SLOs (initial targets)
- Availability: 99.9% monthly for `/ollama/*`.
- Latency: p99 < 2s; p95 < 1s.
- Error budget: 0.1% 5xx over 30 days.

### Incident quick checks
```bash
# DNS
dig elevatediq.ai +short

# Cert status
gcloud compute ssl-certificates describe elevatediq-ssl-cert --global | grep 'managed:' -A3

# URL map routing
gcloud compute url-maps describe ollama-url-map --global | grep -A2 'pathMatchers'

# Backend reachable via LB
curl -i https://elevatediq.ai/ollama/health

# Backend direct
curl -i http://d8r978f08m4.d.firewalla.org:11000/health
```

### Post-incident checklist
- Root cause captured in runbook with timeline and impact.
- Any temporary firewall loosening reverted.
- NEG endpoint/cert restored to intended values.
- Monitoring gaps closed with alerts/dashboards.
- User communication sent if impacted.
