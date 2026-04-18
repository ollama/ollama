# 🚀 Deployment Checklist for elevatediq.ai/ollama

## Pre-Deployment Phase

### GCP Project Setup
- [ ] Google Cloud project created
- [ ] Billing enabled
- [ ] APIs enabled:
  - [ ] Compute Engine API
  - [ ] Cloud Load Balancing API
  - [ ] Cloud Logging API
  - [ ] Cloud Monitoring API
  - [ ] GKE API (if using Kubernetes)
  - [ ] Cloud Armor API
- [ ] Service account created with proper roles
- [ ] Key authentication configured

### DNS & Domain Setup
- [ ] Domain `elevatediq.ai` verified in GCP
- [ ] DNS records configured:
  - [ ] A record for `elevatediq.ai` → Load Balancer IP
  - [ ] A record for `api.elevatediq.ai` → Load Balancer IP
  - [ ] CNAME for CDN distribution (if applicable)
- [ ] DNS propagated and verified

### SSL/TLS Certificates
- [ ] Certificate for `elevatediq.ai` obtained
- [ ] Certificate for `*.elevatediq.ai` obtained
- [ ] Certificate uploaded to GCP (or auto-managed)
- [ ] Certificate expiration monitoring setup
- [ ] Certificate renewal process documented

### Infrastructure Planning
- [ ] Deployment method selected:
  - [ ] GKE (Kubernetes) OR
  - [ ] Compute Engine (VMs)
- [ ] Instance type/size finalized:
  - [ ] CPU requirements
  - [ ] GPU requirements (if needed)
  - [ ] Memory requirements
  - [ ] Storage requirements
- [ ] Auto-scaling limits set:
  - [ ] Minimum replicas: 2
  - [ ] Maximum replicas: 10
  - [ ] Target CPU utilization: 70%

---

## Docker & Container Setup

### Container Image
- [ ] Dockerfile reviewed and tested locally
- [ ] `.dockerignore` configured
- [ ] Multi-stage build optimized
- [ ] HEALTHCHECK directive added
- [ ] Environment variables documented
- [ ] Docker image built locally
- [ ] Docker image tested
- [ ] Docker image tagged: `gcr.io/PROJECT/ollama:prod`
- [ ] Docker image pushed to Google Artifact Registry
- [ ] Image pull verified from GCP

### Container Security
- [ ] Non-root user configured
- [ ] Read-only filesystem enforced where possible
- [ ] Resource limits set:
  - [ ] CPU limits: 4 cores
  - [ ] Memory limits: 8GB
  - [ ] Memory requests: 4GB
- [ ] Security scanning enabled on registry

---

## Load Balancer Configuration

### Health Checks
- [ ] Health check created
- [ ] Path: `/health`
- [ ] Port: `8000`
- [ ] Check interval: `10s`
- [ ] Timeout: `5s`
- [ ] Healthy threshold: `2`
- [ ] Unhealthy threshold: `3`
- [ ] Health check tested and verified

### Backend Service
- [ ] Backend service created
- [ ] Load balancing scheme: `EXTERNAL`
- [ ] Session affinity: `CLIENT_IP`
- [ ] Connection draining: `30s`
- [ ] CDN enabled: `true`
- [ ] Cache mode: `CACHE_ALL_STATIC`
- [ ] Negative caching: enabled
- [ ] Health checks attached

### URL Map & Routing
- [ ] URL map created
- [ ] Routes configured:
  - [ ] `/` → backend service
  - [ ] `/health` → backend service
  - [ ] `/api/*` → backend service
  - [ ] `/v1/*` → backend service
- [ ] Default backend set
- [ ] Path matchers verified

### SSL/TLS Configuration
- [ ] SSL policy created
- [ ] TLS version: `TLS_1_2` minimum
- [ ] HTTPS proxy created
- [ ] Certificate attached to proxy
- [ ] Certificate binding verified
- [ ] SSL scan test passed

### Forwarding Rules
- [ ] Forwarding rule created (HTTPS)
- [ ] Static IP allocated and named
- [ ] IP address noted and documented
- [ ] Firewall rules created:
  - [ ] Allow HTTPS (443)
  - [ ] Allow HTTP (80) for redirect
- [ ] Firewall rules verified

### Cloud Armor Setup
- [ ] Security policy created
- [ ] Rate limiting rule:
  - [ ] Rate: 100 requests/minute
  - [ ] Action: `deny(429)`
- [ ] DDoS protection rules added
- [ ] IP whitelist configured (if needed)
- [ ] Geo-blocking configured (if needed)
- [ ] Policy attached to backend service

---

## Application Configuration

### Environment Variables
- [ ] `.env` configured for production
- [ ] All required variables set:
  - [ ] `ENVIRONMENT=production`
  - [ ] `OLLAMA_PUBLIC_URL=https://elevatediq.ai/ollama`
  - [ ] `OLLAMA_DOMAIN=elevatediq.ai`
  - [ ] `API_KEY_AUTH_ENABLED=true`
  - [ ] `RATE_LIMIT_ENABLED=true`
  - [ ] `TLS_ENABLED=true`
  - [ ] `WORKERS=8`
- [ ] Sensitive variables in Secret Manager:
  - [ ] API keys
  - [ ] Database passwords
  - [ ] Service account keys
- [ ] Environment variables rotated if reused

### Application Secrets
- [ ] API keys generated and stored
- [ ] Database credentials secured
- [ ] Service account key secured
- [ ] API key rotation policy setup
- [ ] Secret management (Google Secret Manager) configured

### Application Startup
- [ ] Application startup scripts verified
- [ ] Startup time benchmarked
- [ ] Initial health check passes
- [ ] Resource usage monitored during startup
- [ ] Error handling verified

---

## Deployment Execution

### Deployment Steps (GKE)
- [ ] Kubernetes cluster verified
- [ ] Namespace created: `ollama-prod`
- [ ] Secrets created from variables
- [ ] Deployment manifest applied
- [ ] Service created with type `LoadBalancer` or `ClusterIP`
- [ ] Horizontal Pod Autoscaler configured
- [ ] Pod startup verified
- [ ] Pod logs checked for errors
- [ ] Pod resource usage monitored

### Deployment Steps (Compute Engine)
- [ ] Instance template created
- [ ] Startup script verified
- [ ] Instance group created
- [ ] Min instances: 2
- [ ] Max instances: 10
- [ ] Target CPU: 70%
- [ ] Instances launched
- [ ] Instance startup logs reviewed
- [ ] SSH access verified (for debugging)

### Network Configuration
- [ ] VPC network configured
- [ ] Subnet created with CIDR range
- [ ] Firewall rules created:
  - [ ] Allow internal communication
  - [ ] Allow health check traffic
  - [ ] Allow LB to backend traffic
  - [ ] Allow egress to external APIs
- [ ] NAT or VPN configured if needed
- [ ] VPC Flow Logs enabled

---

## Testing & Validation

### Connectivity Tests
- [ ] DNS resolution verified
  ```bash
  nslookup elevatediq.ai
  ```
- [ ] HTTPS endpoint accessible
  ```bash
  curl -I https://elevatediq.ai/ollama
  ```
- [ ] Health check responds
  ```bash
  curl https://elevatediq.ai/ollama/health
  ```

### API Tests
- [ ] Health endpoint:
  ```bash
  curl -H "X-API-Key: TEST_KEY" https://elevatediq.ai/ollama/health
  ```
- [ ] Models endpoint:
  ```bash
  curl -H "X-API-Key: TEST_KEY" https://elevatediq.ai/ollama/api/models
  ```
- [ ] Generate endpoint:
  ```bash
  curl -X POST https://elevatediq.ai/ollama/api/generate \
    -H "X-API-Key: TEST_KEY" \
    -d '{"model":"llama2","prompt":"test"}'
  ```
- [ ] Chat endpoint (OpenAI-compatible)
- [ ] Embeddings endpoint
- [ ] Admin stats endpoint

### Authentication Tests
- [ ] Missing API key returns 401
- [ ] Invalid API key returns 401
- [ ] Valid API key accepts request
- [ ] Bearer token authentication works
- [ ] CORS headers present on OPTIONS request

### Rate Limiting Tests
- [ ] First 100 requests succeed
- [ ] 101st request in minute returns 429
- [ ] Rate limit headers present:
  - [ ] X-RateLimit-Limit: 100
  - [ ] X-RateLimit-Remaining
  - [ ] X-RateLimit-Reset
- [ ] Burst capacity (150) verified
- [ ] Rate limit resets after window

### Security Tests
- [ ] HTTPS only (HTTP redirects or fails)
- [ ] TLS certificate valid
- [ ] TLS version >= 1.2
- [ ] Security headers present:
  - [ ] Strict-Transport-Security
  - [ ] X-Content-Type-Options
  - [ ] X-Frame-Options
  - [ ] X-XSS-Protection
  - [ ] Content-Security-Policy
- [ ] CORS origin whitelist enforced

### Performance Tests
- [ ] Health check response < 100ms
- [ ] API response latency acceptable
- [ ] Load test with expected traffic
- [ ] Load test with burst traffic (150 req/min)
- [ ] CPU utilization stays below 70% (normal load)
- [ ] Memory usage stable
- [ ] No memory leaks observed

### Failover Tests
- [ ] Backend service marked unhealthy
- [ ] Traffic routes to other backends
- [ ] Health check recovery triggers
- [ ] Pod recovers and rejoins backend
- [ ] Zero downtime verified

### Rollback Tests
- [ ] Previous version accessible (blue-green)
- [ ] Rollback procedure documented
- [ ] Rollback procedure tested
- [ ] Rollback time measured
- [ ] Database migrations reversible

---

## Monitoring & Alerting Setup

### Cloud Monitoring
- [ ] Monitoring dashboard created
- [ ] Metrics configured:
  - [ ] Request rate (requests/sec)
  - [ ] Error rate (4xx, 5xx)
  - [ ] Latency (p50, p95, p99)
  - [ ] Backend health status
  - [ ] CPU utilization
  - [ ] Memory utilization
  - [ ] Network throughput
  - [ ] Rate limit hits
- [ ] Custom metrics configured
- [ ] Dashboard published
- [ ] Access permissions set

### Alerting Policies
- [ ] Alert for error rate > 5%
- [ ] Alert for latency p99 > 1000ms
- [ ] Alert for backend unhealthy
- [ ] Alert for certificate expiration (30 days)
- [ ] Alert for rate limit attacks (>10x normal)
- [ ] Alert for CPU > 80%
- [ ] Alert for memory > 85%
- [ ] Notification channels configured:
  - [ ] Email
  - [ ] Slack
  - [ ] PagerDuty (if applicable)

### Cloud Logging
- [ ] Log sink configured for LB
- [ ] Log sink configured for backends
- [ ] Log sink configured for Cloud Armor
- [ ] Logs retained for 30 days minimum
- [ ] Log aggregation dashboard created
- [ ] Log search queries saved for common issues

### Distributed Tracing
- [ ] Jaeger or equivalent configured (optional)
- [ ] Trace sampling configured (1% for production)
- [ ] Trace queries for latency analysis
- [ ] Integration with Cloud Logging

---

## Documentation & Runbooks

### Documentation
- [ ] Deployment guide reviewed: `docs/public-deployment.md`
- [ ] GCP LB guide reviewed: `docs/gcp-load-balancer.md`
- [ ] API reference reviewed: `PUBLIC_API.md`
- [ ] Configuration documented: `.env.example`
- [ ] Architecture documented: `docs/architecture.md`
- [ ] Deployment architecture diagram included
- [ ] Network diagram documented
- [ ] Database schema documented
- [ ] API endpoint documentation current

### Runbooks
- [ ] Incident response runbook created
- [ ] Escalation procedures documented
- [ ] Rollback procedures documented
- [ ] Database recovery procedures documented
- [ ] Key rotation procedures documented
- [ ] Certificate renewal procedures documented
- [ ] Scaling adjustment procedures documented

### Team Handoff
- [ ] Operations team trained
- [ ] Support team briefed on API
- [ ] Development team documentation reviewed
- [ ] On-call rotation established
- [ ] Escalation contacts documented
- [ ] Change management process defined

---

## Post-Deployment Phase

### Day 1 Monitoring
- [ ] System stable (no errors)
- [ ] Latency within expectations
- [ ] Memory stable (no leaks)
- [ ] Disk space healthy
- [ ] Network throughput normal
- [ ] Health checks passing consistently

### Week 1 Validation
- [ ] No alerts triggered (false positives checked)
- [ ] Performance baseline established
- [ ] User feedback positive
- [ ] Client library working correctly
- [ ] Documentation accurate
- [ ] Support team confident

### Month 1 Review
- [ ] Cost analysis completed
- [ ] Performance optimizations applied (if needed)
- [ ] Auto-scaling limits adjusted (if needed)
- [ ] Monitoring refined (low signal-to-noise ratio)
- [ ] Documentation updated from experience
- [ ] Lessons learned documented

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| DevOps Lead | ________ | ________ | ________ |
| Security Lead | ________ | ________ | ________ |
| Product Lead | ________ | ________ | ________ |
| Engineering Lead | ________ | ________ | ________ |

---

## Related Documents

- [GCP Load Balancer Setup](docs/gcp-load-balancer.md)
- [Public Deployment Guide](docs/public-deployment.md)
- [Public API Reference](PUBLIC_API.md)
- [Environment Configuration](.env.example)
- [Production Configuration](config/production.yaml)

---

**Status**: Use this checklist before deploying to production  
**Maintainer**: elevatediq.ai engineering team  
**Last Updated**: January 12, 2026  
**Version**: 1.0
