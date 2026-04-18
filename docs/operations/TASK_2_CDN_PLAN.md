# Task 2: CDN for Static Assets - Implementation Plan

## Quick Reference

**Objective**: Deploy Cloud CDN for asset distribution to reduce latency and costs
**Estimated Effort**: 40 hours (1 week)
**Expected Impact**: 70% latency reduction for assets, 40% bandwidth savings
**Three-Lens**: CEO ($50K savings), CTO (faster asset delivery), CFO (0.2x cost ratio)

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│      External Clients (Global)          │
└────────────────────┬────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Cloud CDN            │
        │ - Caching layer        │
        │ - 150+ edge locations  │
        │ - HTTP/2 termination   │
        │ - DDoS protection      │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Cloud Storage (GCS)    │
        │ - Docs (Markdown)      │
        │ - Images (PNG, WebP)   │
        │ - Models (ONNX)        │
        │ - Config (YAML, JSON)  │
        └────────────────────────┘
```

---

## Assets to Cache

### 1. Documentation (15 MB)

- Markdown files (auto-compiled to HTML)
- OpenAPI schemas
- Architecture diagrams

### 2. Images & Media (50 MB)

- PNG/WebP format (optimized)
- Logos, icons, diagrams
- Video thumbnails

### 3. Model Artifacts (5 GB+)

- ONNX quantized models
- Model metadata (YAML)
- Configuration files

### 4. Static Assets (2 MB)

- CSS, JavaScript (minified)
- Font files (WOFF2)
- Favicon, manifest

---

## Implementation Steps

### Step 1: GCS Bucket Setup (Terraform)

```hcl
resource "google_storage_bucket" "ollama_assets" {
  name          = "prod-ollama-assets"
  location      = "US"
  force_destroy = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "STANDARD"
    }
    condition {
      age = 30
    }
  }
}

resource "google_storage_bucket_access_control" "public" {
  bucket = google_storage_bucket.ollama_assets.name
  role   = "READER"
  entity = "allUsers"
}
```

### Step 2: Cloud CDN Configuration (Terraform)

```hcl
resource "google_compute_backend_bucket" "ollama_cdn" {
  name            = "prod-ollama-cdn"
  bucket_name     = google_storage_bucket.ollama_assets.name
  enable_cdn      = true
  compression_mode = "AUTOMATIC"

  cdn_policy {
    cache_mode        = "CACHE_ALL_STATIC"
    client_ttl        = 3600       # 1 hour
    default_ttl       = 86400      # 1 day
    max_ttl           = 604800     # 1 week
    negative_caching  = true
    negative_caching_policy {
      code = 404
      ttl  = 120
    }
  }
}
```

### Step 3: Load Balancer Integration

- Route `/assets/*` to CDN backend
- Route `/docs/*` to CDN with HTML cache policy
- Route `/models/*` to CDN with long TTL (7 days)

### Step 4: Cache Invalidation Strategy

```python
from google.cloud import compute_v1

async def invalidate_cache(pattern: str) -> None:
    """Invalidate CDN cache by pattern."""
    client = compute_v1.UrlMapsClient()
    # Invalidate when assets are updated
    # Called during CI/CD deployment
```

### Step 5: Monitoring & Metrics

```yaml
# Prometheus queries
cdn_cache_hit_ratio = (cache_hits / total_requests) * 100
cdn_bandwidth_saved = (cache_bytes / total_bytes) * 100
asset_latency_p99 = histogram_quantile(0.99, cdn_latency)
```

---

## Files to Create

1. **docker/terraform/gcp_cdn.tf** (250 lines)
   - GCS bucket configuration
   - Cloud CDN configuration
   - Load balancer routing rules

2. **ollama/config/cdn.py** (100 lines)
   - CDN configuration models
   - Cache policy definitions
   - Asset type mappings

3. **scripts/sync-assets-to-cdn.py** (150 lines)
   - Upload documentation to GCS
   - Upload images to GCS
   - Optimize images before upload
   - Invalidate CDN cache on update

4. **docs/CDN_IMPLEMENTATION.md** (300 lines)
   - Architecture documentation
   - Configuration guide
   - Monitoring setup
   - Troubleshooting

---

## Success Metrics

| Metric            | Target | Current |
| ----------------- | ------ | ------- |
| Cache Hit Ratio   | 85%+   | TBD     |
| Asset Latency P99 | <100ms | TBD     |
| Bandwidth Savings | 40%+   | 0%      |
| Cost Reduction    | 50%+   | 0%      |

---

## Dependencies

- ✅ Terraform 5.0+ (already deployed)
- ✅ Cloud Load Balancer (existing)
- ✅ Cloud Storage (GCP service)
- ✅ Cloud CDN (GCP service)
- ⏳ Documentation structure (exists, needs migration)

---

## Risk Mitigation

| Risk                     | Mitigation                                  |
| ------------------------ | ------------------------------------------- |
| Cache invalidation lag   | Set short TTL for frequently updated assets |
| Stale content served     | Implement versioned URLs for assets         |
| High egress costs        | Monitor CDN egress and set alerts           |
| Security (public assets) | Use signed URLs for sensitive assets        |

---

## Timeline

- **Week 1**: Terraform configuration + GCS bucket setup
- **Week 1**: Asset sync script + documentation
- **Week 2**: Monitoring and testing
- **Week 2**: Production rollout + optimization

---

## Related Tasks

- Task 1 (Feature Flags): ✅ COMPLETE
- Task 2 (CDN): → You are here
- Task 3 (Chaos Engineering): Uses feature flags for gradual injection
- Task 4 (Automated Failover): Relies on CDN for fast asset serving
- Task 5 (MXdocs Integration): Uses CDN for documentation hosting

---

**Status**: Ready to start implementation
**Owner**: infra-team
**Slack Channel**: #infrastructure-enhancements
