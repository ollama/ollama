# Elite Docker Deployment Architecture Review

## Overview

This document provides a comprehensive deep-dive into each container, focusing on:
- ✅ Elite deployment practices
- ✅ Repeatable, immutable infrastructure
- ✅ GCP backup strategy for large files
- ✅ Production-grade reliability
- ✅ Developer experience optimization

---

## Architecture Principles

### 1. **Immutable Infrastructure**
- Containers are stateless application layers
- State stored in volumes (backed up to GCP)
- Configuration via environment variables
- Version-pinned images (no `latest` in production)

### 2. **12-Factor App Compliance**
- Configuration in environment (`.env`)
- Logs to stdout/stderr (captured by Docker)
- Backing services as attached resources
- Port binding for external visibility

### 3. **GCP Backup Strategy**
- Large files (models, databases, metrics) → GCP Cloud Storage
- Automated daily backups via `gsutil rsync`
- Volume snapshots for disaster recovery
- Point-in-time recovery capability

---

## Container Deep Dive

### 1. Ollama API Container

**Current State:**
```yaml
ollama-api:
  image: ollama:prod  # ❌ No version pinning
  restart: always
  volumes:
    - ollama-models:/models  # Large files, needs GCP backup
```

**Elite Enhancement:**

```yaml
ollama-api:
  # Image Management
  image: ollama:${OLLAMA_VERSION:-1.0.0}  # ✅ Version pinned
  build:
    context: .
    dockerfile: docker/Dockerfile
    target: production
    args:
      PYTHON_VERSION: "3.11"
      BUILD_DATE: "${BUILD_DATE}"
      VCS_REF: "${GIT_COMMIT}"
  
  container_name: ollama-api
  hostname: ollama-api
  restart: unless-stopped  # ✅ Better than 'always'
  
  # Resource Limits (prevents OOM, ensures QoS)
  deploy:
    resources:
      limits:
        cpus: '8'
        memory: 32G
      reservations:
        cpus: '4'
        memory: 16G
        devices:
          - driver: nvidia
            device_ids: ['0']  # Specific GPU
            capabilities: [gpu]
  
  # Security
  security_opt:
    - no-new-privileges:true
  read_only: true  # Immutable root filesystem
  tmpfs:
    - /tmp:size=1G,mode=1777
    - /app/cache:size=2G,mode=1777
  
  # Networking
  ports:
    - "127.0.0.1:8000:8000"  # ✅ Bind to localhost, use nginx proxy
  networks:
    ollama-network:
      aliases:
        - api.ollama.local
  
  # Environment
  env_file:
    - .env
    - .env.production
  environment:
    - LOG_LEVEL=${LOG_LEVEL:-INFO}
    - LOG_FORMAT=json
    - DATABASE_URL=${DATABASE_URL}
    - REDIS_URL=${REDIS_URL}
    - API_KEY_AUTH_ENABLED=true
    - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    - PYTHONUNBUFFERED=1
    - PYTHONDONTWRITEBYTECODE=1
  
  # Volumes (with GCP backup)
  volumes:
    # Models (LARGE - backed up to GCS)
    - type: volume
      source: ollama-models
      target: /models
      read_only: false
      volume:
        nocopy: true
    
    # Config (read-only, immutable)
    - type: bind
      source: ./config/production.yaml
      target: /app/config.yaml
      read_only: true
    
    # Logs (ephemeral, shipped to GCP Logging)
    - type: volume
      source: ollama-logs
      target: /var/log/ollama
  
  # Health Check (for auto-recovery)
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 60s  # ✅ Allow startup time
  
  # Dependencies
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
    qdrant:
      condition: service_healthy
  
  # Logging
  logging:
    driver: "json-file"
    options:
      max-size: "100m"
      max-file: "5"
      labels: "service,environment"
      
  # Labels (for monitoring & management)
  labels:
    com.ollama.service: "api"
    com.ollama.environment: "production"
    com.ollama.version: "${OLLAMA_VERSION}"
    prometheus.scrape: "true"
    prometheus.port: "8000"
    prometheus.path: "/metrics"
```

**Best Practices Applied:**
- ✅ Version pinning for reproducibility
- ✅ Resource limits prevent resource exhaustion
- ✅ Read-only root filesystem (security)
- ✅ Health checks for auto-recovery
- ✅ Structured logging with rotation
- ✅ Network aliases for service discovery
- ✅ Labels for Prometheus auto-discovery

---

### 2. PostgreSQL Container

**Elite Enhancement:**

```yaml
postgres:
  # Image (pinned version)
  image: postgres:15.5-alpine  # ✅ Specific version
  container_name: ollama-postgres
  hostname: postgres
  restart: unless-stopped
  
  # Resource Limits
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
  
  # Security
  security_opt:
    - no-new-privileges:true
  
  # Networking
  ports:
    - "127.0.0.1:5432:5432"  # ✅ Localhost only
  networks:
    ollama-network:
      aliases:
        - db.ollama.local
  
  # Environment
  environment:
    POSTGRES_USER: ollama
    POSTGRES_PASSWORD_FILE: /run/secrets/db_password  # ✅ Use secrets
    POSTGRES_DB: ollama
    POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=en_US.UTF-8"
    PGDATA: /var/lib/postgresql/data/pgdata
    
    # Performance tuning
    POSTGRES_SHARED_BUFFERS: "2GB"
    POSTGRES_EFFECTIVE_CACHE_SIZE: "6GB"
    POSTGRES_MAINTENANCE_WORK_MEM: "512MB"
    POSTGRES_CHECKPOINT_COMPLETION_TARGET: "0.9"
    POSTGRES_WAL_BUFFERS: "16MB"
    POSTGRES_MAX_WAL_SIZE: "4GB"
    POSTGRES_MIN_WAL_SIZE: "1GB"
  
  # Secrets (instead of plain env vars)
  secrets:
    - db_password
  
  # Volumes (with GCP backup)
  volumes:
    # Data (backed up to GCS daily)
    - type: volume
      source: postgres-data
      target: /var/lib/postgresql/data
    
    # Initialization scripts
    - type: bind
      source: ./docker/postgres/init
      target: /docker-entrypoint-initdb.d
      read_only: true
    
    # Custom config
    - type: bind
      source: ./docker/postgres/postgresql.conf
      target: /etc/postgresql/postgresql.conf
      read_only: true
    
    # Backup staging area
    - type: volume
      source: postgres-backups
      target: /backups
  
  # Health Check
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ollama -d ollama"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 30s
  
  # Logging
  logging:
    driver: "json-file"
    options:
      max-size: "50m"
      max-file: "3"
  
  # Labels
  labels:
    com.ollama.service: "database"
    com.ollama.backup: "gcp-daily"
    com.ollama.backup.retention: "30d"
```

**Backup Strategy (GCP):**
```bash
# Automated daily backup script
#!/bin/bash
# /opt/ollama/backup-postgres.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="/backups/ollama_${BACKUP_DATE}.sql.gz"
GCS_BUCKET="gs://elevatediq-ollama-backups/postgres"

# Create backup
docker exec ollama-postgres pg_dump -U ollama -Fc ollama | gzip > "$BACKUP_FILE"

# Upload to GCP
gsutil -m cp "$BACKUP_FILE" "${GCS_BUCKET}/"

# Verify upload
gsutil ls "${GCS_BUCKET}/ollama_${BACKUP_DATE}.sql.gz"

# Clean local backups older than 7 days
find /backups -name "ollama_*.sql.gz" -mtime +7 -delete

# Retention: Keep 30 daily backups in GCS
gsutil ls "${GCS_BUCKET}/" | tail -n +31 | xargs -I {} gsutil rm {}
```

**Cron Entry:**
```bash
# Daily at 2 AM
0 2 * * * /opt/ollama/backup-postgres.sh >> /var/log/ollama-backup.log 2>&1
```

---

### 3. Redis Container

**Elite Enhancement:**

```yaml
redis:
  image: redis:7.2.3-alpine  # ✅ Pinned version
  container_name: ollama-redis
  hostname: redis
  restart: unless-stopped
  
  # Resource Limits
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
  
  # Security
  security_opt:
    - no-new-privileges:true
  
  # Networking
  ports:
    - "127.0.0.1:6379:6379"
  networks:
    ollama-network:
      aliases:
        - cache.ollama.local
  
  # Command with production settings
  command: >
    redis-server
    --requirepass ${REDIS_PASSWORD}
    --appendonly yes
    --appendfsync everysec
    --maxmemory 3gb
    --maxmemory-policy allkeys-lru
    --save 900 1
    --save 300 10
    --save 60 10000
    --tcp-backlog 511
    --timeout 0
    --tcp-keepalive 300
    --daemonize no
    --loglevel notice
  
  # Volumes
  volumes:
    - type: volume
      source: redis-data
      target: /data
    
    - type: bind
      source: ./docker/redis/redis.conf
      target: /usr/local/etc/redis/redis.conf
      read_only: true
  
  # Health Check
  healthcheck:
    test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "${REDIS_PASSWORD}", "ping"]
    interval: 10s
    timeout: 3s
    retries: 3
    start_period: 10s
  
  # Logging
  logging:
    driver: "json-file"
    options:
      max-size: "20m"
      max-file: "3"
  
  # Labels
  labels:
    com.ollama.service: "cache"
    com.ollama.backup: "gcp-daily"
```

**Backup Strategy:**
```bash
# Redis RDB backup to GCS
#!/bin/bash
docker exec ollama-redis redis-cli -a "${REDIS_PASSWORD}" BGSAVE
sleep 10
docker cp ollama-redis:/data/dump.rdb "/backups/redis_$(date +%Y%m%d).rdb"
gsutil cp "/backups/redis_$(date +%Y%m%d).rdb" "gs://elevatediq-ollama-backups/redis/"
```

---

### 4. Qdrant Vector Database

**Elite Enhancement:**

```yaml
qdrant:
  image: qdrant/qdrant:v1.7.3  # ✅ Pinned version
  container_name: ollama-qdrant
  hostname: qdrant
  restart: unless-stopped
  
  # Resource Limits
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
  
  # Security
  security_opt:
    - no-new-privileges:true
  
  # Networking
  ports:
    - "127.0.0.1:6333:6333"  # HTTP API
    - "127.0.0.1:6334:6334"  # gRPC
  networks:
    ollama-network:
      aliases:
        - vector.ollama.local
  
  # Environment
  environment:
    QDRANT__SERVICE__HTTP_PORT: "6333"
    QDRANT__SERVICE__GRPC_PORT: "6334"
    QDRANT__STORAGE__STORAGE_PATH: "/qdrant/storage"
    QDRANT__STORAGE__SNAPSHOTS_PATH: "/qdrant/snapshots"
    QDRANT__SERVICE__ENABLE_TLS: "false"  # TLS at LB level
    QDRANT__LOG_LEVEL: "INFO"
  
  # Volumes
  volumes:
    - type: volume
      source: qdrant-data
      target: /qdrant/storage
    
    - type: volume
      source: qdrant-snapshots
      target: /qdrant/snapshots
  
  # Health Check
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 30s
  
  # Logging
  logging:
    driver: "json-file"
    options:
      max-size: "50m"
      max-file: "3"
  
  # Labels
  labels:
    com.ollama.service: "vector-db"
    com.ollama.backup: "gcp-hourly"
```

**Backup Strategy:**
```bash
# Qdrant snapshot backup to GCS
#!/bin/bash
COLLECTION="embeddings"
SNAPSHOT_NAME="snapshot_$(date +%Y%m%d_%H%M%S)"

# Create snapshot via API
curl -X POST "http://localhost:6333/collections/${COLLECTION}/snapshots"

# Download snapshot
SNAPSHOT_PATH=$(curl "http://localhost:6333/collections/${COLLECTION}/snapshots" | jq -r '.[0].name')
curl "http://localhost:6333/collections/${COLLECTION}/snapshots/${SNAPSHOT_PATH}" \
  -o "/backups/${SNAPSHOT_NAME}.snapshot"

# Upload to GCS
gsutil cp "/backups/${SNAPSHOT_NAME}.snapshot" "gs://elevatediq-ollama-backups/qdrant/"
```

---

### 5. Prometheus (Metrics)

**Elite Enhancement:**

```yaml
prometheus:
  image: prom/prometheus:v2.48.1  # ✅ Pinned
  container_name: ollama-prometheus
  hostname: prometheus
  restart: unless-stopped
  user: "nobody"  # ✅ Non-root
  
  # Resource Limits
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
  
  # Networking
  ports:
    - "127.0.0.1:9090:9090"
  networks:
    ollama-network:
      aliases:
        - metrics.ollama.local
  
  # Volumes
  volumes:
    - type: bind
      source: ./monitoring/prometheus.yml
      target: /etc/prometheus/prometheus.yml
      read_only: true
    
    - type: bind
      source: ./monitoring/alerts.yml
      target: /etc/prometheus/alerts.yml
      read_only: true
    
    - type: volume
      source: prometheus-data
      target: /prometheus
  
  # Command
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.path=/prometheus'
    - '--storage.tsdb.retention.time=30d'
    - '--storage.tsdb.retention.size=10GB'
    - '--web.console.libraries=/etc/prometheus/console_libraries'
    - '--web.console.templates=/etc/prometheus/consoles'
    - '--web.enable-lifecycle'
    - '--web.enable-admin-api'
  
  # Health Check
  healthcheck:
    test: ["CMD", "wget", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
    interval: 30s
    timeout: 10s
    retries: 3
  
  # Labels
  labels:
    com.ollama.service: "monitoring"
    com.ollama.backup: "gcp-weekly"
```

**Backup Strategy:**
```bash
# Backup Prometheus TSDB to GCS
#!/bin/bash
docker exec ollama-prometheus promtool tsdb create-blocks-from openmetrics /prometheus /backups/prom-snapshot
gsutil -m rsync -r /backups/prom-snapshot "gs://elevatediq-ollama-backups/prometheus/"
```

---

### 6. Grafana (Dashboards)

**Elite Enhancement:**

```yaml
grafana:
  image: grafana/grafana:10.2.3  # ✅ Pinned
  container_name: ollama-grafana
  hostname: grafana
  restart: unless-stopped
  user: "472"  # ✅ Non-root grafana user
  
  # Resource Limits
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '0.5'
        memory: 512M
  
  # Networking
  ports:
    - "127.0.0.1:3000:3000"
  networks:
    ollama-network:
      aliases:
        - dashboards.ollama.local
  
  # Environment
  environment:
    GF_SECURITY_ADMIN_PASSWORD__FILE: /run/secrets/grafana_password
    GF_SERVER_ROOT_URL: "https://elevatediq.ai/grafana"
    GF_DATABASE_TYPE: postgres
    GF_DATABASE_HOST: postgres:5432
    GF_DATABASE_NAME: grafana
    GF_DATABASE_USER: grafana
    GF_DATABASE_PASSWORD__FILE: /run/secrets/grafana_db_password
    GF_INSTALL_PLUGINS: "grafana-clock-panel,grafana-simple-json-datasource"
    GF_USERS_ALLOW_SIGN_UP: "false"
    GF_AUTH_ANONYMOUS_ENABLED: "false"
    GF_SECURITY_DISABLE_GRAVATAR: "true"
    GF_ANALYTICS_REPORTING_ENABLED: "false"
    GF_ANALYTICS_CHECK_FOR_UPDATES: "false"
  
  # Secrets
  secrets:
    - grafana_password
    - grafana_db_password
  
  # Volumes
  volumes:
    - type: volume
      source: grafana-data
      target: /var/lib/grafana
    
    - type: bind
      source: ./monitoring/grafana/provisioning
      target: /etc/grafana/provisioning
      read_only: true
    
    - type: bind
      source: ./monitoring/grafana/dashboards
      target: /var/lib/grafana/dashboards
      read_only: true
  
  # Health Check
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
    interval: 30s
    timeout: 10s
    retries: 3
  
  # Dependencies
  depends_on:
    - prometheus
    - postgres
  
  # Labels
  labels:
    com.ollama.service: "visualization"
    com.ollama.backup: "config-only"
```

---

### 7. Jaeger (Tracing)

**Elite Enhancement:**

```yaml
jaeger:
  image: jaegertracing/all-in-one:1.52.0  # ✅ Pinned
  container_name: ollama-jaeger
  hostname: jaeger
  restart: unless-stopped
  
  # Resource Limits
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '0.5'
        memory: 512M
  
  # Networking
  ports:
    - "127.0.0.1:5775:5775/udp"  # Zipkin compact
    - "127.0.0.1:6831:6831/udp"  # Jaeger compact
    - "127.0.0.1:6832:6832/udp"  # Jaeger binary
    - "127.0.0.1:5778:5778"      # Serve configs
    - "127.0.0.1:16686:16686"    # UI
    - "127.0.0.1:14268:14268"    # Direct send
    - "127.0.0.1:14250:14250"    # Model proto
    - "127.0.0.1:9411:9411"      # Zipkin
  networks:
    ollama-network:
      aliases:
        - tracing.ollama.local
  
  # Environment
  environment:
    COLLECTOR_ZIPKIN_HOST_PORT: ":9411"
    COLLECTOR_OTLP_ENABLED: "true"
    SPAN_STORAGE_TYPE: "badger"
    BADGER_EPHEMERAL: "false"
    BADGER_DIRECTORY_VALUE: "/badger/data"
    BADGER_DIRECTORY_KEY: "/badger/key"
  
  # Volumes
  volumes:
    - type: volume
      source: jaeger-data
      target: /badger
  
  # Health Check
  healthcheck:
    test: ["CMD", "wget", "--spider", "http://localhost:14269/"]
    interval: 30s
    timeout: 10s
    retries: 3
  
  # Labels
  labels:
    com.ollama.service: "tracing"
    com.ollama.backup: "gcp-weekly"
```

---

## Additional Elite Containers

### 8. Nginx Reverse Proxy (Recommended)

```yaml
nginx:
  image: nginx:1.25.3-alpine
  container_name: ollama-nginx
  hostname: nginx
  restart: unless-stopped
  
  ports:
    - "80:80"
    - "443:443"
  
  volumes:
    - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    - ./docker/nginx/conf.d:/etc/nginx/conf.d:ro
    - nginx-cache:/var/cache/nginx
    - nginx-logs:/var/log/nginx
  
  networks:
    ollama-network:
      aliases:
        - gateway.ollama.local
  
  depends_on:
    - ollama-api
    - grafana
  
  healthcheck:
    test: ["CMD", "nginx", "-t"]
    interval: 30s
    timeout: 10s
    retries: 3
  
  labels:
    com.ollama.service: "gateway"
```

### 9. GCS Sync Container (Backup Automation)

```yaml
gcs-sync:
  image: google/cloud-sdk:alpine
  container_name: ollama-gcs-sync
  hostname: gcs-sync
  restart: unless-stopped
  
  environment:
    GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/gcp_sa_key
    GCS_BUCKET: "gs://elevatediq-ollama-backups"
  
  secrets:
    - gcp_sa_key
  
  volumes:
    - ollama-models:/data/models:ro
    - postgres-backups:/data/postgres:ro
    - redis-data:/data/redis:ro
    - qdrant-snapshots:/data/qdrant:ro
    - ./scripts/gcs-sync.sh:/app/sync.sh:ro
  
  command: >
    sh -c "
    while true; do
      /app/sync.sh
      sleep 3600
    done
    "
  
  networks:
    - ollama-network
  
  labels:
    com.ollama.service: "backup"
```

**GCS Sync Script:**
```bash
#!/bin/bash
# /scripts/gcs-sync.sh

set -e

echo "Starting GCS sync at $(date)"

# Sync models (large files, incremental)
gsutil -m rsync -r -d /data/models "${GCS_BUCKET}/models/"

# Sync database backups
gsutil -m rsync -r /data/postgres "${GCS_BUCKET}/postgres/"

# Sync Redis snapshots
gsutil -m rsync -r /data/redis "${GCS_BUCKET}/redis/"

# Sync Qdrant snapshots
gsutil -m rsync -r /data/qdrant "${GCS_BUCKET}/qdrant/"

echo "GCS sync completed at $(date)"
```

---

## Volume Management Strategy

### Local Volumes (Docker)
```yaml
volumes:
  # Application Data (small, frequent changes)
  ollama-logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /var/lib/ollama/logs
  
  # Large Data (backed up to GCS)
  ollama-models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/ollama/models  # Large fast disk
  
  postgres-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/ollama/postgres
  
  postgres-backups:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/backups/postgres
  
  redis-data:
    driver: local
  
  qdrant-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/ollama/qdrant
  
  qdrant-snapshots:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/backups/qdrant
  
  prometheus-data:
    driver: local
  
  grafana-data:
    driver: local
  
  jaeger-data:
    driver: local
  
  nginx-cache:
    driver: local
  
  nginx-logs:
    driver: local
```

---

## Secrets Management

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt
  
  redis_password:
    file: ./secrets/redis_password.txt
  
  grafana_password:
    file: ./secrets/grafana_password.txt
  
  grafana_db_password:
    file: ./secrets/grafana_db_password.txt
  
  gcp_sa_key:
    file: ./secrets/gcp-service-account.json
```

**Generate Secrets:**
```bash
mkdir -p secrets
openssl rand -base64 32 > secrets/db_password.txt
openssl rand -base64 32 > secrets/redis_password.txt
openssl rand -base64 32 > secrets/grafana_password.txt
openssl rand -base64 32 > secrets/grafana_db_password.txt
chmod 600 secrets/*
```

---

## Network Configuration

```yaml
networks:
  ollama-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/16
          gateway: 172.25.0.1
    driver_opts:
      com.docker.network.bridge.name: br-ollama
      com.docker.network.bridge.enable_icc: "true"
      com.docker.network.bridge.enable_ip_masquerade: "true"
      com.docker.network.driver.mtu: "1500"
    labels:
      com.ollama.network: "internal"
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Create directory structure: `/mnt/data/ollama`, `/mnt/backups`
- [ ] Generate secrets in `./secrets/`
- [ ] Create `.env.production` with all variables
- [ ] Set up GCP service account with Storage Admin role
- [ ] Create GCS bucket: `elevatediq-ollama-backups`
- [ ] Configure firewall rules (port 8000 for GCP LB)
- [ ] Install NVIDIA drivers and nvidia-docker2
- [ ] Pull all Docker images
- [ ] Set up monitoring directories: `./monitoring/{prometheus,grafana}`

### Deployment
- [ ] Run: `docker-compose -f docker-compose.elite.yml pull`
- [ ] Run: `docker-compose -f docker-compose.elite.yml up -d`
- [ ] Verify: `docker-compose ps`
- [ ] Check logs: `docker-compose logs -f`
- [ ] Test health: `curl http://localhost:8000/health`

### Post-Deployment
- [ ] Set up cron jobs for backups
- [ ] Configure GCP Load Balancer
- [ ] Point DNS to LB IP
- [ ] Run smoke tests
- [ ] Configure alerting rules
- [ ] Document runbooks

---

## Elite Best Practices Summary

✅ **Immutability**
- Pinned image versions
- Read-only root filesystems
- Configuration via environment

✅ **Security**
- Non-root users
- Secrets management
- Network isolation
- No-new-privileges

✅ **Reliability**
- Health checks on all services
- Resource limits
- Graceful shutdowns
- Automated restarts

✅ **Observability**
- Structured logging
- Metrics collection
- Distributed tracing
- Labels for discovery

✅ **Backup Strategy**
- Automated GCS backups
- Retention policies
- Point-in-time recovery
- Disaster recovery tested

✅ **Developer Experience**
- Environment templates
- One-command deployment
- Clear documentation
- Reproducible builds

---

**Next Step**: Review this architecture, then I'll generate the complete `docker-compose.elite.yml` with all enhancements applied.
