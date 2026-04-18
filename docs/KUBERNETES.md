# Kubernetes Deployment Guide

## Overview
Complete Kubernetes setup for production deployment with auto-scaling, monitoring, and high availability.

## Prerequisites

- Kubernetes cluster 1.24+
- kubectl configured
- Helm 3.0+
- cert-manager (for SSL/TLS)
- NGINX Ingress Controller
- StorageClass configured (fast-ssd)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   NGINX Ingress                         │
│                  (Load Balancer)                        │
└────────────┬────────────────────────────────────────────┘
             │
         HTTPS/TLS
             │
    ┌────────┴────────┐
    │   Ollama API    │
    │  (3-10 replicas)│  ← HPA Auto-scaling
    │  Load Balanced  │
    └─┬──┬──┬─────┬──┬┘
      │  │  │     │
      ├──┴──┼─────┤
      │     │     │
  ┌───▼──┐ │  ┌──▼────┐
  │Postgres│ │  │Redis  │
  │(1 replica)  │(1 replica)
  └────────┘    └───────┘
      ┌─────────────┐
      │   Qdrant    │
      │(1 replica)  │
      └─────────────┘
  ┌──────────────────────┐
  │ Monitoring Stack     │
  │ ├─ Prometheus        │
  │ ├─ Grafana           │
  │ └─ Jaeger            │
  └──────────────────────┘
```

## Deployment Steps

### 1. Setup Prerequisites

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --create-namespace --namespace ingress-nginx

# Create StorageClass for fast SSD
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # or your cloud provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
reclaimPolicy: Delete
allowVolumeExpansion: true
EOF
```

### 2. Create Secrets

```bash
# Generate secret values
SECRET_KEY=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
QDRANT_API_KEY=$(openssl rand -hex 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)

# Create secret
kubectl create secret generic ollama-secrets \
  --from-literal=SECRET_KEY="$SECRET_KEY" \
  --from-literal=DATABASE_PASSWORD="$DB_PASSWORD" \
  --from-literal=REDIS_PASSWORD="$REDIS_PASSWORD" \
  --from-literal=QDRANT_API_KEY="$QDRANT_API_KEY" \
  --from-literal=GRAFANA_PASSWORD="$GRAFANA_PASSWORD" \
  -n ollama

# Save secrets to safe location
cat > /secure/location/secrets.env <<EOF
SECRET_KEY=$SECRET_KEY
DATABASE_PASSWORD=$DB_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
QDRANT_API_KEY=$QDRANT_API_KEY
GRAFANA_PASSWORD=$GRAFANA_PASSWORD
EOF
chmod 600 /secure/location/secrets.env
```

### 3. Deploy Infrastructure

```bash
# Deploy in order
kubectl apply -f k8s/0-namespace-and-services.yaml
kubectl apply -f k8s/1-databases.yaml

# Wait for databases to be ready
kubectl wait --for=condition=Ready pod \
  -l app=postgres -n ollama --timeout=300s
kubectl wait --for=condition=Ready pod \
  -l app=redis -n ollama --timeout=300s
kubectl wait --for=condition=Ready pod \
  -l app=qdrant -n ollama --timeout=300s

# Deploy API
kubectl apply -f k8s/2-api.yaml

# Deploy monitoring
kubectl apply -f k8s/3-monitoring.yaml

# Deploy ingress
kubectl apply -f k8s/4-ingress.yaml

# Wait for all pods
kubectl wait --for=condition=Ready pod \
  -l app=ollama-api -n ollama --timeout=600s
```

### 4. Initialize Database

```bash
# Port-forward to postgres
kubectl port-forward svc/postgres -n ollama 5432:5432 &

# Run migrations
PGPASSWORD=$DB_PASSWORD psql -h localhost -U ollama -d ollama \
  -f migrations/001_initial_schema.sql

# Create initial data
PGPASSWORD=$DB_PASSWORD psql -h localhost -U ollama -d ollama \
  -f migrations/002_seed_data.sql
```

## Verification

### Check Deployment Status

```bash
# View all resources
kubectl get all -n ollama

# Check pod status
kubectl get pods -n ollama -o wide

# View logs
kubectl logs -f deployment/ollama-api -n ollama

# Check events
kubectl get events -n ollama --sort-by='.lastTimestamp'
```

### Test Endpoints

```bash
# Get LoadBalancer IP
kubectl get svc ollama-api -n ollama

# Test API health
curl https://api.elevatediq.ai/health

# Test metrics
curl https://api.elevatediq.ai/metrics

# Access Grafana
# https://api.elevatediq.ai/grafana (admin/password)

# Access Jaeger
# https://api.elevatediq.ai/jaeger
```

## Scaling

### Manual Scaling

```bash
# Scale API to 5 replicas
kubectl scale deployment ollama-api --replicas=5 -n ollama

# Check HPA status
kubectl get hpa -n ollama
```

### Auto-Scaling Configuration

The HPA automatically scales based on:
- CPU utilization: 70%
- Memory utilization: 80%
- Min replicas: 2
- Max replicas: 10

Adjust in `k8s/2-api.yaml`:
```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      averageUtilization: 70  # Increase this for less aggressive scaling
```

## Monitoring

### Access Dashboards

```bash
# Prometheus
kubectl port-forward svc/prometheus -n ollama 9090:9090

# Grafana (admin/password from secrets)
kubectl port-forward svc/grafana -n ollama 3000:3000

# Jaeger
kubectl port-forward svc/jaeger -n ollama 16686:16686
```

### Key Metrics to Monitor

- API response time (p95, p99)
- Request rate (RPS)
- Error rate
- Database connection pool utilization
- Redis memory usage
- Pod restart count
- Node resource utilization

## Updates and Rollouts

### Update API Image

```bash
# Update image
kubectl set image deployment/ollama-api \
  ollama-api=ollama-api:v1.1.0 -n ollama

# Check rollout status
kubectl rollout status deployment/ollama-api -n ollama

# View rollout history
kubectl rollout history deployment/ollama-api -n ollama

# Rollback if needed
kubectl rollout undo deployment/ollama-api -n ollama
```

### Blue-Green Deployment

```bash
# Deploy new version in separate deployment
kubectl apply -f k8s/2-api-v2.yaml

# Switch traffic (update service selector)
kubectl patch service ollama-api -p \
  '{"spec":{"selector":{"version":"v2"}}}'

# Verify and cleanup old deployment
kubectl delete deployment ollama-api-v1 -n ollama
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
kubectl exec -it $(kubectl get pod -l app=postgres -n ollama -o jsonpath='{.items[0].metadata.name}') \
  -n ollama -- pg_dump -U ollama ollama > backup_$(date +%Y%m%d).sql

# Restore from backup
kubectl cp backup_20240101.sql \
  $(kubectl get pod -l app=postgres -n ollama -o jsonpath='{.items[0].metadata.name}'):/tmp/ \
  -n ollama

kubectl exec -it $(kubectl get pod -l app=postgres -n ollama -o jsonpath='{.items[0].metadata.name}') \
  -n ollama -- psql -U ollama ollama < /tmp/backup_20240101.sql
```

### PVC Snapshots

```bash
# Create snapshot (cloud-specific)
# AWS:
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: postgres-snapshot-$(date +%Y%m%d)
  namespace: ollama
spec:
  volumeSnapshotClassName: csi-ebs-vsc
  source:
    persistentVolumeClaimName: postgres-pvc
EOF
```

## Security

### Network Policies

Network policies are configured to:
- Restrict ingress only from ingress controller
- Allow API to Prometheus scraping
- Restrict egress to only required services
- Allow DNS queries

### RBAC Configuration

Service accounts and roles configured with minimal permissions:
- API service account can read ConfigMaps and Secrets
- Prometheus service account can list Kubernetes API resources

### Pod Security

- Non-root user (uid: 1000)
- Read-only root filesystem
- No privilege escalation
- Dropped all capabilities

## Troubleshooting

### Pod Stuck in Pending

```bash
# Check resource availability
kubectl describe nodes

# Check PVC status
kubectl get pvc -n ollama

# Check events
kubectl describe pod <pod-name> -n ollama
```

### High CPU/Memory Usage

```bash
# Check metrics
kubectl top nodes
kubectl top pods -n ollama

# Check resource limits
kubectl describe deployment ollama-api -n ollama

# Increase limits if needed
kubectl set resources deployment ollama-api \
  -n ollama --limits=memory=4Gi,cpu=2000m
```

### Database Connection Issues

```bash
# Test connection
kubectl run -it --rm debug --image=postgres:15 --restart=Never \
  -- psql -h postgres.ollama.svc -U ollama -d ollama -c "SELECT 1"

# Check logs
kubectl logs deployment/postgres -n ollama
```

### Ingress Not Working

```bash
# Check ingress status
kubectl describe ingress ollama-ingress -n ollama

# Check certificate status
kubectl get certificate -n ollama

# Check cert-manager logs
kubectl logs -f deployment/cert-manager -n cert-manager
```

## Cost Optimization

### Resource Limits

Adjust based on actual usage:
```yaml
resources:
  requests:
    memory: "256Mi"  # Reduce if not needed
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Storage Optimization

```bash
# Check disk usage
kubectl exec -it postgres-pod -n ollama -- du -sh /var/lib/postgresql/data

# Enable compression in PostgreSQL
kubectl exec postgres-pod -n ollama -- psql -U ollama -d ollama \
  -c "ALTER SYSTEM SET wal_compression = on;"
```

### Pod Disruption Budgets

Adjust `minAvailable` based on HA requirements:
```yaml
spec:
  minAvailable: 2  # Require at least 2 pods running
```

## Compliance and Audit

### Audit Logging

```bash
# Enable audit logging
kubectl apply -f k8s/audit-policy.yaml

# View audit logs
kubectl logs -n kube-system -l component=kube-apiserver
```

### Pod Security Standards

Configured to enforce:
- Restricted security profiles
- No privileged containers
- No host network access

## Support and Maintenance

- Monitor cluster capacity quarterly
- Update Kubernetes and dependencies monthly
- Review and optimize costs monthly
- Test disaster recovery quarterly
- Update security patches immediately

---

**Last Updated**: 2024
**Version**: 1.0.0
**Kubernetes**: 1.24+
