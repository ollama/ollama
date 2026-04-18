# Kustomize Deployment Guide

Using Kustomize for environment-specific Kubernetes deployments.

## Directory Structure

```
k8s/
├── kustomization.yaml          # Base configuration
├── 0-namespace-and-services.yaml
├── 1-databases.yaml
├── 2-api.yaml
├── 3-monitoring.yaml
├── 4-ingress.yaml
└── overlays/
    ├── dev/
    │   └── kustomization.yaml   # Development overrides
    ├── staging/
    │   ├── kustomization.yaml   # Staging overrides
    │   └── secrets.env          # Staging secrets
    └── prod/
        ├── kustomization.yaml   # Production overrides
        └── secrets.env          # Production secrets
```

## Deployment Commands

### Development Environment

```bash
# Preview changes
kubectl kustomize k8s/overlays/dev

# Apply development configuration
kubectl apply -k k8s/overlays/dev

# Watch deployment
kubectl get pods -n ollama-dev -w

# Verify
kubectl get all -n ollama-dev
```

### Staging Environment

```bash
# Create staging secrets file
cat > k8s/overlays/staging/secrets.env <<EOF
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
QDRANT_API_KEY=$(openssl rand -hex 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)
EOF

# Apply staging configuration
kubectl apply -k k8s/overlays/staging

# Watch deployment
kubectl get pods -n ollama-staging -w
```

### Production Environment

```bash
# Create production secrets file
cat > k8s/overlays/prod/secrets.env <<EOF
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
QDRANT_API_KEY=$(openssl rand -hex 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)
EOF

# Store secrets securely (use external secret management)
# kubectl apply -k k8s/overlays/prod

# For now, preview the production manifest
kubectl kustomize k8s/overlays/prod | head -100
```

## Configuration Differences

### Development (1 replica, low resources)
- API replicas: 1
- HPA: 1-2 max
- Memory: 128Mi request → 256Mi limit
- Log level: DEBUG
- Cache TTL: 300s

### Staging (2 replicas, medium resources)
- API replicas: 2
- HPA: 2-5 max
- Memory: 256Mi request → 512Mi limit
- Log level: INFO
- Cache TTL: 1800s

### Production (5 replicas, high resources)
- API replicas: 5
- HPA: 3-20 max
- Memory: 512Mi request → 1Gi limit
- Log level: WARN
- Cache TTL: 3600s
- PDB minAvailable: 2
- Tracing: enabled
- Metrics: enabled

## Using Kustomize Features

### Build and Save Manifests

```bash
# Build dev manifests
kubectl kustomize k8s/overlays/dev > manifests-dev.yaml

# Build staging manifests
kubectl kustomize k8s/overlays/staging > manifests-staging.yaml

# Build production manifests
kubectl kustomize k8s/overlays/prod > manifests-prod.yaml

# Review before applying
cat manifests-prod.yaml | head -50
```

### Update Configurations

```bash
# Change log level in dev
cd k8s/overlays/dev
# Edit kustomization.yaml
# Change: LOG_LEVEL=debug -> LOG_LEVEL=trace

# Apply changes
kubectl apply -k .

# Verify ConfigMap updated
kubectl get configmap ollama-config -n ollama-dev -o yaml | grep LOG_LEVEL
```

### Scale Replicas

```bash
# Temporarily scale for testing
kubectl scale deployment ollama-api --replicas=10 -n ollama-dev

# Or update in kustomization.yaml
# replicas:
#   - name: ollama-api
#     count: 10

# Apply changes
kubectl apply -k k8s/overlays/dev
```

## GitOps Workflow

### Using ArgoCD

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Create ArgoCD Application for dev
kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ollama-dev
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/elevatediq/ollama
    targetRevision: HEAD
    path: k8s/overlays/dev
  destination:
    server: https://kubernetes.default.svc
    namespace: ollama-dev
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
EOF

# Access ArgoCD UI
kubectl port-forward -n argocd svc/argocd-server 8080:443
# https://localhost:8080 (admin/password)
```

### Using Flux

```bash
# Install Flux
curl -s https://fluxcd.io/install.sh | sudo bash

# Initialize Flux
flux bootstrap github \
  --owner=elevatediq \
  --repo=ollama \
  --branch=main \
  --path=flux

# Create Kustomization for dev
flux create kustomization ollama-dev \
  --source=GitRepository/ollama \
  --path=./k8s/overlays/dev \
  --prune=true \
  --interval=1m

# Watch reconciliation
flux get kustomization --watch
```

## Validation

### Pre-deployment Checks

```bash
# Validate base manifests
kubectl apply -k k8s --dry-run=client

# Validate development overlay
kubectl apply -k k8s/overlays/dev --dry-run=client

# Validate staging overlay
kubectl apply -k k8s/overlays/staging --dry-run=client

# Validate production overlay
kubectl apply -k k8s/overlays/prod --dry-run=client
```

### Lint Manifests

```bash
# Using kubeval
kubeval <(kubectl kustomize k8s/overlays/dev)

# Using kubesec
kubesec scan <(kubectl kustomize k8s/overlays/prod)

# Using kube-linter
kube-linter lint <(kubectl kustomize k8s/overlays/prod)
```

## Common Operations

### Update Image Tags

```bash
# In kustomization.yaml
images:
  - name: ollama-api
    newTag: "v1.2.0"

# Apply changes
kubectl apply -k k8s/overlays/prod
```

### Update Secrets

```bash
# Regenerate secrets
kubectl delete secret ollama-secrets -n ollama

# Create new secrets
kubectl create secret generic ollama-secrets \
  --from-env-file=k8s/overlays/prod/secrets.env \
  -n ollama

# Restart pods to pick up new secrets
kubectl rollout restart deployment/ollama-api -n ollama
```

### Add New ConfigMap Entry

```bash
# Edit kustomization.yaml
configMapGenerator:
  - name: ollama-config
    behavior: merge
    literals:
      - NEW_VAR=value

# Apply
kubectl apply -k k8s/overlays/dev
```

## Troubleshooting

### Manifest Generation Issues

```bash
# Debug kustomize build
kubectl kustomize k8s/overlays/dev --enable-alpha-plugins

# Verbose output
kustomize build k8s/overlays/dev -o=json | jq .
```

### Secret Issues

```bash
# View generated secret (base64)
kubectl kustomize k8s/overlays/dev | grep -A 20 "kind: Secret"

# Decode secret
kubectl get secret ollama-secrets -o yaml | grep DATABASE_PASSWORD | awk '{print $2}' | base64 -d
```

### ConfigMap Issues

```bash
# View generated ConfigMap
kubectl kustomize k8s/overlays/dev | grep -A 20 "kind: ConfigMap"

# Compare configurations
kubectl kustomize k8s/overlays/dev > dev.yaml
kubectl kustomize k8s/overlays/prod > prod.yaml
diff dev.yaml prod.yaml | head -50
```

## Best Practices

1. **Never commit secrets to git**
   - Use external secret management (Sealed Secrets, External Secrets)
   - Or create secrets.env and add to .gitignore

2. **Use ConfigMaps for configuration**
   - Don't hardcode values in manifests
   - Override per environment

3. **Validate before applying**
   - Use `--dry-run=client`
   - Use kubeval/kubesec for linting

4. **Keep overlays simple**
   - Use patches for small changes
   - Don't duplicate entire manifests

5. **Document custom patches**
   - Add comments explaining why patches exist
   - Keep overlays maintainable

6. **Test locally first**
   - Deploy to local cluster
   - Verify with `kubectl get all`
   - Test endpoints

---

**Version**: 1.0.0
**Last Updated**: 2024
