# Issue #42 Design: Kubernetes Hub Support

**Date:** April 18, 2026
**Phase:** 2 - Design & Planning
**Status:** In Progress

---

## Design Document

### 1. Architecture Overview

```
┌─────────────────────────────────────────┐
│     Ollama Client/API                   │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  Kubernetes Integration Layer            │
│  ├─ KubernetesProvider                  │
│  ├─ DeploymentController                │
│  └─ ServiceManager                      │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  Kubernetes Go Client                    │
│  (kubernetes/client-go)                 │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  Target Kubernetes Cluster               │
│  (1.24+)                                │
└─────────────────────────────────────────┘
```

### 2. Core Components

#### 2.1 KubernetesProvider
```go
type KubernetesProvider struct {
    clientset    *kubernetes.Clientset
    config       *rest.Config
    namespace    string
    storageClass string
}

// Key methods:
- Connect(kubeconfig string) error
- Disconnect() error
- IsAvailable() bool
```

#### 2.2 DeploymentController
```go
type DeploymentController struct {
    provider *KubernetesProvider
    store    *ModelStore
}

// Key methods:
- Deploy(modelName, version string) error
- Undeploy(modelName string) error
- GetStatus(modelName string) *DeploymentStatus
- Scale(modelName string, replicas int) error
```

#### 2.3 ServiceManager
```go
type ServiceManager struct {
    provider *KubernetesProvider
}

// Key methods:
- CreateService(name string, selector map[string]string) (*v1.Service, error)
- UpdateService(name string, svc *v1.Service) error
- DeleteService(name string) error
- GetEndpoints(name string) (*v1.Endpoints, error)
```

### 3. API Endpoints

**New endpoints for Kubernetes operations:**

```
POST   /api/kubernetes/deploy
GET    /api/kubernetes/status/{model}
DELETE /api/kubernetes/{model}
POST   /api/kubernetes/{model}/scale
GET    /api/kubernetes/clusters
```

**Request/Response Contracts:**

```json
// Deploy request
{
  "model": "llama2",
  "version": "latest",
  "replicas": 3,
  "resources": {
    "cpu": "2",
    "memory": "8Gi",
    "gpu": "1"
  }
}

// Status response
{
  "model": "llama2",
  "state": "running",
  "replicas": 3,
  "ready_replicas": 3,
  "service_endpoint": "llama2.default.svc.cluster.local:11434",
  "created_at": "2026-04-18T10:30:00Z",
  "last_updated": "2026-04-18T10:35:00Z"
}
```

### 4. File Structure

```
ollama/
├── kubernetes/          # New package
│   ├── client.go       # K8s client wrapper
│   ├── deployment.go   # Deployment logic
│   ├── service.go      # Service management
│   ├── storage.go      # Storage provisioning
│   ├── status.go       # Status tracking
│   ├── errors.go       # Error types
│   └── kubernetes_test.go
├── llm/
│   ├── kubernetes.go   # Integration with LLM layer (NEW)
│   └── llm.go          # Existing
├── api/
│   └── kubernetes.go   # API handlers (NEW)
└── server/
    └── routes.go       # Register new routes
```

### 5. Data Models

**Kubernetes Manifest:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-{model}
  namespace: ollama-models
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ollama
      model: {model}
  template:
    metadata:
      labels:
        app: ollama
        model: {model}
    spec:
      containers:
      - name: ollama
        image: ollama:latest
        ports:
        - containerPort: 11434
        resources:
          requests:
            cpu: 2
            memory: 8Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 4
            memory: 16Gi
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: ollama-{model}-pvc
```

### 6. Configuration Requirements

**Environment Variables:**
```
OLLAMA_K8S_ENABLED=true
OLLAMA_K8S_KUBECONFIG=/path/to/kubeconfig
OLLAMA_K8S_NAMESPACE=ollama-models
OLLAMA_K8S_STORAGE_CLASS=fast-ssd
OLLAMA_K8S_REGISTRY=docker.io
```

### 7. Implementation Plan

**Phase 2a: Core Client (Week 1)**
- [ ] Create kubernetes package
- [ ] Implement KubernetesProvider
- [ ] Implement basic connectivity tests
- [ ] Add error handling

**Phase 2b: Deployment Logic (Week 1-2)**
- [ ] Implement DeploymentController
- [ ] Create manifest generation
- [ ] Implement status tracking
- [ ] Add health checks

**Phase 2c: API Integration (Week 2)**
- [ ] Create API handlers
- [ ] Register endpoints
- [ ] Add request validation
- [ ] Implement response formatting

**Phase 2d: Testing & Validation (Week 2-3)**
- [ ] Unit tests (90%+ coverage)
- [ ] Integration tests with kind
- [ ] E2E tests
- [ ] Performance benchmarks

**Phase 2e: Documentation (Week 3)**
- [ ] API documentation
- [ ] Deployment guide
- [ ] Troubleshooting guide
- [ ] Examples

### 8. Dependencies to Add

```go
// go.mod additions
require (
    k8s.io/client-go v0.28.0
    k8s.io/api v0.28.0
    k8s.io/apimachinery v0.28.0
)
```

### 9. Testing Strategy

**Unit Tests:**
- Client initialization under various conditions
- Manifest generation correctness
- Error handling and retries
- Status tracking logic

**Integration Tests:**
- Actual Kubernetes deployment with kind
- Service creation and discovery
- Scaling operations
- Failure scenarios

**E2E Tests:**
- Full deployment workflow
- Multi-node scenarios
- Network issues handling
- Resource constraint handling

### 10. Quality Gates

```
✓ 95%+ code coverage
✓ All tests passing
✓ No linting errors
✓ No type check errors
✓ Documentation complete
✓ API contract verified
✓ Performance benchmarks met
✓ Security review passed
```

---

**Design Status:** ✅ COMPLETE

**Next Phase:** Phase 3 - Branch Creation

**Approval:** Ready to implement
