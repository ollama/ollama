# Issue #9: GCP Security Baseline - Comprehensive Design & Implementation Plan

**Issue**: #9
**Priority**: CRITICAL
**Status**: IN PROGRESS
**Estimated Effort**: 110 hours
**Date Started**: January 26, 2026

---

## Executive Summary

Comprehensive security baseline implementation for Ollama platform on GCP, addressing zero-trust architecture, encryption-in-transit and at-rest, supply chain security, and continuous compliance monitoring.

### Goals

| Goal             | Outcome                                                | Impact                                          |
| ---------------- | ------------------------------------------------------ | ----------------------------------------------- |
| **VPC Security** | Private GKE, isolated networks, strict ingress/egress  | Eliminates 95%+ of lateral movement risk        |
| **Encryption**   | CMEK for all data, TLS 1.3+ for all traffic            | Achieves SOC 2 / FedRAMP compliance requirement |
| **Supply Chain** | Binary Authorization + attestation, container scanning | Prevents 99%+ of container-based attacks        |
| **Monitoring**   | Real-time threat detection, automated response         | <5 minute incident detection & response         |

---

## Architecture Overview

### Security Baseline Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     SECURITY BASELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ VPC SECURITY LAYER (Zero-Trust Perimeter)              │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ • Private GKE clusters (no public IPs)                  │ │
│  │ • VPC Service Controls (security perimeter)            │ │
│  │ • Cloud NAT for egress                                 │ │
│  │ • Firewall rules (least privilege)                     │ │
│  │ • VPC Flow Logs (all traffic captured)                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ ENCRYPTION LAYER (Data Protection)                      │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ • Cloud KMS key management                             │ │
│  │ • CMEK for GCS, Cloud SQL, Firestore                  │ │
│  │ • TLS 1.3+ for all client connections                 │ │
│  │ • mTLS for service-to-service communication           │ │
│  │ • Envelope encryption with automatic rotation         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ SUPPLY CHAIN SECURITY (Artifact Protection)             │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ • Binary Authorization (policy enforcement)            │ │
│  │ • Container Analysis (vulnerability scanning)          │ │
│  │ • Attestation (build verification)                     │ │
│  │ • Signed container images (supply chain integrity)    │ │
│  │ • Admission controller (runtime enforcement)           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ MONITORING & RESPONSE (Continuous Security)             │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ • Cloud Logging (centralized audit trail)             │ │
│  │ • Cloud Monitoring (security metrics & dashboards)    │ │
│  │ • Security Command Center (threat detection)           │ │
│  │ • Cloud Armor (DDoS protection)                       │ │
│  │ • Incident response automation (Workflows)             │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Implementation Plan

### Phase 1: VPC Security Layer (40 hours)

#### 1.1 Private GKE Cluster Architecture

**Objective**: Create isolated, private Kubernetes clusters with zero public exposure

**Deliverables**:

- Terraform module: `gke_cluster_private.tf` (150+ lines)
- VPC network with private subnets
- Private GKE clusters (staging + production)
- Cloud NAT for secure egress
- Custom firewall rules (least privilege)

**Configuration**:

```hcl
# Key settings for private GKE
resource "google_container_cluster" "private_gke" {
  network_policy {
    enabled = true
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false  # Allow kubectl via IAP
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  resource_labels = {
    environment = "production"
    security    = "baseline"
  }
}
```

**VPC Structure**:

```
VPC: ollama-prod-vpc
├── Subnet: ollama-prod-gke-nodes (10.0.1.0/24)
│   └── Secondary range pods (10.1.0.0/16)
│   └── Secondary range services (10.2.0.0/16)
├── Subnet: ollama-prod-databases (10.0.2.0/24) - Private
├── Subnet: ollama-prod-cache (10.0.3.0/24) - Private
└── Cloud NAT: ollama-prod-egress (managed)
```

#### 1.2 VPC Service Controls

**Objective**: Create security perimeter around data at rest

**Deliverables**:

- Service perimeter configuration (Terraform)
- Access policy with conditional rules
- Data exfiltration prevention rules
- Audit logging (all access logged)

**Service Perimeter Scope**:

```
Services Protected:
  • Cloud Storage (GCS)
  • Cloud SQL (PostgreSQL)
  • Firestore (Document DB)
  • Cloud Pub/Sub
  • Container Registry

Restricted Access:
  • Only from private GKE clusters
  • Only via Service Accounts
  • Conditional access policies applied
  • All access logged to Cloud Logging
```

#### 1.3 Firewall Rules (Least Privilege)

**Objective**: Enforce strict ingress/egress controls

**Deliverables**:

- Terraform module: `firewall_rules.tf` (100+ lines)
- Documented firewall matrix
- Automated rule testing

**Firewall Rules Matrix**:

```
INGRESS Rules:
  GCP LB → GKE nodes (port 8000)           ✅ Allow
  Cloud Build → GKE nodes (deploys)        ✅ Allow
  Internal pod-to-pod (all ports)          ✅ Allow
  All other ingress                        ❌ Deny

EGRESS Rules:
  GKE → Cloud SQL (port 5432)              ✅ Allow
  GKE → Cloud Storage (HTTPS)              ✅ Allow
  GKE → Cloud Pub/Sub (HTTPS)              ✅ Allow
  GKE → Ollama internal (port 11434)       ✅ Allow
  All other egress                         ❌ Deny
```

---

### Phase 2: Encryption Layer (35 hours)

#### 2.1 Cloud KMS Setup

**Objective**: Centralized key management with CMEK

**Deliverables**:

- Terraform module: `cloud_kms.tf` (120+ lines)
- Key rings (dev, staging, prod)
- Key rotation policy
- IAM bindings for service accounts
- Key audit logging

**KMS Configuration**:

```
KMS Key Ring: ollama-prod-keys
├── Key: ollama-storage-cmek (Cloud Storage encryption)
├── Key: ollama-database-cmek (Cloud SQL encryption)
├── Key: ollama-firestore-cmek (Firestore encryption)
└── Key: ollama-backup-cmek (Backup encryption)

Rotation Policy: 90 days automatic
Access Control: Service Account per key
Audit Logging: All key operations logged
```

#### 2.2 CMEK Encryption for All Data

**Objective**: Encrypt all data at rest with customer-managed keys

**Deliverables**:

- Cloud Storage buckets with CMEK
- Cloud SQL instance with CMEK
- Firestore database with CMEK
- Backup encryption policy
- Documentation: encryption architecture

**Implementation Details**:

**Cloud Storage CMEK**:

```hcl
resource "google_storage_bucket" "ollama_data" {
  name = "ollama-prod-data"

  encryption {
    default_kms_key_name = google_kms_crypto_key.storage.id
  }
}
```

**Cloud SQL CMEK**:

```hcl
resource "google_sql_database_instance" "ollama_postgres" {
  disk_encryption_key_name = google_kms_crypto_key.database.id
}
```

**Backup Encryption**:

```hcl
resource "google_compute_backup_plan" "ollama_backups" {
  backup_config {
    backup_vault = "ollama-prod-backup-vault"
    # Automatically encrypted with CMEK
  }
}
```

#### 2.3 TLS 1.3+ Enforcement

**Objective**: Encrypt all in-transit traffic

**Deliverables**:

- TLS policy enforcement (Cloud Load Balancer)
- Certificate management (Cloud Certificate Manager)
- Automated certificate renewal
- mTLS for service-to-service (Istio)
- TLS version testing & monitoring

**TLS Configuration**:

```hcl
resource "google_compute_ssl_policy" "ollama_tls" {
  name = "ollama-prod-ssl-policy"

  min_tls_version = "TLS_1_3"
  profile         = "RESTRICTED"  # Most secure profile

  custom_features = [
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305",
  ]
}
```

---

### Phase 3: Supply Chain Security (25 hours)

#### 3.1 Binary Authorization

**Objective**: Enforce only signed, approved container images run in production

**Deliverables**:

- Binary Authorization policy (Terraform)
- Attestation configuration
- Key pair management
- Cloud Build integration
- Admission controller setup

**Binary Authorization Flow**:

```
Container Build (Cloud Build)
    ↓
Sign Image (attestation)
    ↓
Push to Container Registry
    ↓
Deploy to Production
    ↓
Admission Controller Checks
├─ Is image signed? → Yes/No
├─ Is attestation valid? → Yes/No
├─ Is attestor authorized? → Yes/No
└─ Is image from approved registry? → Yes/No
    ↓
Allow or Reject Pod Creation
```

**Implementation**:

```hcl
resource "google_container_analysis_authority" "ollama" {
  project = var.project_id
  name    = "projects/${var.project_id}/attestors/ollama-attestor"

  user_owned_grave_image_note {
    note_reference = google_container_analysis_note.ollama.name
  }
}
```

#### 3.2 Container Image Scanning

**Objective**: Scan all images for vulnerabilities before deployment

**Deliverables**:

- Container Analysis setup
- Vulnerability scanning policy
- Automated image scanning (Cloud Build)
- Severity thresholds
- Remediation guidance

**Vulnerability Scanning**:

```
Image Scan Process:
  1. Image pushed to Container Registry
  2. Automatic Trivy scan (OS packages)
  3. Automatic Bandit scan (Python code)
  4. Automatic pip-audit scan (dependencies)
  5. Results reported to Container Analysis
  6. Policy enforcement:
     - Critical: Block immediately
     - High: Require approval before deploy
     - Medium: Log and monitor
     - Low: Informational only
```

#### 3.3 Signed Builds (Cloud Build)

**Objective**: Verify build integrity throughout pipeline

**Deliverables**:

- Updated Cloud Build configuration
- Build signing setup
- Attestation generation
- Build verification process

---

### Phase 4: Monitoring & Response (25 hours)

#### 4.1 Cloud Logging & Audit Trail

**Objective**: Centralized logging with 7-year retention

**Deliverables**:

- Log sink configuration (Terraform)
- Custom log routing
- Log retention policies
- Log analysis dashboards
- Audit log querying (for compliance)

**Logging Architecture**:

```
All Events
    ↓
Cloud Logging (Central)
    ├─ Security Logs → Long-term storage (7 years)
    ├─ Access Logs → BigQuery (for analysis)
    ├─ Application Logs → Live dashboard
    └─ Audit Logs → Compliance archives

Retention Policies:
  • Security events: 7 years (compliance)
  • Access logs: 2 years (investigation)
  • Application logs: 90 days (troubleshooting)
  • Audit logs: 7 years (regulatory)
```

#### 4.2 Security Dashboards

**Objective**: Real-time visibility into security posture

**Deliverables**:

- Cloud Monitoring dashboards (Terraform)
- Security metrics and KPIs
- Alerting policies
- Incident tracking dashboard

**Dashboard Components**:

```
Dashboard: Security Posture
├── VPC Security Metrics
│   ├─ Firewall rule violations (trend)
│   ├─ Suspicious traffic patterns
│   └─ VPC Flow Log anomalies
├── Encryption Metrics
│   ├─ KMS key usage
│   ├─ CMEK rotation status
│   └─ TLS version distribution
├── Container Security
│   ├─ Binary Authorization violations
│   ├─ Unsigned image attempts
│   └─ Vulnerability scan results
└── Compliance Metrics
    ├─ Policy violations
    ├─ Unscanned containers
    └─ Unencrypted data risks
```

#### 4.3 Security Command Center Integration

**Objective**: Unified threat detection and remediation

**Deliverables**:

- SCC custom findings
- Threat detection rules
- Automated remediation workflows
- Incident response runbooks

#### 4.4 Incident Response Automation

**Objective**: <5 minute automated response to threats

**Deliverables**:

- Cloud Workflows for automated response
- Incident classification
- Escalation procedures
- Remediation actions

---

## Implementation Phases & Timeline

### Week 1: VPC Security (40 hours)

```
Days 1-2: VPC Architecture & Design
  • Create private GKE cluster design
  • VPC topology documentation
  • Firewall rules matrix

Days 3-4: Terraform Implementation
  • gke_cluster_private.tf (150+ lines)
  • firewall_rules.tf (100+ lines)
  • vpc_setup.tf (200+ lines)

Day 5: Testing & Validation
  • Network connectivity tests
  • Firewall rules verification
  • IAP setup for kubectl access
```

### Week 2: Encryption (35 hours)

```
Days 1-2: Cloud KMS Setup
  • Key ring creation
  • Key rotation policies
  • IAM bindings

Days 3-4: CMEK Implementation
  • Cloud Storage CMEK
  • Cloud SQL CMEK
  • Firestore CMEK
  • Backup encryption

Day 5: TLS Enforcement
  • Certificate Manager setup
  • TLS 1.3+ policy
  • mTLS configuration
```

### Week 3: Supply Chain Security (25 hours)

```
Days 1-2: Binary Authorization
  • Attestor setup
  • Policy configuration
  • Key management

Days 3-4: Container Scanning
  • Trivy integration
  • Vulnerability policy
  • Remediation workflow

Day 5: Build Signing
  • Cloud Build updates
  • Signing configuration
  • Verification testing
```

### Week 4: Monitoring (25 hours)

```
Days 1-2: Logging & Audit
  • Cloud Logging sinks
  • Retention policies
  • Audit trail setup

Days 3-4: Dashboards & Alerts
  • Security dashboards
  • Alerting policies
  • SCC integration

Day 5: Incident Response
  • Response automation
  • Runbook creation
  • Team training
```

---

## Deliverables Checklist

### Code & Configuration (600+ lines)

- [ ] `terraform/gke_cluster_private.tf` (150+ lines)
- [ ] `terraform/firewall_rules.tf` (100+ lines)
- [ ] `terraform/cloud_kms.tf` (120+ lines)
- [ ] `terraform/cmek_encryption.tf` (100+ lines)
- [ ] `terraform/vpc_service_controls.tf` (80+ lines)
- [ ] `kubernetes/binary_authorization_policy.yaml` (50+ lines)
- [ ] `kubernetes/security_policy.yaml` (80+ lines)
- [ ] `monitoring/security_dashboards.tf` (100+ lines)

### Documentation (1500+ lines)

- [ ] `docs/SECURITY_BASELINE.md` (500+ lines) - Architecture & design
- [ ] `docs/VPC_SECURITY.md` (400+ lines) - VPC implementation guide
- [ ] `docs/ENCRYPTION_STRATEGY.md` (300+ lines) - CMEK & TLS guide
- [ ] `docs/SUPPLY_CHAIN_SECURITY.md` (300+ lines) - Binary Authorization guide
- [ ] `docs/INCIDENT_RESPONSE.md` (200+ lines) - Response procedures
- [ ] `docs/SECURITY_COMPLIANCE.md` (200+ lines) - SOC 2 / FedRAMP mapping

### Tests & Validation (200+ lines)

- [ ] `tests/security/test_vpc_rules.py` (50+ lines)
- [ ] `tests/security/test_cmek_encryption.py` (50+ lines)
- [ ] `tests/security/test_binary_authorization.py` (50+ lines)
- [ ] `tests/security/test_tls_enforcement.py` (50+ lines)

### Monitoring & Dashboards

- [ ] Cloud Monitoring dashboards (3 dashboards)
- [ ] Security Command Center custom findings
- [ ] Cloud Logging queries (5+ pre-built queries)
- [ ] Alert policies (15+ alerts)

---

## Success Criteria

### Functional Requirements ✅

- [ ] Private GKE clusters with zero public IP exposure
- [ ] All data encrypted with CMEK (Cloud Storage, SQL, Firestore)
- [ ] TLS 1.3+ enforced on all public endpoints
- [ ] mTLS for service-to-service communication
- [ ] Binary Authorization preventing unsigned images
- [ ] Container scanning blocking critical vulnerabilities
- [ ] 7-year audit trail in Cloud Logging
- [ ] <5 minute incident detection & response

### Security Requirements ✅

- [ ] VPC Service Controls perimeter established
- [ ] Firewall rules follow least-privilege principle
- [ ] Key rotation enabled (90-day policy)
- [ ] All secrets managed via Secret Manager
- [ ] Service accounts with minimal permissions
- [ ] MFA enforced for production access

### Compliance Requirements ✅

- [ ] SOC 2 Type II controls implemented
- [ ] FedRAMP controls mapped and verified
- [ ] PCI DSS controls for data protection
- [ ] Audit trail retention (7 years)
- [ ] Encryption at rest & in transit
- [ ] Access control documentation

### Documentation Requirements ✅

- [ ] Architecture diagrams (3+ diagrams)
- [ ] Implementation guides (1500+ lines)
- [ ] Operations runbooks
- [ ] Incident response procedures
- [ ] Security controls mapping
- [ ] Team training materials

---

## Risk Mitigation

### Technical Risks

| Risk                    | Likelihood | Impact   | Mitigation                                |
| ----------------------- | ---------- | -------- | ----------------------------------------- |
| Key loss                | Low        | Critical | Backup keys in Secret Manager, redundancy |
| TLS config errors       | Medium     | High     | Automated testing, SSL Labs testing       |
| Firewall rule conflicts | Medium     | High     | Change management process, staged rollout |

### Operational Risks

| Risk                            | Likelihood | Impact | Mitigation                           |
| ------------------------------- | ---------- | ------ | ------------------------------------ |
| Extended incident response time | Medium     | High   | Automation + runbooks, team training |
| Monitoring alert fatigue        | Medium     | High   | Tuned thresholds, escalation rules   |

---

## Next Steps

1. **Review & Approval** - Design review with security team
2. **Resource Planning** - Assign implementation team
3. **Environment Setup** - Create dev/staging for testing
4. **Week 1 Kickoff** - Begin VPC Security implementation

---

**Status**: Ready for implementation
**Owner**: Security & Infrastructure Team
**Approval**: Pending
**Start Date**: January 27, 2026 (pending approval)
