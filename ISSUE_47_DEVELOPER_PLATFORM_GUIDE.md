# Issue #47: Developer Self-Service Platform Implementation Guide

**Issue**: [#47 - Developer Self-Service Platform](https://github.com/kushin77/ollama/issues/47)  
**Status**: OPEN - Ready for Assignment  
**Priority**: MEDIUM  
**Estimated Hours**: 95h (13.6 days)  
**Timeline**: Week 2-4 (Feb 10-28, 2026)  
**Dependencies**: #42 (Federation), #43 (Security)  
**Parallel Work**: #44, #45, #46, #48, #49, #50  

## Overview

Implement **Backstage** developer portal with **10 golden paths**, service catalogs, tech debt tracking, and self-service deployment. Enable engineers to deploy, monitor, and manage services without manual intervention.

## Architecture

```
Developer Portal (Backstage) → Service Catalog → Golden Paths → Deployment Templates
                             → Tech Debt Tracker
                             → Cost Visibility
                             → On-Call Schedules
```

## Phase 1: Backstage Setup (Week 2, 35 hours)

### 1.1 Backstage Installation & Configuration
- Docker container setup
- PostgreSQL backend
- GitHub integration
- OIDC authentication

**Docker Compose** (80 lines):
```yaml
services:
  backstage:
    image: backstage:latest
    ports:
      - "3000:3000"
    environment:
      GITHUB_TOKEN: ${GITHUB_TOKEN}
      BACKSTAGE_BASE_URL: https://backstage.example.com
    depends_on:
      - postgres

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: backstage
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - backstage_data:/var/lib/postgresql/data
```

### 1.2 Service Catalog Integration
- Software component model
- Service ownership
- Dependencies graph
- Technology stack tagging

**Code** (300 lines - `backstage/catalog/service_catalog.py`):
```python
class ServiceCatalog:
    """Manages service metadata and catalog."""

    async def register_service(
        self,
        name: str,
        description: str,
        owner: str,
        repo_url: str,
        techs: list[str]
    ) -> Service:
        """Register new service in catalog."""
        service = Service(
            name=name,
            description=description,
            owner=owner,
            repo_url=repo_url,
            technologies=techs,
            created_at=datetime.now()
        )
        
        await self.db.services.insert(service)
        return service
```

### 1.3 GitHub Integration
- Auto-register services from CODEOWNERS
- Sync service metadata
- Link to PR/issues
- Deployment status sync

## Phase 2: Golden Paths (Week 2-3, 35 hours)

### 2.1 10 Golden Paths Defined

1. **Deploy Python Service**
2. **Deploy Node.js Service**
3. **Deploy Kubernetes Workload**
4. **Create Microservice**
5. **Set Up Monitoring**
6. **Configure Logging**
7. **Enable Cost Tracking**
8. **Set Up On-Call Schedule**
9. **Deploy to Multiple Regions**
10. **Set Up GitOps Pipeline**

**Golden Path Template** (200 lines each):
```yaml
# golden-path-deploy-python-service.yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: python-service
  title: Deploy Python Service
spec:
  owner: platform-team
  type: service
  parameters:
    - title: Service Details
      properties:
        name:
          type: string
          title: Service Name
        description:
          type: string
          title: Description
        framework:
          type: string
          enum: [fastapi, flask, django]
  steps:
    - id: template
      name: Create Repository
      action: fetch:template
      input:
        url: https://github.com/ollama/templates/python-service
        values:
          name: ${{ parameters.name }}
    - id: publish
      name: Publish to GitHub
      action: publish:github
      input:
        repo: ${{ parameters.name }}
    - id: register
      name: Register in Catalog
      action: catalog:register
      input:
        repoContentsUrl: ${{ steps.publish.output.repoContentsUrl }}
```

### 2.2 Template Customization
- Framework-specific templates
- Tech stack variations
- Region templates
- Compliance templates

### 2.3 Scaffolder Workflows
- Repository creation
- CI/CD setup
- Monitoring setup
- Documentation generation

## Phase 3: Tech Debt & Self-Service (Week 3-4, 25 hours)

### 3.1 Tech Debt Tracker
- Track known issues
- Dependency upgrade tracking
- Security vulnerability tracking
- Performance improvement backlog

**Code** (250 lines - `backstage/tech_debt/tracker.py`)

### 3.2 Cost Visibility
- Service cost breakdown
- Cost trends per service
- Cost allocation
- Budget tracking

### 3.3 On-Call Schedule Management
- PagerDuty integration
- Rotation schedule
- Escalation policies
- On-call handoff

## Acceptance Criteria

- [ ] Backstage deployed and accessible
- [ ] 10 golden paths implemented
- [ ] Service catalog populated (50+ services)
- [ ] GitHub integration working
- [ ] Self-service deployments working
- [ ] Tech debt dashboard visible
- [ ] Cost visibility showing
- [ ] On-call schedule integrated

## Testing Strategy

- Unit tests for catalog (15 tests)
- Golden path tests (20 tests)
- Integration tests (10 tests)
- E2E user workflows (8 tests)

## Success Metrics

- **Time to Deploy**: <5 minutes via golden path
- **Service Catalog Completeness**: 95%+
- **Self-Service Adoption**: 60%+ of teams by Week 4
- **Manual Deployment Reduction**: 40%+

---

**Next Steps**: Assign to platform engineering lead, begin Week 2
