# 🚀 Quick Reference Card - Ollama Elite AI Platform

This is a legacy compatibility snapshot. Use [Shared Documentation Navigation](../shared/README.md) for current documentation entry points, [On-Prem Execution Index](ON_PREM_EXECUTION_INDEX.md) for target-server-local navigation, and [On-Prem Deployment Model](ON_PREM_DEPLOYMENT_MODEL.md) for workflow details.

**Status**: 🟢 **LIVE** | **Project**: elevatediq | **Date**: January 13, 2026

---

## Common Commands

| Action | Command |
|--------|---------|
| Start services | `docker-compose -f docker/docker-compose.local.yml up -d` |
| Stop services | `docker-compose -f docker/docker-compose.local.yml down` |
| Run tests | `pytest tests/ -v --cov=ollama --cov-report=term-missing` |
| Check health | `curl http://localhost:8000/health` |

## Documentation

- [docs/indexed/README.md](../indexed/README.md) - find legacy indexes
- [docs/operations/ON_PREM_DEPLOYMENT_MODEL.md](ON_PREM_DEPLOYMENT_MODEL.md) - target-server-local workflow details and validation
- [docs/shared/README.md](../shared/README.md) - current documentation entry points

**Documentation**: See [docs/indexed/README.md](../indexed/README.md)

**Logs**: https://console.cloud.google.com/logs?project=elevatediq

**Created By**: GitHub Copilot
**Project**: Ollama Elite AI Platform

For complete information, see: [docs/indexed/README.md](../indexed/README.md)
