# GCP Landing Zone Onboarding Guide

This guide provides the necessary steps for engineers to onboard into the infrastructure layer of the Ollama project, specifically the [GCP Landing Zone](https://github.com/kushin77/GCP-landing-zone).

## 🚀 Overview

The GCP Landing Zone is the foundation for all Ollama deployments. It is engineered to FAANG standards, ensuring simplicity, observability, and security (FedRAMP-aligned).

**Infrastructure Repository:** [https://github.com/kushin77/GCP-landing-zone](https://github.com/kushin77/GCP-landing-zone)

---

## 🛠 Prerequisites

### 1. Required Tools

Install the core infrastructure toolset:

```bash
# Core Cloud & Infrastructure tools
brew install terraform terraform-docs tflint checkov gcloud kubectl jq

# Recommended tooling
brew install direnv pre-commit
```

### 2. Git Discipline

All commits MUST be signed and follow the standard format:

```bash
# Configure GPG signing
git config --global commit.gpgsign true
git config --global user.signingkey <your-gpg-key-id>

# Commit format: type(scope): description
# Example: feat(networking): add subnet for inference-v2
```

---

## 🔐 GCP Access & Authentication

To interact with the infrastructure, you need proper authorization:

1. **Request access** from the platform team.
2. **Authenticate locally:**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
3. **Set your target project:**
   ```bash
   gcloud config set project gcp-eiq
   ```
4. **Verify Organization Access:**
   ```bash
   gcloud organizations list  # Should show org 266397081400
   ```

---

## 🏗 Repository Standards (The 5-Layer Mandate)

All infrastructure code must strictly follow the 5-layer filesystem hierarchy. No files deeper than 5 levels, and every level has a dedicated responsibility.

### Hierarchy Example:

1. **Level 1 (Root):** `GCP-landing-zone/`
2. **Level 2 (Domain):** `terraform/`
3. **Level 3 (Functional):** `01-organization/` or `02-networking/`
4. **Level 4 (Resource):** `orchestration/governance/global/`
5. **Level 5 (Implementation):** `main.tf`, `variables.tf`, `outputs.tf`

---

## 🏃‍♂️ First Change Workflow

When making your first infrastructure change:

1. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/add-vpc-subnet
   ```
2. **Setup Local Provider Mirror:**
   This speeds up `terraform init` significantly.
   ```bash
   ./scripts/deployment/provider-mirror/logic/global/mirror-ctl.sh start
   ./scripts/deployment/provider-mirror/logic/global/mirror-ctl.sh configure
   ```
3. **Format & Validate:**
   ```bash
   terraform fmt -recursive
   terraform validate
   tflint
   ```
4. **Run Local Validation:**
   ```bash
   python scripts/validation/folder/hierarchy/logic/global/validate.sh
   ```
5. **Submit for Review:**
   Push your changes and create a PR. All PRs require at least one approval from the `@platform-team`.

---

## 🛡 Security Checklist

- [ ] All resources use **CMEK Dual-Key** encryption.
- [ ] No public IP addresses assigned to compute resources.
- [ ] All traffic routed through the **GCP Load Balancer**.
- [ ] Firewall rules follow "Principle of Least Privilege".
- [ ] Labels applied to all resources (env, team, app, cost_center).

## 📞 Support

- **Slack:** `#platform-infra-support`
- **Docs:** [GCP Landing Zone Docs](https://github.com/kushin77/GCP-landing-zone/tree/main/docs)
- **Issues:** Report in the `GCP-landing-zone` repository issues tab.
