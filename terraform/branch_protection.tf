/**
 * Branch Protection Configuration for kushin77/ollama
 * 
 * Enforces:
 * - Required status checks (CI validation)
 * - Required PR reviews before merge
 * - Required signed commits (GPG)
 * - Dismisses stale PR approvals when new commits pushed
 * - Blocks force pushes to main
 * 
 * Prerequisites:
 * - GitHub token with admin:repo_hook scope
 * - Terraform AWS provider configured
 * - CI status check "validate-landing-zone" must be created first
 * 
 * Usage:
 *   terraform init
 *   terraform plan
 *   terraform apply
 */

terraform {
  required_version = ">= 1.0"
  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 5.0"
    }
  }
}

provider "github" {
  owner = "kushin77"
  token = var.github_token
}

variable "github_token" {
  description = "GitHub personal access token with admin:repo_hook scope"
  type        = string
  sensitive   = true
}

variable "repository" {
  description = "Repository name"
  type        = string
  default     = "ollama"
}

variable "main_branch" {
  description = "Main branch name"
  type        = string
  default     = "main"
}

# Fetch the repository
data "github_repository" "ollama" {
  name = var.repository
}

# Configure branch protection for main
resource "github_branch_protection" "main" {
  repository_id          = data.github_repository.ollama.node_id
  pattern                = var.main_branch
  require_conversation_resolution = true

  # Require status checks
  required_status_checks {
    strict   = true
    contexts = [
      "validate-landing-zone",  # Custom CI check (created separately)
      "codeql-analysis",         # CodeQL (if enabled)
    ]
  }

  # Require pull request reviews
  required_pull_request_reviews {
    required_approving_review_count = 1
    require_code_owner_reviews      = false
    dismiss_stale_reviews           = true
  }

  # Require commit signatures
  require_signed_commits = true

  # Restrict force pushes and deletions
  allow_force_pushes = false
  allow_deletions    = false

  enforce_admins = true

  # Allow auto-merge (optional, for Dependabot)
  # allow_auto_merge = true
}

# Optional: Create a status check for branch protection to reference
# (This would be created by your CI/CD pipeline, e.g., GitHub Actions)
# resource "github_repository_status_check" "validate_landing_zone" {
#   repository       = var.repository
#   context          = "validate-landing-zone"
#   url              = "https://github.com/kushin77/ollama/actions"
#   description      = "Landing Zone compliance validation"
# }

output "branch_protection" {
  description = "Branch protection configuration applied to main"
  value = {
    repository           = data.github_repository.ollama.name
    branch               = var.main_branch
    require_pr_reviews   = true
    require_signed_commits = true
    enforce_admins       = true
    restrict_force_push  = true
  }
}
