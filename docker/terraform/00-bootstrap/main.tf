# ==============================================================================
# Bootstrap: Main Project Initialization
# ==============================================================================
# This phase initializes the project state, encryption keys, and base networking.
# Compliance: Enforces 24-label mandate and Elite Architecture Standards

terraform {
  required_version = ">= 1.7"

  # Initial run uses local backend; subsequent runs move to GCS
  backend "local" {
    path = "terraform.tfstate"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

locals {
  # Merge labels for easy reference
  labels = var.project_labels
}

# 1. Enable Core APIs
resource "google_project_service" "services" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "cloudkms.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "vpcaccess.googleapis.com",
    "dlp.googleapis.com",
    "aiplatform.googleapis.com"
  ])
  service            = each.key
  disable_on_destroy = false
}

# 2. GCS Bucket for Terraform State (Backend for next phases)
data "google_storage_project_service_account" "gcs_account" {
  project = var.project_id
}

resource "google_kms_crypto_key_iam_member" "gcs_kms" {
  crypto_key_id = google_kms_crypto_key.state_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${data.google_storage_project_service_account.gcs_account.email_address}"
}

resource "google_storage_bucket" "tf_state" {
  name                        = "${var.project_id}-tf-state"
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true
  labels                      = local.labels

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.state_key.id
  }

  depends_on = [
    google_project_service.services,
    google_kms_crypto_key_iam_member.gcs_kms
  ]
}

# 3. Cloud KMS: KeyRing and CryptoKeys (CMEK)
resource "google_kms_key_ring" "main" {
  name     = "${var.project_id}-keyring"
  location = var.region

  depends_on = [google_project_service.services]
}

resource "google_kms_crypto_key" "state_key" {
  name     = "tf-state-key"
  key_ring = google_kms_key_ring.main.id
  purpose  = "ENCRYPT_DECRYPT"
  labels   = local.labels

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "app_key" {
  name     = "app-data-key"
  key_ring = google_kms_key_ring.main.id
  purpose  = "ENCRYPT_DECRYPT"
  labels   = local.labels

  lifecycle {
    prevent_destroy = true
  }
}

# 4. Service Account for CI/CD Workflow (GCP Landing Zone Identity)
resource "google_service_account" "github_actions" {
  account_id   = "github-actions-lz-onboard"
  display_name = "GitHub Actions Onboarding Service Account"
  project      = var.project_id
}

# 5. Workload Identity Pool and Provider (Identity Federation)
resource "google_iam_workload_identity_pool" "github_pool" {
  workload_identity_pool_id = "github-actions-pool-ollama"
  display_name              = "GitHub Actions Pool"
  description               = "Identity pool for GitHub Actions authentication"
  project                   = var.project_id

  depends_on = [google_project_service.services]
}

resource "google_iam_workload_identity_pool_provider" "github_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub Provider"
  project                            = var.project_id

  attribute_mapping = {
    "google.subject"             = "assertion.sub"
    "attribute.actor"            = "assertion.actor"
    "attribute.repository"       = "assertion.repository"
    "attribute.repository_owner" = "assertion.repository_owner"
  }

  attribute_condition = "assertion.repository_owner == 'kushin77'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Allow GitHub Actions to use the Service Account
resource "google_service_account_iam_member" "workload_identity_user" {
  service_account_id = google_service_account.github_actions.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/kushin77/ollama"
}

output "bootstrap_status" {
  value = "Ready for execution"
}

output "state_bucket" {
  value = google_storage_bucket.tf_state.name
}

output "kms_key_ring" {
  value = google_kms_key_ring.main.id
}

output "github_actions_sa" {
  value = google_service_account.github_actions.email
}
