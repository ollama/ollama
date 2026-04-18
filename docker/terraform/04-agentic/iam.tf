# ==============================================================================
# Service Accounts & Zero Trust Authentication
# ==============================================================================
# Landing Zone Mandate #4: Zero Trust Auth
# - Workload Identity for GCP authentication (no service account keys)
# - Least-privilege IAM roles
# - Cross-service authentication via Workload Identity

# =============================================================================
# Service Account: Agents Service
# =============================================================================

resource "google_service_account" "agents" {
  account_id   = "${var.environment}-${var.application}-agents"
  display_name = "Service account for Ollama agents (${var.environment})"
  project      = var.project_id
  description  = "Cloud Run service account with permissions for Ollama agent execution"
}

# IAM Roles for Agents Service Account
resource "google_project_iam_member" "agents_logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.agents.email}"
}

resource "google_project_iam_member" "agents_metrics_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.agents.email}"
}

resource "google_project_iam_member" "agents_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.agents.email}"
}

resource "google_project_iam_member" "agents_artifactreader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.agents.email}"
}

resource "google_project_iam_member" "agents_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.agents.email}"
}

resource "google_project_iam_member" "agents_firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.agents.email}"
}

resource "google_project_iam_member" "agents_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.agents.email}"
}

# =============================================================================
# Service Account: Orchestrator
# =============================================================================

resource "google_service_account" "orchestrator" {
  account_id   = "${var.environment}-${var.application}-orchestrator"
  display_name = "Service account for Ollama orchestrator (${var.environment})"
  project      = var.project_id
  description  = "Cloud Run service account for task orchestration and distribution"
}

# IAM Roles for Orchestrator Service Account
resource "google_project_iam_member" "orchestrator_logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_project_iam_member" "orchestrator_tasks_creator" {
  project = var.project_id
  role    = "roles/cloudtasks.taskRunner"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_project_iam_member" "orchestrator_tasks_enqueuer" {
  project = var.project_id
  role    = "roles/cloudtasks.enqueuer"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_project_iam_member" "orchestrator_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_project_iam_member" "orchestrator_firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

# =============================================================================
# Workload Identity Federation (GitHub Actions)
# =============================================================================
# Zero Trust auth from GitHub Actions to GCP (no long-lived credentials)

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = "github-actions-pool-ollama"
  workload_identity_pool_provider_id = "github-provider"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.environment" = "assertion.environment"
  }
  attribute_condition = "assertion.repository == 'kushin77/ollama'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
  project = var.project_id
}

# Allow GitHub Actions to impersonate service accounts
resource "google_service_account_iam_member" "github_agents_deployer" {
  service_account_id = google_service_account.agents.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${var.project_id}/locations/global/workloadIdentityPools/github-actions-pool-ollama/attribute.repository/kushin77/ollama"
}

resource "google_service_account_iam_member" "github_orchestrator_deployer" {
  service_account_id = google_service_account.orchestrator.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${var.project_id}/locations/global/workloadIdentityPools/github-actions-pool-ollama/attribute.repository/kushin77/ollama"
}

# =============================================================================
# Secret Manager Access (IAM)
# =============================================================================
# Grant service accounts access to secrets in Secret Manager

resource "google_secret_manager_secret_iam_member" "agents_accessor" {
  secret_id = "ollama-agents-config"  # Must exist in Secret Manager
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agents.email}"
}

resource "google_secret_manager_secret_iam_member" "orchestrator_accessor" {
  secret_id = "ollama-orchestrator-config"  # Must exist in Secret Manager
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.orchestrator.email}"
}
