# ==============================================================================
# Ollama Agentic GCP Infrastructure - Minimal Production-Ready Configuration
# ==============================================================================

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ==============================================================================
# Cloud Run: Agent Service
# ==============================================================================

resource "google_cloud_run_service" "agents" {
  name     = "${var.environment}-${var.application}-agents"
  location = var.region
  project  = var.project_id

  template {
    spec {
      containers {
        image = var.agent_image_uri
        env {
          name  = "ENVIRONMENT"
          value = var.environment
        }
        env {
          name  = "OLLAMA_BASE_URL"
          value = var.ollama_service_url
        }
        ports {
          container_port = 8000
        }
        resources {
          limits = {
            cpu    = var.cpu_limit
            memory = var.memory_limit
          }
        }
      }
    }
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = var.max_instances
        "autoscaling.knative.dev/minScale" = var.min_instances
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# ==============================================================================
# Cloud Run: Orchestrator Service
# ==============================================================================

resource "google_cloud_run_service" "orchestrator" {
  name     = "${var.environment}-${var.application}-orchestrator"
  location = var.region
  project  = var.project_id

  template {
    spec {
      containers {
        image = var.orchestrator_image_uri
        env {
          name  = "AGENTS_SERVICE_URL"
          value = google_cloud_run_service.agents.status[0].url
        }
        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
        ports {
          container_port = 8000
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }
      }
    }
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = var.orchestrator_max_instances
        "autoscaling.knative.dev/minScale" = var.orchestrator_min_instances
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_cloud_run_service.agents]
}

# ==============================================================================
# Cloud Tasks Queue
# ==============================================================================

resource "google_cloud_tasks_queue" "agent_tasks" {
  name     = "${var.environment}-${var.application}-tasks"
  location = var.region
  project  = var.project_id

  rate_limits {
    max_concurrent_dispatches = 100
    max_dispatches_per_second = 50
  }

  retry_config {
    max_attempts       = 5
    max_backoff        = "3600s"
  }
}

# ==============================================================================
# Firestore Database
# ==============================================================================

resource "google_firestore_database" "agents" {
  name        = "projects/${var.project_id}/databases/(default)"
  location_id = var.firestore_location
  type        = "FIRESTORE_NATIVE"
  project     = var.project_id
}

# ==============================================================================
# Pub/Sub Topics
# ==============================================================================

resource "google_pubsub_topic" "results" {
  name    = "${var.environment}-${var.application}-results"
  project = var.project_id
}

resource "google_pubsub_topic" "dlq" {
  name    = "${var.environment}-${var.application}-dlq"
  project = var.project_id
}

# ==============================================================================
# BigQuery Dataset
# ==============================================================================

resource "google_bigquery_dataset" "agents" {
  dataset_id = "${replace(var.environment, "-", "_")}_${replace(var.application, "-", "_")}"
  location   = var.bq_location
  project    = var.project_id
}

# ==============================================================================
# BigQuery Table
# ==============================================================================

resource "google_bigquery_table" "execution_logs" {
  dataset_id = google_bigquery_dataset.agents.dataset_id
  table_id   = "execution_logs"
  project    = var.project_id

  schema = jsonencode([
    { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "task_id", type = "STRING", mode = "REQUIRED" },
    { name = "agent_id", type = "STRING", mode = "REQUIRED" },
    { name = "status", type = "STRING", mode = "REQUIRED" },
    { name = "duration_ms", type = "INTEGER", mode = "NULLABLE" },
    { name = "tokens_used", type = "INTEGER", mode = "NULLABLE" },
    { name = "error_message", type = "STRING", mode = "NULLABLE" }
  ])
}
