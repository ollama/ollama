# GCP Scheduled Scaling Infrastructure
#
# Implements time-based autoscaling to reduce costs during off-hours
# while maintaining performance during peak hours.
#
# Cost Optimization:
#   - Peak hours (9am-6pm): Full capacity
#   - Off-peak (6pm-9am): Minimal capacity (50% reduction)
#   - Weekend: Minimal capacity (60% reduction)
#   - Expected savings: 30-50% monthly compute costs

# Cloud Scheduler for scale-down jobs
locals {
  mandatory_labels = {
    environment      = var.environment
    team             = var.team
    application      = "ollama"
    component        = "api"
    cost-center      = var.cost_center
    managed-by       = "terraform"
    git_repo         = "github.com/kushin77/ollama"
    lifecycle_status = var.lifecycle_status
  }
}

resource "google_cloud_scheduler_job" "ollama_scale_down_evening" {
  name             = "${var.environment}-ollama-scale-down-evening"
  description      = "Scale down Ollama to minimal capacity at 6 PM (off-hours)"
  schedule         = "0 18 * * 1-5"  # 6 PM weekdays
  time_zone        = "America/New_York"
  attempt_deadline = "320s"
  region           = var.gcp_region

  http_target {
    http_method = "PATCH"
    uri         = "https://${var.gcp_region}-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/${var.project_id}/services/${var.environment}-ollama-api"

    headers = {
      "Content-Type" = "application/json"
    }

    oidc_token {
      service_account_email = google_service_account.cloud_scheduler.email
      audience              = "https://${var.gcp_region}-run.googleapis.com"
    }

    body = base64encode(jsonencode({
      apiVersion = "serving.knative.dev/v1"
      kind       = "Service"
      metadata = {
        name      = "${var.environment}-ollama-api"
        namespace = var.project_id
        labels    = local.mandatory_labels
      }
      spec = {
        template = {
          metadata = {
            annotations = {
              "autoscaling.knative.dev/maxScale"     = "2"
              "autoscaling.knative.dev/minScale"     = "1"
              "autoscaling.knative.dev/targetUtilization" = "70"
            }
          }
          spec = {
            containerConcurrency = 10
            containers = [
              {
                image = "ollama:latest"
                resources = {
                  limits = {
                    cpu    = "1"
                    memory = "2Gi"
                  }
                }
              }
            ]
          }
        }
      }
    }))
  }
}

# Cloud Scheduler for scale-up jobs
resource "google_cloud_scheduler_job" "ollama_scale_up_morning" {
  name             = "${var.environment}-ollama-scale-up-morning"
  description      = "Scale up Ollama to full capacity at 9 AM (peak hours)"
  schedule         = "0 9 * * 1-5"  # 9 AM weekdays
  time_zone        = "America/New_York"
  attempt_deadline = "320s"
  region           = var.gcp_region

  http_target {
    http_method = "PATCH"
    uri         = "https://${var.gcp_region}-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/${var.project_id}/services/${var.environment}-ollama-api"

    headers = {
      "Content-Type" = "application/json"
    }

    oidc_token {
      service_account_email = google_service_account.cloud_scheduler.email
      audience              = "https://${var.gcp_region}-run.googleapis.com"
    }

    body = base64encode(jsonencode({
      apiVersion = "serving.knative.dev/v1"
      kind       = "Service"
      metadata = {
        name      = "${var.environment}-ollama-api"
        namespace = var.project_id
        labels    = local.mandatory_labels
      }
      spec = {
        template = {
          metadata = {
            annotations = {
              "autoscaling.knative.dev/maxScale"     = "10"
              "autoscaling.knative.dev/minScale"     = "3"
              "autoscaling.knative.dev/targetUtilization" = "80"
            }
          }
          spec = {
            containerConcurrency = 50
            containers = [
              {
                image = "ollama:latest"
                resources = {
                  limits = {
                    cpu    = "4"
                    memory = "8Gi"
                  }
                }
              }
            ]
          }
        }
      }
    }))
  }
}

# Weekend scaling (minimal)
resource "google_cloud_scheduler_job" "ollama_scale_weekend" {
  name             = "${var.environment}-ollama-scale-weekend"
  description      = "Scale down Ollama on weekends (minimal capacity)"
  schedule         = "0 0 * * 0,6"  # Midnight on Saturday/Sunday
  time_zone        = "America/New_York"
  attempt_deadline = "320s"
  region           = var.gcp_region

  http_target {
    http_method = "PATCH"
    uri         = "https://${var.gcp_region}-run.googleapis.com/apis/serving.knative.dev/v1/namespaces/${var.project_id}/services/${var.environment}-ollama-api"

    headers = {
      "Content-Type" = "application/json"
    }

    oidc_token {
      service_account_email = google_service_account.cloud_scheduler.email
      audience              = "https://${var.gcp_region}-run.googleapis.com"
    }

    body = base64encode(jsonencode({
      apiVersion = "serving.knative.dev/v1"
      kind       = "Service"
      metadata = {
        name      = "${var.environment}-ollama-api"
        namespace = var.project_id
        labels    = local.mandatory_labels
      }
      spec = {
        template = {
          metadata = {
            annotations = {
              "autoscaling.knative.dev/maxScale"     = "1"
              "autoscaling.knative.dev/minScale"     = "1"
              "autoscaling.knative.dev/targetUtilization" = "50"
            }
          }
          spec = {
            containerConcurrency = 5
            containers = [
              {
                image = "ollama:latest"
                resources = {
                  limits = {
                    cpu    = "500m"
                    memory = "1Gi"
                  }
                }
              }
            ]
          }
        }
      }
    }))
  }
}

# Service account for Cloud Scheduler
resource "google_service_account" "cloud_scheduler" {
  account_id   = "cloud-scheduler-ollama"
  display_name = "${var.environment}-ollama-scheduler"
  description  = "Service account for managing ${var.environment} Ollama Cloud Run scaling"
}

# IAM binding for scheduler to update Cloud Run services
resource "google_cloud_run_service_iam_member" "scheduler_updater" {
  service  = google_cloud_run_service.ollama_api.name
  location = var.gcp_region
  role     = "roles/run.admin"
  member   = "serviceAccount:${google_service_account.cloud_scheduler.email}"
}

# Monitoring alert for scaling failures
resource "google_monitoring_alert_policy" "scaling_failure" {
  display_name = "${var.environment}-ollama-policy-scaling-failure"
  combiner     = "OR"

  conditions {
    display_name = "${var.environment}-ollama-scheduler-job-failed"

    condition_threshold {
      filter          = "metric.type=\"cloudscheduler.googleapis.com/job_attempts\" AND resource.labels.job_name=~\"${var.environment}-ollama-scale.*\" AND metric.labels.attempt_type=\"FAILED\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period  = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.slack_alerts.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

# Outputs
output "scheduler_evening_job" {
  value       = google_cloud_scheduler_job.ollama_scale_down_evening.name
  description = "Evening scale-down scheduler job"
}

output "scheduler_morning_job" {
  value       = google_cloud_scheduler_job.ollama_scale_up_morning.name
  description = "Morning scale-up scheduler job"
}

output "scheduler_weekend_job" {
  value       = google_cloud_scheduler_job.ollama_scale_weekend.name
  description = "Weekend scale-down scheduler job"
}

output "estimated_monthly_savings" {
  value       = "30-50% compute cost reduction"
  description = "Expected monthly savings from scheduled scaling"
}
