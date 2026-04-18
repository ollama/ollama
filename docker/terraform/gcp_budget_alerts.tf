# GCP Budget Alerts Configuration
# Sets up budget monitoring and alerts at 50%, 80%, and 100% thresholds
# Part of GCP Landing Zone compliance for cost governance
# Note: terraform block centralized in main.tf

# Local labels for budget resources
locals {
  budget_labels = {
    environment      = var.environment
    team             = var.team
    application      = "ollama"
    component        = "budget"
    cost-center      = var.cost_center
    managed-by       = "terraform"
    git_repo         = "github.com/kushin77/ollama"
    lifecycle_status = var.lifecycle_status
  }
}

# Get current billing account
data "google_billing_account" "account" {
  billing_account = "0119B0-6AF18C-A12474"
}

# Budget for Ollama project with threshold rules
resource "google_billing_budget" "ollama_budget" {
  billing_account = data.google_billing_account.account.id
  display_name    = "${var.environment}-ollama-budget-alerts"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = var.monthly_budget_usd
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }

  threshold_rules {
    threshold_percent = 0.8
  }

  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    monitoring_notification_channels = [
      google_monitoring_notification_channel.budget_email.id,
      google_monitoring_notification_channel.budget_email_critical.id
    ]
  }
}

# Email notification channel for warning alerts (50% and 80%)
resource "google_monitoring_notification_channel" "budget_email" {
  display_name = "${var.environment}-ollama-budget-warning-email"
  type         = "email"

  labels = {
    email_address = var.budget_alert_email
  }

  enabled = true
}

# Email notification channel for critical alerts (100%)
resource "google_monitoring_notification_channel" "budget_email_critical" {
  display_name = "${var.environment}-ollama-budget-critical-email"
  type         = "email"

  labels = {
    email_address = var.budget_alert_email_critical
  }

  enabled = true
}

# Alert policy for budget threshold - Removed because budget resource itself handles notifications
# via all_updates_rule, and the billing_budget resource type is not supported in Monitoring Alert Policies directly.

# Cloud Monitoring dashboard for budget tracking
resource "google_monitoring_dashboard" "budget_dashboard" {
  dashboard_json = jsonencode({
    displayName = "${var.environment}-ollama-budget-dashboard"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          xPos = 0
          yPos = 0
          width  = 6
          height = 4
          widget = {
            title = "Monthly Spend Forecast"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter       = "metric.type=\"billing.googleapis.com/billing_account_aggregated_transaction_amount\""
                      aggregation = {
                        alignmentPeriod = "2592000s"
                      }
                    }
                  }
                }
              ]
            }
          }
        },
        {
          xPos = 6
          yPos = 0
          width  = 6
          height = 4
          widget = {
            title = "Budget Utilization"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"gce_instance\" AND metric.type=\"compute.googleapis.com/instance/cpu/utilization\""
                    }
                  }
                }
              ]
            }
          }
        }
      ]
    }
  })
}

# Outputs for integration with other systems
output "budget_id" {
  value       = google_billing_budget.ollama_budget.id
  description = "Budget alert ID"
}

output "notification_channel_warning_id" {
  value       = google_monitoring_notification_channel.budget_email.id
  description = "Notification channel ID for warning alerts (50%/80%)"
}

output "notification_channel_critical_id" {
  value       = google_monitoring_notification_channel.budget_email_critical.id
  description = "Notification channel ID for critical alerts (100%)"
}

output "dashboard_url" {
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.budget_dashboard.id}?project=${var.project_id}"
  description = "URL to budget monitoring dashboard"
}
