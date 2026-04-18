# Security Monitoring Dashboards for Real-time Threat Detection
# Visualizes security metrics, incidents, and compliance status

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

locals {
  dashboard_prefix = "ollama-security-${var.environment}"

  # Dashboard configuration
  dashboard_layout = {
    columns = 12
    widgets = 8
  }
}

# Main Security Dashboard
resource "google_monitoring_dashboard" "security_overview" {
  dashboard_json = jsonencode({
    displayName = "Ollama Security Overview (${var.environment})"
    mosaicLayout = {
      columns = local.dashboard_layout.columns
      tiles = [
        # Title widget
        {
          width  = 12
          height = 1
          widget = {
            title = "Security Posture Dashboard"
            text = {
              content = "Real-time security metrics and threat indicators"
            }
          }
        },

        # Binary Authorization Status
        {
          xPos   = 0
          yPos   = 1
          width  = 6
          height = 4
          widget = {
            title = "Binary Authorization Policy Decisions"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"custom.googleapis.com/binary_auth/policy_decisions\" resource.type=\"k8s_cluster\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields = [
                        "resource.label.cluster_name",
                        "metric.label.decision"
                      ]
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              }]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Decisions/sec"
                scale = "LINEAR"
              }
              chartOptions = {
                mode = "COLOR"
              }
            }
          }
        },

        # Vulnerability Scan Results
        {
          xPos   = 6
          yPos   = 1
          width  = 6
          height = 4
          widget = {
            title = "Container Vulnerabilities Detected"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"custom.googleapis.com/container/vulnerabilities_detected\" resource.type=\"k8s_cluster\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                      groupByFields = ["metric.label.severity"]
                    }
                  }
                }
                plotType = "STACKED_AREA"
              }]
              yAxis = {
                label = "Vulnerabilities/sec"
                scale = "LINEAR"
              }
            }
          }
        },

        # Pod Admission Status
        {
          xPos   = 0
          yPos   = 5
          width  = 4
          height = 4
          widget = {
            title = "Pod Admission Rate"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"kubernetes.io/pod/request_count\" resource.type=\"k8s_pod\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
              sparkChartType = "SPARK_BAR"
              thresholds = [
                { value = 50, color = "RED" },
                { value = 100, color = "YELLOW" },
                { value = 200, color = "GREEN" }
              ]
            }
          }
        },

        # Attestation Verification Status
        {
          xPos   = 4
          yPos   = 5
          width  = 4
          height = 4
          widget = {
            title = "Attestation Verification Rate"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/attestation/verification_rate\" resource.type=\"k8s_cluster\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_PERCENT_CHANGE"
                  }
                }
              }
              sparkChartType = "SPARK_LINE"
              thresholds = [
                { value = 99, color = "RED" },
                { value = 99.5, color = "YELLOW" },
                { value = 99.9, color = "GREEN" }
              ]
            }
          }
        },

        # IAM Changes
        {
          xPos   = 8
          yPos   = 5
          width  = 4
          height = 4
          widget = {
            title = "IAM Policy Changes"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "protoPayload.methodName=~\"SetIamPolicy\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_COUNT"
                  }
                }
              }
              sparkChartType = "SPARK_BAR"
            }
          }
        },

        # Firewall Rule Changes
        {
          xPos   = 0
          yPos   = 9
          width  = 6
          height = 4
          widget = {
            title = "Network Firewall Changes"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"gce_firewall_rule\" protoPayload.methodName=~\".*Firewall.*\""
                    aggregation = {
                      alignmentPeriod  = "300s"
                      perSeriesAligner = "ALIGN_COUNT"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Changes"
                scale = "LINEAR"
              }
            }
          }
        },

        # Failed Authentication Attempts
        {
          xPos   = 6
          yPos   = 9
          width  = 6
          height = 4
          widget = {
            title = "Failed Authentication Attempts"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "severity=\"ERROR\" AND protoPayload.status.code=401"
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Failures/sec"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })

  depends_on = [
    google_monitoring_alert_policy.binary_auth_violations,
    google_monitoring_alert_policy.critical_vulnerabilities
  ]
}

# Vulnerability Management Dashboard
resource "google_monitoring_dashboard" "vulnerability_dashboard" {
  dashboard_json = jsonencode({
    displayName = "Vulnerability Management (${var.environment})"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 12
          height = 1
          widget = {
            title = "Container Vulnerability Tracking"
          }
        },

        # Critical Vulnerabilities
        {
          xPos   = 0
          yPos   = 1
          width  = 3
          height = 3
          widget = {
            title = "Critical (CVSS ≥ 9.0)"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/vulnerabilities/critical_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_MAX"
                  }
                }
              }
              thresholds = [
                { value = 0, color = "GREEN" },
                { value = 1, color = "RED" }
              ]
            }
          }
        },

        # High Vulnerabilities
        {
          xPos   = 3
          yPos   = 1
          width  = 3
          height = 3
          widget = {
            title = "High (CVSS 7.0-8.9)"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/vulnerabilities/high_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_MAX"
                  }
                }
              }
              thresholds = [
                { value = 0, color = "GREEN" },
                { value = 5, color = "YELLOW" },
                { value = 10, color = "RED" }
              ]
            }
          }
        },

        # Medium Vulnerabilities
        {
          xPos   = 6
          yPos   = 1
          width  = 3
          height = 3
          widget = {
            title = "Medium (CVSS 4.0-6.9)"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/vulnerabilities/medium_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_MAX"
                  }
                }
              }
            }
          }
        },

        # Low Vulnerabilities
        {
          xPos   = 9
          yPos   = 1
          width  = 3
          height = 3
          widget = {
            title = "Low (CVSS < 4.0)"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/vulnerabilities/low_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_MAX"
                  }
                }
              }
            }
          }
        },

        # Vulnerability Trend
        {
          xPos   = 0
          yPos   = 4
          width  = 12
          height = 4
          widget = {
            title = "Vulnerability Trend Over Time"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/vulnerabilities/critical_count\""
                      aggregation = {
                        alignmentPeriod  = "3600s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                  legendTemplate = "Critical"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/vulnerabilities/high_count\""
                      aggregation = {
                        alignmentPeriod  = "3600s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                  legendTemplate = "High"
                }
              ]
              yAxis = {
                label = "Vulnerability Count"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}

# Alert Policy 1: Binary Authorization Violations
resource "google_monitoring_alert_policy" "binary_auth_violations" {
  display_name = "Binary Authorization Policy Violations (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Unauthorized image deployment attempts"
    condition_threshold = {
      filter          = "metric.type=\"custom.googleapis.com/binary_auth/policy_violations\" resource.type=\"k8s_cluster\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations = [{
        alignment_period  = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields = ["resource.label.cluster_name"]
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  alert_strategy = {
    auto_close = "1800s"
  }

  documentation = {
    content   = "Binary Authorization policy violation detected. Unauthorized image attempted deployment to ${var.environment} cluster."
    mime_type = "text/markdown"
  }
}

# Alert Policy 2: Critical Vulnerabilities
resource "google_monitoring_alert_policy" "critical_vulnerabilities" {
  display_name = "Critical Container Vulnerabilities Detected (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Critical vulnerability found in container image"
    condition_threshold = {
      filter          = "metric.type=\"custom.googleapis.com/vulnerabilities/critical_count\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  alert_strategy = {
    auto_close = "3600s"
  }

  documentation = {
    content   = "CRITICAL vulnerability detected. Image deployment blocked until patched."
    mime_type = "text/markdown"
  }
}

# Alert Policy 3: High Vulnerabilities Requiring Approval
resource "google_monitoring_alert_policy" "high_vulnerabilities" {
  display_name = "High Vulnerabilities Detected (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "High severity vulnerabilities found"
    condition_threshold = {
      filter          = "metric.type=\"custom.googleapis.com/vulnerabilities/high_count\" metric.value > 5"
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  documentation = {
    content   = "High severity vulnerabilities detected. Security review required for deployment approval."
    mime_type = "text/markdown"
  }
}

# Alert Policy 4: Failed Authentication Attempts
resource "google_monitoring_alert_policy" "failed_auth_attempts" {
  display_name = "Failed Authentication Attempts (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "Multiple failed authentication attempts"
    condition_threshold = {
      filter          = "protoPayload.status.code=401 OR protoPayload.status.code=403"
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  documentation = {
    content   = "Potential unauthorized access attempt detected (multiple auth failures)."
    mime_type = "text/markdown"
  }
}

# Alert Policy 5: IAM Policy Changes
resource "google_monitoring_alert_policy" "iam_changes_alert" {
  display_name = "Unexpected IAM Policy Changes (${var.environment})"
  combiner     = "OR"
  enabled      = true

  conditions = [{
    display_name = "IAM policy change outside business hours"
    condition_threshold = {
      filter          = "protoPayload.methodName=~\"SetIamPolicy\" AND timestamp NOT BETWEEN \"09:00\" AND \"17:00\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations = [{
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_COUNT"
      }]
    }
  }]

  notification_channels = [var.security_notification_channel_id]

  documentation = {
    content   = "IAM policy change detected outside business hours. Review change for legitimacy."
    mime_type = "text/markdown"
  }
}

# Output
output "security_dashboards" {
  value = {
    overview           = google_monitoring_dashboard.security_overview.id
    vulnerabilities    = google_monitoring_dashboard.vulnerability_dashboard.id
  }
  description = "Security dashboards for monitoring"
}

output "alert_policies" {
  value = {
    binary_auth        = google_monitoring_alert_policy.binary_auth_violations.id
    critical_vulns     = google_monitoring_alert_policy.critical_vulnerabilities.id
    high_vulns         = google_monitoring_alert_policy.high_vulnerabilities.id
    failed_auth        = google_monitoring_alert_policy.failed_auth_attempts.id
    iam_changes        = google_monitoring_alert_policy.iam_changes_alert.id
  }
  description = "Alert policy IDs"
}
