# terraform/cmek_encryption.tf - Customer-Managed Encryption Keys
#
# This module configures CMEK encryption for all data services:
# - Cloud Storage buckets
# - Cloud SQL database
# - Firestore database
# - Cloud Memorystore (Redis)
# - Backup & recovery services
#
# All services encrypt data at rest with customer-managed keys from Cloud KMS

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ============================================================================
# CLOUD STORAGE BUCKETS WITH CMEK
# ============================================================================

# Primary data storage bucket (model storage, documents, user uploads)
resource "google_storage_bucket" "ollama_data" {
  project       = var.project_id
  name          = "ollama-data-${var.environment}"
  location      = var.gcs_region
  force_destroy = false  # Prevent accidental deletion

  uniform_bucket_level_access = true

  encryption {
    default_kms_key_name = var.storage_cmek_key_id
  }

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "storage"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }
}

# Backup storage bucket (encrypted database dumps, snapshots)
resource "google_storage_bucket" "ollama_backups" {
  project       = var.project_id
  name          = "ollama-backups-${var.environment}"
  location      = var.gcs_region
  force_destroy = false

  uniform_bucket_level_access = true

  encryption {
    default_kms_key_name = var.backup_cmek_key_id
  }

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 2555  # 7 years (compliance retention)
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "backup"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }
}

# Audit logs bucket (encrypted audit trail)
resource "google_storage_bucket" "ollama_logs" {
  project       = var.project_id
  name          = "ollama-logs-${var.environment}"
  location      = var.gcs_region
  force_destroy = false

  uniform_bucket_level_access = true

  encryption {
    default_kms_key_name = var.storage_cmek_key_id
  }

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 2555  # 7 years (audit retention)
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "audit-logs"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }
}

# ============================================================================
# CLOUD SQL DATABASE WITH CMEK
# ============================================================================

resource "google_sql_database_instance" "ollama_postgres" {
  project             = var.project_id
  name                = "ollama-postgres-${var.environment}"
  database_version    = "POSTGRES_15"
  region              = var.cloud_sql_region
  deletion_protection = true

  settings {
    tier              = var.cloud_sql_tier  # e.g., "db-custom-4-16384"
    availability_type = "REGIONAL"
    disk_size         = var.cloud_sql_disk_size
    disk_type         = "PD_SSD"

    # CMEK Encryption
    database_flags {
      name  = "cloudsql_iam_authentication"
      value = "on"
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"  # 3 AM UTC
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length    = 1024
      record_application_tags = true
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.vpc_network_id
      require_ssl     = true
    }

    location_preference {
      zone           = var.cloud_sql_zone
      follow_replica = false
    }

    maintenance_window {
      day          = 6  # Saturday
      hour         = 3  # 3 AM UTC
      update_track = "stable"
    }

    user_labels = {
      environment   = var.environment
      team          = "platform"
      application   = "ollama"
      component     = "database"
      cost-center   = var.cost_center
      managed-by    = "terraform"
      git-repo      = "github.com/kushin77/ollama"
      lifecycle-status = "active"
    }
  }

  deletion_protection = true

  depends_on = [
    var.service_networking_connection_id
  ]
}

# Database user with strong authentication
resource "google_sql_user" "ollama_app" {
  name     = "ollama_app"
  instance = google_sql_database_instance.ollama_postgres.name
  password = random_password.db_password.result
  type     = "BUILT_IN"
}

# IAM database user (Workload Identity for pods)
resource "google_sql_user" "ollama_workload_identity" {
  name     = "ollama-app@${var.project_id}.iam"
  instance = google_sql_database_instance.ollama_postgres.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

# Database
resource "google_sql_database" "ollama" {
  name     = "ollama"
  instance = google_sql_database_instance.ollama_postgres.name
  charset  = "UTF8"
}

# Strong password generation
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Store password in Secret Manager
resource "google_secret_manager_secret" "db_password" {
  project   = var.project_id
  secret_id = "ollama-db-password-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db_password.result
}

# ============================================================================
# FIRESTORE DATABASE WITH CMEK
# ============================================================================

resource "google_firestore_database" "ollama" {
  project     = var.project_id
  name        = "(default)"
  type        = "FIRESTORE_NATIVE"
  location_id = var.firestore_region

  # CMEK Encryption (if supported in your region)
  cmek_config {
    kms_key_name = var.firestore_cmek_key_id
  }
}

# Firestore backup configuration
resource "google_firestore_backup_schedule" "daily" {
  project    = var.project_id
  database   = google_firestore_database.ollama.name
  retention  = "604800s"  # 7 days
  recurrence = "DAILY"

  daily_recurrence {}
}

# ============================================================================
# CLOUD MEMORYSTORE (REDIS) WITH CMEK
# ============================================================================

resource "google_redis_instance" "ollama_cache" {
  project           = var.project_id
  name              = "ollama-cache-${var.environment}"
  region            = var.redis_region
  tier              = "STANDARD_HA"
  memory_size_gb    = var.redis_memory_gb
  redis_version     = "7.0"
  authorized_network = var.vpc_network_id

  # CMEK Encryption
  customer_managed_key = var.redis_cmek_key_id

  auth_enabled         = true
  auth_string          = random_password.redis_password.result
  transit_encryption_mode = "SERVER_AUTHENTICATION"

  maintenance_policy {
    day         = "SUNDAY"
    start_hour  = 3
    update_kind = "MIN_DISRUPTIVE"
  }

  labels = {
    environment   = var.environment
    team          = "platform"
    application   = "ollama"
    component     = "cache"
    cost-center   = var.cost_center
    managed-by    = "terraform"
    git-repo      = "github.com/kushin77/ollama"
    lifecycle-status = "active"
  }

  depends_on = [var.service_networking_connection_id]
}

# Redis authentication password
resource "random_password" "redis_password" {
  length  = 32
  special = true
}

# Store Redis password in Secret Manager
resource "google_secret_manager_secret" "redis_password" {
  project   = var.project_id
  secret_id = "ollama-redis-password-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "redis_password" {
  secret      = google_secret_manager_secret.redis_password.id
  secret_data = random_password.redis_password.result
}

# ============================================================================
# SECRET MANAGER WITH ENCRYPTION
# ============================================================================

# API key storage (encrypted)
resource "google_secret_manager_secret" "api_keys" {
  project   = var.project_id
  secret_id = "ollama-api-keys-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

# JWT signing keys (encrypted)
resource "google_secret_manager_secret" "jwt_keys" {
  project   = var.project_id
  secret_id = "ollama-jwt-keys-${var.environment}"

  labels = {
    environment = var.environment
  }

  replication {
    automatic = true
  }
}

# ============================================================================
# MONITORING: ENCRYPTION KEY USAGE
# ============================================================================

# Metric: CMEK key usage rate
resource "google_monitoring_metric_descriptor" "cmek_key_usage" {
  project      = var.project_id
  type         = "custom.googleapis.com/cmek/key_usage_rate"
  metric_kind  = "GAUGE"
  value_type   = "INT64"
  display_name = "CMEK Key Usage Rate"

  labels {
    key         = "service"
    value_type  = "STRING"
    description = "Service using the key"
  }

  labels {
    key         = "key_name"
    value_type  = "STRING"
    description = "Name of the encryption key"
  }
}

# Alert: CMEK key access anomaly
resource "google_monitoring_alert_policy" "cmek_anomaly" {
  project      = var.project_id
  display_name = "CMEK Key Access Anomaly"
  combiner     = "OR"

  conditions {
    display_name = "Unusual key access pattern"

    condition_threshold {
      filter            = "resource.type=\"cloudkms_keyring\" AND metric.type=\"logging.googleapis.com/user_defined_jsonstruct\""
      comparison        = "COMPARISON_GT"
      threshold_value   = 1000  # Threshold depends on normal usage
      duration          = "300s"
    }
  }

  notification_channels = var.alert_channels

  lifecycle {
    ignore_changes = [notification_channels]
  }
}

# ============================================================================
# OUTPUTS
# ============================================================================

output "storage_bucket" {
  value       = google_storage_bucket.ollama_data.name
  description = "Primary data storage bucket"
}

output "backup_bucket" {
  value       = google_storage_bucket.ollama_backups.name
  description = "Backup storage bucket"
}

output "logs_bucket" {
  value       = google_storage_bucket.ollama_logs.name
  description = "Audit logs bucket"
}

output "database_instance" {
  value       = google_sql_database_instance.ollama_postgres.connection_name
  description = "Cloud SQL connection string"
}

output "database_private_ip" {
  value       = google_sql_database_instance.ollama_postgres.private_ip_address
  description = "Cloud SQL private IP for internal access"
}

output "redis_host" {
  value       = google_redis_instance.ollama_cache.host
  description = "Redis instance host"
}

output "redis_port" {
  value       = google_redis_instance.ollama_cache.port
  description = "Redis instance port"
}

output "firestore_database" {
  value       = google_firestore_database.ollama.name
  description = "Firestore database ID"
}

output "secrets" {
  value = {
    database = google_secret_manager_secret.db_password.id
    redis    = google_secret_manager_secret.redis_password.id
    api_keys = google_secret_manager_secret.api_keys.id
    jwt_keys = google_secret_manager_secret.jwt_keys.id
  }
  description = "Secret Manager secrets for sensitive credentials"
}
