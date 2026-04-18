# ==============================================================================
# Agentic Infrastructure Outputs
# ==============================================================================

output "agent_service_url" {
  description = "Cloud Run agent service public URL"
  value       = google_cloud_run_service.agents.status[0].url
}

output "orchestrator_service_url" {
  description = "Cloud Run orchestrator service public URL"
  value       = google_cloud_run_service.orchestrator.status[0].url
}

output "cloud_tasks_queue_name" {
  description = "Cloud Tasks queue name"
  value       = google_cloud_tasks_queue.agent_tasks.name
}

output "pubsub_results_topic" {
  description = "Pub/Sub results topic"
  value       = google_pubsub_topic.results.name
}

output "pubsub_dlq_topic" {
  description = "Pub/Sub DLQ topic"
  value       = google_pubsub_topic.dlq.name
}

output "firestore_database" {
  description = "Firestore database"
  value       = google_firestore_database.agents.name
}

output "bigquery_dataset" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.agents.dataset_id
}

output "bigquery_execution_logs_table" {
  description = "BigQuery execution logs table"
  value       = google_bigquery_table.execution_logs.table_id
}
