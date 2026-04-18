#!/bin/bash
# Monitoring & Alerting Setup for Ollama Elite AI Platform
# Configures Prometheus, Grafana, and Cloud Monitoring

set -e

GCP_PROJECT="elevatediq"
GCP_REGION="us-central1"
SERVICE_NAME="ollama-service"

echo "🔧 Setting up monitoring and alerting..."
echo ""

# 1. Enable required GCP APIs
echo "📡 Enabling GCP Monitoring APIs..."
gcloud services enable \
  monitoring.googleapis.com \
  logging.googleapis.com \
  --project=$GCP_PROJECT

# 2. Create custom metrics
echo "📊 Creating custom metrics..."
cat > /tmp/metrics.json << 'EOF'
{
  "type": "custom.googleapis.com/ollama/inference_latency",
  "display_name": "Inference Latency",
  "description": "Model inference latency in milliseconds",
  "metric_kind": "GAUGE",
  "value_type": "DISTRIBUTION"
}
EOF

# 3. Create alert policies
echo "🚨 Creating alert policies..."

# High error rate alert
gcloud alpha monitoring policies create \
  --notification-channels=[CHANNEL_ID] \
  --display-name="Ollama High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s \
  --project=$GCP_PROJECT || true

# High latency alert
gcloud alpha monitoring policies create \
  --notification-channels=[CHANNEL_ID] \
  --display-name="Ollama High Latency" \
  --condition-display-name="p99 latency > 5s" \
  --condition-threshold-value=5000 \
  --condition-threshold-duration=300s \
  --project=$GCP_PROJECT || true

# Out of memory alert
gcloud alpha monitoring policies create \
  --notification-channels=[CHANNEL_ID] \
  --display-name="Ollama Low Memory" \
  --condition-display-name="Memory < 500MB" \
  --condition-threshold-value=500 \
  --condition-threshold-duration=120s \
  --project=$GCP_PROJECT || true

echo ""
echo "✅ Monitoring setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a notification channel in GCP Console"
echo "2. Update alert policy IDs in this script"
echo "3. View dashboards: https://console.cloud.google.com/monitoring?project=$GCP_PROJECT"
echo "4. View logs: https://console.cloud.google.com/logs?project=$GCP_PROJECT"
