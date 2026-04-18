#!/bin/bash
# =============================================================================
# CONTINUOUS MONITORING & ALERTING AUTOMATION
# Real-time health checks, metrics collection, and alert management
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT="${1:-dev}"
ACTION="${2:-health}"
STATUS_FILE="/tmp/ollama-status-${ENVIRONMENT}.json"
ALERT_LOG="${PROJECT_ROOT}/logs/alerts-${ENVIRONMENT}.log"
HEALTH_HISTORY="${PROJECT_ROOT}/logs/health-history-${ENVIRONMENT}.log"

# Thresholds
API_RESPONSE_TIME_THRESHOLD=500    # ms
ERROR_RATE_THRESHOLD=1              # %
CPU_THRESHOLD=80                    # %
MEMORY_THRESHOLD=85                 # %
DISK_THRESHOLD=90                   # %
DB_CONNECTION_THRESHOLD=100         # connections

# Metrics endpoints
API_ENDPOINT="${API_ENDPOINT:-https://elevatediq.ai/ollama}"
PROMETHEUS_ENDPOINT="${PROMETHEUS_ENDPOINT:-http://localhost:9090}"
GRAFANA_ENDPOINT="${GRAFANA_ENDPOINT:-http://localhost:3000}"

mkdir -p "$(dirname "$ALERT_LOG")" "$(dirname "$HEALTH_HISTORY")"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

error() {
    echo -e "${RED}[✗]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[!]${NC} $*"
}

info() {
    echo -e "${CYAN}[i]${NC} $*"
}

alert() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

    echo "[${timestamp}] [${level}] ${message}" >> "$ALERT_LOG"

    if [[ "$level" == "CRITICAL" ]]; then
        error "$message"
        send_alert "$message" "critical"
    elif [[ "$level" == "WARNING" ]]; then
        warning "$message"
        send_alert "$message" "warning"
    fi
}

send_alert() {
    local message="$1"
    local severity="${2:info}"

    # Send to Slack (if webhook configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"[${severity}] ${message}\"}" \
            2>/dev/null || true
    fi

    # Send to CloudWatch (if configured)
    if command -v aws &> /dev/null; then
        aws logs put-log-events \
            --log-group-name "/ollama/${ENVIRONMENT}" \
            --log-stream-name "alerts" \
            --log-events "timestamp=$(date +%s000),message='$message'" \
            2>/dev/null || true
    fi
}

# =============================================================================
# API HEALTH CHECKS
# =============================================================================

check_api_health() {
    log "🏥 Checking API health..."

    local start_time=$(date +%s%N)
    local response=$(curl -s -w "\n%{http_code}" "$API_ENDPOINT/api/v1/health" 2>/dev/null || echo "000")
    local end_time=$(date +%s%N)

    local http_code=$(echo "$response" | tail -n1)
    local response_time=$(( (end_time - start_time) / 1000000 ))  # Convert to ms

    if [[ "$http_code" == "200" ]]; then
        success "API is healthy (${response_time}ms)"

        if [[ $response_time -gt $API_RESPONSE_TIME_THRESHOLD ]]; then
            alert "WARNING" "API response time ${response_time}ms exceeds threshold ${API_RESPONSE_TIME_THRESHOLD}ms"
        fi

        return 0
    else
        alert "CRITICAL" "API health check failed with HTTP $http_code"
        return 1
    fi
}

check_api_endpoints() {
    log "🔗 Checking critical API endpoints..."

    local endpoints=(
        "/api/v1/models"
        "/api/v1/health"
        "/metrics"
    )

    for endpoint in "${endpoints[@]}"; do
        local response=$(curl -s -o /dev/null -w "%{http_code}" "$API_ENDPOINT$endpoint")

        if [[ "$response" == "200" ]]; then
            success "Endpoint $endpoint: OK"
        else
            alert "WARNING" "Endpoint $endpoint returned HTTP $response"
        fi
    done
}

# =============================================================================
# DATABASE HEALTH CHECKS
# =============================================================================

check_database_health() {
    log "🐘 Checking database health..."

    if command -v psql &> /dev/null; then
        # Local database check
        if psql -h localhost -U postgres -d ollama -c "SELECT 1" &>/dev/null; then
            success "Database connectivity: OK"
        else
            alert "CRITICAL" "Database connection failed"
            return 1
        fi

        # Check connection pool
        local active_connections=$(psql -h localhost -U postgres -d ollama \
            -t -c "SELECT count(*) FROM pg_stat_activity" 2>/dev/null || echo "0")

        if [[ $active_connections -gt $DB_CONNECTION_THRESHOLD ]]; then
            alert "WARNING" "Active database connections ($active_connections) exceeds threshold ($DB_CONNECTION_THRESHOLD)"
        else
            success "Database connections: $active_connections"
        fi

        # Check database size
        local db_size=$(psql -h localhost -U postgres -d ollama \
            -t -c "SELECT pg_size_pretty(pg_database_size('ollama'))" 2>/dev/null || echo "unknown")

        info "Database size: $db_size"

        return 0
    else
        warning "PostgreSQL client not found"
        return 1
    fi
}

check_redis_health() {
    log "🔴 Checking Redis health..."

    if command -v redis-cli &> /dev/null; then
        if redis-cli -h localhost ping &>/dev/null; then
            success "Redis connectivity: OK"

            local memory_usage=$(redis-cli -h localhost info memory | grep used_memory_human | cut -d':' -f2)
            info "Redis memory: $memory_usage"

            return 0
        else
            alert "WARNING" "Redis connection failed"
            return 1
        fi
    else
        warning "Redis client not found"
        return 1
    fi
}

# =============================================================================
# SYSTEM METRICS
# =============================================================================

check_system_resources() {
    log "💻 Checking system resources..."

    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}' | cut -d'.' -f1)

    if [[ $cpu_usage -gt $CPU_THRESHOLD ]]; then
        alert "WARNING" "CPU usage at ${cpu_usage}% (threshold: ${CPU_THRESHOLD}%)"
    else
        success "CPU usage: ${cpu_usage}%"
    fi

    # Memory usage
    local memory_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')

    if [[ $memory_usage -gt $MEMORY_THRESHOLD ]]; then
        alert "WARNING" "Memory usage at ${memory_usage}% (threshold: ${MEMORY_THRESHOLD}%)"
    else
        success "Memory usage: ${memory_usage}%"
    fi

    # Disk usage
    local disk_usage=$(df -h / | tail -n1 | awk '{print $(NF-1)}' | sed 's/%//')

    if [[ $disk_usage -gt $DISK_THRESHOLD ]]; then
        alert "CRITICAL" "Disk usage at ${disk_usage}% (threshold: ${DISK_THRESHOLD}%)"
    else
        success "Disk usage: ${disk_usage}%"
    fi
}

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

query_prometheus() {
    local query="$1"
    local result=$(curl -s "$PROMETHEUS_ENDPOINT/api/v1/query?query=$query" | jq '.data.result[0].value[1]' 2>/dev/null || echo "null")
    echo "$result"
}

check_prometheus_metrics() {
    log "📊 Checking Prometheus metrics..."

    if ! curl -sf "$PROMETHEUS_ENDPOINT/-/healthy" >/dev/null 2>&1; then
        warning "Prometheus is not accessible"
        return 1
    fi

    # Inference request rate
    local request_rate=$(query_prometheus 'rate(ollama_inference_requests_total[5m])')
    info "Request rate (5m): $request_rate req/s"

    # Error rate
    local error_rate=$(query_prometheus 'rate(ollama_inference_errors_total[5m])')
    if [[ "$error_rate" != "null" ]] && (( $(echo "$error_rate > 0.01" | bc -l) )); then
        alert "WARNING" "Error rate: $error_rate"
    else
        success "Error rate: $error_rate"
    fi

    # Inference latency p99
    local latency_p99=$(query_prometheus 'histogram_quantile(0.99, ollama_inference_latency_seconds_bucket)')
    if (( $(echo "$latency_p99 > 10" | bc -l 2>/dev/null || echo 0) )); then
        alert "WARNING" "Inference latency p99: ${latency_p99}s exceeds threshold"
    else
        success "Inference latency p99: ${latency_p99}s"
    fi

    # Model cache hit rate
    local cache_hit_rate=$(query_prometheus 'rate(ollama_model_cache_hits_total[5m]) / rate(ollama_model_cache_requests_total[5m])')
    info "Cache hit rate: $cache_hit_rate"

    # Active connections
    local active_connections=$(query_prometheus 'ollama_active_connections')
    success "Active connections: $active_connections"
}

# =============================================================================
# COLLECT METRICS
# =============================================================================

collect_metrics() {
    log "📈 Collecting metrics..."

    local timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

    cat > "$METRICS_FILE" << EOF
{
  "timestamp": "$timestamp",
  "environment": "$ENVIRONMENT",
  "api": {
    "endpoint": "$API_ENDPOINT",
    "status": "checking..."
  },
  "system": {
    "cpu_usage": 0,
    "memory_usage": 0,
    "disk_usage": 0
  },
  "database": {
    "status": "checking..."
  },
  "cache": {
    "status": "checking..."
  }
}
EOF

    success "Metrics collected to: $METRICS_FILE"
}

# =============================================================================
# HEALTH REPORT
# =============================================================================

generate_health_report() {
    log "📋 Generating health report..."

    local timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

    cat > "$STATUS_FILE" << EOF
{
  "timestamp": "$timestamp",
  "environment": "$ENVIRONMENT",
  "status": "checking",
  "components": {
    "api": { "status": "unknown" },
    "database": { "status": "unknown" },
    "cache": { "status": "unknown" },
    "system": { "status": "unknown" }
  },
  "alerts": [],
  "metrics": {},
  "checks": {
    "last_run": "$timestamp",
    "next_run": "pending"
  }
}
EOF

    success "Health status saved to: $STATUS_FILE"
}

# =============================================================================
# CONTINUOUS MONITORING
# =============================================================================

continuous_monitoring() {
    local interval="${1:60}"  # Default: 60 seconds

    log "🔄 Starting continuous monitoring (interval: ${interval}s)"

    while true; do
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Run all health checks
        check_api_health || true
        check_api_endpoints || true
        check_database_health || true
        check_redis_health || true
        check_system_resources || true
        check_prometheus_metrics || true

        collect_metrics
        generate_health_report

        log "Waiting ${interval}s until next check..."
        sleep "$interval"
    done
}

# =============================================================================
# ALERTS DASHBOARD
# =============================================================================

show_alerts() {
    log "🚨 Recent alerts:"

    if [[ -f "$ALERT_LOG" ]]; then
        tail -n 20 "$ALERT_LOG"
    else
        info "No alerts recorded"
    fi
}

show_health_history() {
    log "📅 Health check history:"

    if [[ -f "$HEALTH_HISTORY" ]]; then
        tail -n 50 "$HEALTH_HISTORY"
    else
        info "No history available"
    fi
}

show_dashboard() {
    clear

    cat << 'EOF'
╔════════════════════════════════════════════════════════════════════╗
║                 OLLAMA MONITORING DASHBOARD                       ║
╚════════════════════════════════════════════════════════════════════╝
EOF

    log "Environment: $ENVIRONMENT"
    log "Timestamp: $(date -u +'%Y-%m-%d %H:%M:%SZ')"
    log ""

    # API Status
    log "📡 API Status:"
    if curl -sf "$API_ENDPOINT/api/v1/health" >/dev/null; then
        echo -e "   ${GREEN}✓ Healthy${NC}"
    else
        echo -e "   ${RED}✗ Offline${NC}"
    fi

    # Database Status
    log "🐘 Database Status:"
    if psql -h localhost -U postgres -d ollama -c "SELECT 1" &>/dev/null; then
        echo -e "   ${GREEN}✓ Connected${NC}"
    else
        echo -e "   ${RED}✗ Disconnected${NC}"
    fi

    # System Resources
    log "💻 System Resources:"
    echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}' | cut -d'.' -f1)%"
    echo "   Memory: $(free | grep Mem | awk '{print int($3/$2 * 100)}')%"
    echo "   Disk: $(df -h / | tail -n1 | awk '{print $(NF-1)}')"

    # Recent Alerts
    log "🚨 Recent Alerts:"
    if [[ -f "$ALERT_LOG" ]]; then
        tail -n 5 "$ALERT_LOG" | sed 's/^/   /'
    else
        echo "   No alerts"
    fi

    log ""
    log "Last updated: $(date)"
}

# =============================================================================
# MAIN
# =============================================================================

usage() {
    cat << EOF
Usage: $0 <environment> <action> [options]

Actions:
  check              Run single health check cycle
  continuous [N]     Run continuous monitoring (N=interval in seconds)

  health             Show health status
  alerts             Show recent alerts
  dashboard          Show monitoring dashboard (auto-refresh)
  history            Show health check history

  api                Check API health
  database           Check database health
  redis              Check Redis health
  system             Check system resources
  prometheus         Check Prometheus metrics

  collect            Collect and save metrics
  report             Generate health report

Examples:
  $0 prod check                    # Single health check
  $0 prod continuous 60            # Continuous monitoring (60s interval)
  $0 prod dashboard                # Show dashboard
  $0 prod alerts                   # Show recent alerts

EOF
    exit 0
}

main() {
    if [[ "$ENVIRONMENT" == "--help" ]] || [[ "$ENVIRONMENT" == "-h" ]]; then
        usage
    fi

    local action="${2:check}"

    case "$action" in
        check)
            check_api_health || true
            check_api_endpoints || true
            check_database_health || true
            check_redis_health || true
            check_system_resources || true
            check_prometheus_metrics || true
            collect_metrics
            generate_health_report
            ;;
        continuous)
            continuous_monitoring "${3:60}"
            ;;
        health)
            generate_health_report
            cat "$STATUS_FILE"
            ;;
        alerts)
            show_alerts
            ;;
        dashboard)
            while true; do
                show_dashboard
                sleep 10
            done
            ;;
        history)
            show_health_history
            ;;
        api)
            check_api_health && check_api_endpoints
            ;;
        database)
            check_database_health
            ;;
        redis)
            check_redis_health
            ;;
        system)
            check_system_resources
            ;;
        prometheus)
            check_prometheus_metrics
            ;;
        collect)
            collect_metrics
            ;;
        report)
            generate_health_report
            ;;
        *)
            error "Unknown action: $action"
            usage
            exit 1
            ;;
    esac
}

main "$@"
