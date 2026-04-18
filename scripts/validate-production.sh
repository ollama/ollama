#!/bin/bash
#===============================================================================
# Production Validation Script
# Tests all Ollama API endpoints and infrastructure services
#===============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-11000}"
OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
OLLAMA_PORT="${OLLAMA_PORT:-8000}"
BASE_URL="http://${API_HOST}:${API_PORT}"
OLLAMA_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"

# Counters
PASSED=0
FAILED=0
TOTAL=0

#===============================================================================
# Helper Functions
#===============================================================================

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

print_test() {
    ((TOTAL++))
    echo -e "${YELLOW}[TEST $TOTAL]${NC} $1"
}

print_pass() {
    ((PASSED++))
    echo -e "  ${GREEN}✓ PASS${NC} - $1"
}

print_fail() {
    ((FAILED++))
    echo -e "  ${RED}✗ FAIL${NC} - $1"
}

print_info() {
    echo -e "  ${BLUE}ℹ INFO${NC} - $1"
}

test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_code="${3:-200}"
    local method="${4:-GET}"
    local data="${5:-}"
    
    print_test "$name"
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" 2>&1)
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "$expected_code" ]; then
        print_pass "HTTP $http_code (expected $expected_code)"
        echo "$body"
        return 0
    else
        print_fail "HTTP $http_code (expected $expected_code)"
        echo "$body" | head -5
        return 1
    fi
}

#===============================================================================
# Infrastructure Tests
#===============================================================================

print_header "INFRASTRUCTURE HEALTH CHECKS"

# PostgreSQL
print_test "PostgreSQL Container"
if docker ps | grep -q ollama-postgres; then
    print_pass "Container running"
    if docker exec ollama-postgres pg_isready -U ollama > /dev/null 2>&1; then
        print_pass "Database accepting connections"
    else
        print_fail "Database not ready"
    fi
else
    print_fail "Container not running"
fi

# Redis
print_test "Redis Container"
if docker ps | grep -q ollama-redis; then
    print_pass "Container running"
    if docker exec ollama-redis redis-cli ping | grep -q PONG; then
        print_pass "Redis responding to PING"
    else
        print_fail "Redis not responding"
    fi
else
    print_fail "Container not running"
fi

# Qdrant
print_test "Qdrant Container"
if docker ps | grep -q ollama-qdrant; then
    print_pass "Container running"
    if curl -sf http://localhost:6333/health > /dev/null 2>&1; then
        print_pass "Qdrant health check OK"
    else
        print_fail "Qdrant health check failed"
    fi
else
    print_fail "Container not running"
fi

# Prometheus
print_test "Prometheus Container"
if docker ps | grep -q ollama-prometheus; then
    print_pass "Container running"
    if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_pass "Prometheus healthy"
    else
        print_fail "Prometheus not healthy"
    fi
else
    print_fail "Container not running"
fi

# Grafana
print_test "Grafana Container"
if docker ps | grep -q ollama-grafana; then
    print_pass "Container running"
    if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
        print_pass "Grafana API responding"
    else
        print_fail "Grafana not responding"
    fi
else
    print_fail "Container not running"
fi

# Jaeger
print_test "Jaeger Container"
if docker ps | grep -q ollama-jaeger; then
    print_pass "Container running"
else
    print_fail "Container not running"
fi

#===============================================================================
# Ollama Server Tests
#===============================================================================

print_header "OLLAMA SERVER TESTS"

# Version
test_endpoint "Ollama Version" "$OLLAMA_URL/api/version"

# Models List
print_test "Ollama Models List"
models_response=$(curl -s "$OLLAMA_URL/api/tags")
model_count=$(echo "$models_response" | jq -r '.models | length' 2>/dev/null || echo "0")
if [ "$model_count" -gt 0 ]; then
    print_pass "Found $model_count models"
    echo "$models_response" | jq -r '.models[] | "  - \(.name) (\(.size) bytes)"' 2>/dev/null
else
    print_fail "No models found"
fi

#===============================================================================
# FastAPI Application Tests
#===============================================================================

print_header "FASTAPI APPLICATION TESTS"

# Health Check
test_endpoint "Health Check" "$BASE_URL/health"

# Root Endpoint
test_endpoint "Root Info" "$BASE_URL/"

# API Models List
print_test "API Models Endpoint"
api_models=$(curl -s "$BASE_URL/api/v1/models")
api_model_count=$(echo "$api_models" | jq -r '.models | length' 2>/dev/null || echo "0")
if [ "$api_model_count" -gt 0 ]; then
    print_pass "API returning $api_model_count models"
    echo "$api_models" | jq -r '.models[] | "  - \(.name) (\(.size))"' 2>/dev/null
else
    print_fail "API models endpoint failed"
    echo "$api_models"
fi

# Prometheus Metrics
print_test "Prometheus Metrics Endpoint"
metrics=$(curl -s "$BASE_URL/metrics")
if echo "$metrics" | grep -q "process_cpu_seconds_total"; then
    print_pass "Metrics endpoint responding"
    metric_count=$(echo "$metrics" | grep -c "^# HELP" || echo "0")
    print_info "Exporting $metric_count metrics"
else
    print_fail "Metrics endpoint not working"
fi

#===============================================================================
# AI Inference Tests
#===============================================================================

print_header "AI INFERENCE TESTS"

# Get first available model
FIRST_MODEL=$(curl -s "$OLLAMA_URL/api/tags" | jq -r '.models[0].name' 2>/dev/null || echo "codellama:7b")
print_info "Using model: $FIRST_MODEL"

# Text Generation
print_test "Text Generation"
gen_data="{\"model\": \"$FIRST_MODEL\", \"prompt\": \"What is 2+2? Answer in one word:\"}"
gen_response=$(curl -s -X POST "$BASE_URL/api/v1/generate" \
    -H "Content-Type: application/json" \
    -d "$gen_data")

gen_text=$(echo "$gen_response" | jq -r '.response' 2>/dev/null)
if [ -n "$gen_text" ] && [ "$gen_text" != "null" ]; then
    print_pass "Generation successful"
    echo "  Response: $gen_text" | head -c 200
    echo
else
    print_fail "Generation failed"
    echo "$gen_response" | jq . 2>/dev/null || echo "$gen_response"
fi

# Chat Completion
print_test "Chat Completion"
chat_data="{\"model\": \"$FIRST_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word\"}]}"
chat_response=$(curl -s -X POST "$BASE_URL/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d "$chat_data")

chat_text=$(echo "$chat_response" | jq -r '.message.content' 2>/dev/null)
if [ -n "$chat_text" ] && [ "$chat_text" != "null" ]; then
    print_pass "Chat completion successful"
    echo "  Response: $chat_text"
else
    print_fail "Chat completion failed"
    echo "$chat_response" | jq . 2>/dev/null || echo "$chat_response"
fi

#===============================================================================
# Performance Tests
#===============================================================================

print_header "PERFORMANCE TESTS"

# Response Time Test
print_test "API Response Time"
start_time=$(date +%s%N)
curl -s "$BASE_URL/health" > /dev/null
end_time=$(date +%s%N)
duration=$(( (end_time - start_time) / 1000000 ))
print_info "Health endpoint: ${duration}ms"

if [ $duration -lt 100 ]; then
    print_pass "Response time excellent (<100ms)"
elif [ $duration -lt 500 ]; then
    print_pass "Response time good (<500ms)"
else
    print_fail "Response time slow (>${duration}ms)"
fi

# GPU Check
print_test "GPU Utilization"
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$gpu_info" ]; then
        print_pass "GPU detected"
        echo "  $gpu_info"
    else
        print_fail "nvidia-smi failed"
    fi
else
    print_info "nvidia-smi not available (CPU-only mode)"
fi

#===============================================================================
# Summary
#===============================================================================

print_header "TEST SUMMARY"

echo -e "Total Tests:  ${BLUE}$TOTAL${NC}"
echo -e "Passed:       ${GREEN}$PASSED${NC}"
echo -e "Failed:       ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✓ ALL TESTS PASSED${NC} - System ready for production! 🚀\n"
    exit 0
else
    echo -e "\n${RED}✗ SOME TESTS FAILED${NC} - Please review failures above\n"
    exit 1
fi
