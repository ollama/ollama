#!/bin/bash
# Smoke tests for Ollama staging deployment
# Runs automated validation after staging deployment
# Part of Cloud Build CI/CD pipeline

set -e

echo "🧪 Running Ollama Smoke Tests..."
echo ""

# ============================================================
# Configuration
# ============================================================
API_ENDPOINT="${API_ENDPOINT:-http://ollama-api.staging.svc.cluster.local:8000}"
TIMEOUT=30
RETRIES=3

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Track failures
FAILED=0

# ============================================================
# Helper Functions
# ============================================================
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local expected_code="$4"
    local data="$5"

    echo -n "📋 Testing $name ... "

    local attempt=0
    local response_code
    local response_body

    while [ $attempt -lt $RETRIES ]; do
        if [ "$method" = "GET" ]; then
            response=$(curl -s -w "\n%{http_code}" -X GET "$API_ENDPOINT$endpoint" \
                --max-time $TIMEOUT --connect-timeout 5 2>/dev/null || echo "000")
        else
            response=$(curl -s -w "\n%{http_code}" -X "$method" "$API_ENDPOINT$endpoint" \
                -H "Content-Type: application/json" \
                -d "$data" \
                --max-time $TIMEOUT --connect-timeout 5 2>/dev/null || echo "000")
        fi

        response_code=$(echo "$response" | tail -1)
        response_body=$(echo "$response" | head -n-1)

        if [ "$response_code" = "$expected_code" ]; then
            echo -e "${GREEN}✅${NC} (HTTP $response_code)"
            return 0
        fi

        attempt=$((attempt + 1))
        if [ $attempt -lt $RETRIES ]; then
            echo -n "(retry $attempt/$RETRIES) ... "
            sleep 2
        fi
    done

    echo -e "${RED}❌${NC} (Expected: $expected_code, Got: $response_code)"
    echo "   Response: $response_body"
    FAILED=$((FAILED + 1))
    return 1
}

measure_latency() {
    local name="$1"
    local endpoint="$2"
    local max_latency_ms="$3"

    echo -n "⏱️  Testing $name latency ... "

    # Run 5 requests and measure
    local total_time=0
    local count=0

    for i in {1..5}; do
        local start=$(date +%s%N)
        curl -s -X GET "$API_ENDPOINT$endpoint" \
            -H "Accept: application/json" \
            --max-time $TIMEOUT --connect-timeout 5 > /dev/null 2>&1 || true
        local end=$(date +%s%N)
        local elapsed=$(( (end - start) / 1000000 ))  # Convert to ms
        total_time=$((total_time + elapsed))
        count=$((count + 1))
    done

    local avg_latency=$(( total_time / count ))

    if [ $avg_latency -lt "$max_latency_ms" ]; then
        echo -e "${GREEN}✅${NC} ($avg_latency ms avg, target: < $max_latency_ms ms)"
        return 0
    else
        echo -e "${YELLOW}⚠️ ${NC} ($avg_latency ms avg, target: < $max_latency_ms ms)"
        # Don't fail on latency, just warn
        return 0
    fi
}

# ============================================================
# Test Scenarios
# ============================================================
echo "════════════════════════════════════════════════════════════"
echo "Health & Availability Tests"
echo "════════════════════════════════════════════════════════════"
echo ""

# Test 1: Health Check
test_endpoint "Health Check" "GET" "/api/v1/health" "200" || true

# Test 2: Model List
test_endpoint "Model Availability" "GET" "/api/v1/models" "200" || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Functionality Tests"
echo "════════════════════════════════════════════════════════════"
echo ""

# Test 3: Text Generation (basic)
GENERATE_PAYLOAD='{"prompt":"Hello world","model":"ollama","max_tokens":10}'
test_endpoint "Text Generation" "POST" "/api/v1/generate" "200" "$GENERATE_PAYLOAD" || true

# Test 4: Chat Completion
CHAT_PAYLOAD='{"messages":[{"role":"user","content":"Hello"}],"model":"ollama"}'
test_endpoint "Chat Completion" "POST" "/api/v1/chat" "200" "$CHAT_PAYLOAD" || true

# Test 5: Embeddings
EMBEDDINGS_PAYLOAD='{"text":"Hello world","model":"ollama"}'
test_endpoint "Embeddings" "POST" "/api/v1/embeddings" "200" "$EMBEDDINGS_PAYLOAD" || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Performance Tests"
echo "════════════════════════════════════════════════════════════"
echo ""

# Test 6: Health Check Latency
measure_latency "Health Check" "/api/v1/health" "100" || true

# Test 7: Model List Latency
measure_latency "Model List" "/api/v1/models" "200" || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Database Connectivity Tests"
echo "════════════════════════════════════════════════════════════"
echo ""

# Test 8: Database connectivity (if applicable)
echo -n "📋 Testing Database Connectivity ... "
if command -v psql &> /dev/null; then
    if psql -h postgres -U ollama -d ollama -c "SELECT 1" &>/dev/null; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${YELLOW}⚠️  (Database not accessible, may be expected)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  (psql not installed, skipping)${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Metrics & Monitoring"
echo "════════════════════════════════════════════════════════════"
echo ""

# Test 9: Prometheus metrics endpoint
test_endpoint "Prometheus Metrics" "GET" "/metrics" "200" || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Summary"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All smoke tests passed!${NC}"
    echo ""
    echo "Deployment Status: HEALTHY"
    echo "Ready for production canary deployment"
    exit 0
else
    echo -e "${RED}❌ $FAILED test(s) failed!${NC}"
    echo ""
    echo "Deployment Status: UNHEALTHY"
    echo "Investigate failures before proceeding to production"
    exit 1
fi
