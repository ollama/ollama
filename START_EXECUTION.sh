#!/bin/bash
# 
# START AUTONOMOUS TRIAGE EXECUTION NOW
#
# This is the official entry point to begin the complete autonomous triage
# workflow across all 294 GitHub issues.
#
# Usage:
#   bash START_EXECUTION.sh                    # Interactive
#   bash START_EXECUTION.sh --no-wait          # Start in background
#   bash START_EXECUTION.sh --test              # Test mode first
#   bash START_EXECUTION.sh --validate         # Check environment only

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
TEST_MODE=false
VALIDATE_ONLY=false
NO_WAIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test) TEST_MODE=true; shift ;;
        --validate) VALIDATE_ONLY=true; shift ;;
        --no-wait) NO_WAIT=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} AUTONOMOUS TRIAGE EXECUTION LAUNCHER"
echo -e "${BLUE}║${NC} Status: APPROVED AND READY"
echo -e "${BLUE}║${NC} Issues: 294 real | Batches: 28 | Lanes: 4 (parallel)"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Validate environment
echo -e "${CYAN}▶ Checking environment...${NC}"
errors=0

for file in scripts/orchestrate_agent_execution.py scripts/run_autonomous_agent.py scripts/agent_claim_work.py; do
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file NOT FOUND"
        ((errors++))
    fi
done

for shard in 1 2 3 4; do
    if [[ -f ".github/lane_workpacks/shard_${shard}_workpack.json" ]]; then
        echo -e "${GREEN}✓${NC} .github/lane_workpacks/shard_${shard}_workpack.json"
    else
        echo -e "${RED}✗${NC} shard_${shard}_workpack.json NOT FOUND"
        ((errors++))
    fi
done

if [[ -f ".github/agent_execution_progress.json" ]]; then
    echo -e "${GREEN}✓${NC} .github/agent_execution_progress.json"
else
    echo -e "${RED}✗${NC} Progress tracking file NOT FOUND"
    ((errors++))
fi

echo ""

if [[ $errors -gt 0 ]]; then
    echo -e "${RED}✗ Environment validation failed ($errors errors)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment validated${NC}"
echo ""

if [[ "$VALIDATE_ONLY" == "true" ]]; then
    exit 0
fi

# Show current status
echo -e "${CYAN}▶ Current Execution Status:${NC}"
python3 scripts/agent_claim_work.py
echo ""

# Execution options
echo -e "${CYAN}▶ Execution Options:${NC}"
echo "  Mode: $(if [[ "$TEST_MODE" == "true" ]]; then echo "TEST (5 issues/batch)"; else echo "FULL (all 294 issues)"; fi)"
echo "  Lanes: 4 (parallel)"
echo "  Batches: 28"

echo ""

if [[ "$TEST_MODE" == "true" ]]; then
    echo -e "${YELLOW}⚠ TEST MODE${NC}: This will execute 5 issues per batch for validation"
else
    echo -e "${CYAN}ℹ FULL EXECUTION${NC}: This will execute all 294 issues across 28 batches"
fi

# Confirmation
if [[ "$NO_WAIT" != "true" ]]; then
    echo ""
    read -p "Continue? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo -e "${YELLOW}Execution cancelled${NC}"
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} LAUNCHING AUTONOMOUS EXECUTION"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Build command
CMD="python3 scripts/orchestrate_agent_execution.py --start-all"

if [[ "$TEST_MODE" == "true" ]]; then
    CMD="$CMD --test --limit 5"
fi

# Execute
echo -e "${CYAN}▶ Starting agents...${NC}"
echo ""

if [[ "$NO_WAIT" == "true" ]]; then
    # Background mode
    log_file="execution_$(date +%s).log"
    nohup $CMD > "$log_file" 2>&1 &
    pid=$!
    echo -e "${GREEN}✓${NC} Execution started (PID: $pid)"
    echo -e "${GREEN}✓${NC} Log file: $log_file"
    echo ""
    echo "Monitor with: tail -f $log_file"
else
    # Foreground mode with monitoring
    $CMD
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} EXECUTION MONITORING"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Monitoring progress (press Ctrl+C to stop)...${NC}"
    echo ""
    
    python3 scripts/orchestrate_agent_execution.py --monitor --interval 10
fi

echo ""
echo -e "${CYAN}▶ Final Status:${NC}"
python3 scripts/orchestrate_agent_execution.py --status

echo ""
echo -e "${GREEN}✓ Autonomous triage execution launched!${NC}"
