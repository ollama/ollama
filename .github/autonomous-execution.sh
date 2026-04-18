#!/bin/bash
# Autonomous Execution Pipeline
# 
# This script starts the complete autonomous execution workflow across all 4 lanes.
# It requires NO manual intervention and handles all coordination automatically.
#
# Usage:
#   bash .github/autonomous-execution.sh              # Start execution, monitor progress
#   bash .github/autonomous-execution.sh --dry-run    # Show what would execute
#   bash .github/autonomous-execution.sh --test       # Test mode (limit 5 issues/batch)

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
DRY_RUN=false
TEST_MODE=false
MONITOR_INTERVAL=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --monitor-interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "============================================================================"
echo "AUTONOMOUS EXECUTION PIPELINE - $(date -Iseconds)"
echo "============================================================================"
echo -e "${NC}"

# Validate environment
echo "Validating environment..."

# Check if Python scripts exist
if [[ ! -f scripts/orchestrate_agent_execution.py ]]; then
    echo -e "${RED}❌ orchestrate_agent_execution.py not found${NC}"
    exit 1
fi

if [[ ! -f scripts/run_autonomous_agent.py ]]; then
    echo -e "${RED}❌ run_autonomous_agent.py not found${NC}"
    exit 1
fi

# Check if workpacks exist
for shard in 1 2 3 4; do
    if [[ ! -f .github/lane_workpacks/shard_${shard}_workpack.json ]]; then
        echo -e "${RED}❌ shard_${shard}_workpack.json not found${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ Environment validated${NC}"

# Show status before execution
echo ""
echo -e "${BLUE}Current Execution Status:${NC}"
python3 scripts/agent_claim_work.py

# Show what will run
echo ""
echo -e "${BLUE}Execution Plan:${NC}"
echo "  - 4 autonomous agents (one per lane)"
echo "  - 28 total batches (294 GitHub issues)"
echo "  - Parallel execution across all lanes"
echo "  - Real-time progress tracking"

if [[ "$TEST_MODE" == "true" ]]; then
    echo "  - TEST MODE: 5 issues per batch"
fi

echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN MODE - No actual execution${NC}"
    exit 0
fi

# Start execution
echo -e "${BLUE}Starting autonomous agents...${NC}"
echo ""

# Build orchestrator command
ORCH_CMD="python3 scripts/orchestrate_agent_execution.py --start-all"
if [[ "$TEST_MODE" == "true" ]]; then
    ORCH_CMD="$ORCH_CMD --test --limit 5"
fi

# Run orchestrator
$ORCH_CMD

echo ""
echo -e "${GREEN}Autonomous execution started!${NC}"
echo ""
echo "Monitoring execution progress..."
echo "Press Ctrl+C to stop monitoring (execution will continue)"
echo ""

# Monitor progress
python3 scripts/orchestrate_agent_execution.py --monitor --interval "$MONITOR_INTERVAL"

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo "Execution Summary"
echo -e "${BLUE}============================================================================${NC}"
python3 scripts/orchestrate_agent_execution.py --report
