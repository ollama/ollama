#!/bin/bash
# Master Triage Execution Control
#
# This is the main entry point for beginning the complete autonomous triage
# execution across all 294 GitHub issues.
#
# It provides interactive control over the triage workflow execution with
# options for monitoring, testing, and recovery.

set -e

# Create standard colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

show_menu() {
    echo ""
    echo -e "${CYAN}TRIAGE EXECUTION MENU${NC}"
    echo "  1) Start Autonomous Execution (All 4 lanes)"
    echo "  2) Monitor Progress"
    echo "  3) Show Execution Report"
    echo "  4) Test Mode (5 issues per batch)"
    echo "  5) View Current Status"
    echo "  6) Dry Run (Show what would execute)"
    echo "  7) Exit"
    echo ""
    read -p "Select option (1-7): " choice
}

validate_environment() {
    print_section "Validating Environment"
    
    local errors=0
    
    # Check required files
    if [[ ! -f scripts/orchestrate_agent_execution.py ]]; then
        print_error "orchestrate_agent_execution.py not found"
        ((errors++))
    else
        print_success "orchestrate_agent_execution.py found"
    fi
    
    if [[ ! -f scripts/run_autonomous_agent.py ]]; then
        print_error "run_autonomous_agent.py not found"
        ((errors++))
    else
        print_success "run_autonomous_agent.py found"
    fi
    
    if [[ ! -d .github/lane_workpacks ]]; then
        print_error "lane_workpacks directory not found"
        ((errors++))
    else
        local workpack_count=$(find .github/lane_workpacks -name "shard_*.json" | wc -l)
        if [[ $workpack_count -eq 4 ]]; then
            print_success "All 4 workpack files found"
        else
            print_error "Expected 4 workpack files, found $workpack_count"
            ((errors++))
        fi
    fi
    
    if [[ ! -f .github/agent_execution_progress.json ]]; then
        print_error "agent_execution_progress.json not found"
        ((errors++))
    else
        print_success "Progress tracking file found"
    fi
    
    if [[ ! -f scripts/agent_claim_work.py ]]; then
        print_error "agent_claim_work.py not found"
        ((errors++))
    else
        print_success "Agent claim tool found"
    fi
    
    if [[ $errors -gt 0 ]]; then
        print_error "Environment validation failed with $errors error(s)"
        return 1
    else
        print_success "Environment validation passed"
        return 0
    fi
}

show_execution_plan() {
    print_section "Execution Plan"
    echo ""
    echo "  Total Issues:         294"
    echo "  Total Batches:        28"
    echo "  Batch Size:           ~12 issues"
    echo "  Parallel Lanes:       4 (shard_1, shard_2, shard_3, shard_4)"
    echo "  Issues per Lane:      ~74"
    echo "  Batches per Lane:     7"
    echo ""
    echo "  Estimated Timing (with parallelism):"
    echo "    Per Issue:          20-35 minutes"
    echo "    Per Batch:          4-7 hours"
    echo "    Per Lane:           28-49 hours"
    echo "    All Lanes:          ~28-49 hours (parallel)"
    echo ""
    echo "  Execution Model:"
    echo "    - 4 autonomous agents start simultaneously"
    echo "    - Each claims batches from their assigned lane"
    echo "    - Full issue lifecycle per agent"
    echo "    - Real-time progress tracking"
    echo "    - Automatic error handling and recovery"
    echo ""
}

start_execution() {
    print_header "STARTING AUTONOMOUS TRIAGE EXECUTION"
    
    validate_environment || return 1
    
    show_execution_plan
    
    echo -e "${YELLOW}This will start autonomous execution across all 4 lanes.${NC}"
    echo -e "${YELLOW}Process will run in background. You can monitor with option 2.${NC}"
    echo ""
    read -p "Are you sure? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        print_info "Execution cancelled"
        return 0
    fi
    
    print_info "Starting orchestrator in background..."
    
    # Create a log file for this execution
    local log_file="execution_log_$(date +%s).txt"
    
    # Start the orchestrator in background
    nohup python3 scripts/orchestrate_agent_execution.py --start-all \
        > "$log_file" 2>&1 &
    
    local pid=$!
    print_success "Orchestrator started (PID: $pid)"
    print_info "Log file: $log_file"
    print_info "Monitor progress with option 2 or: tail -f $log_file"
}

monitor_execution() {
    print_header "MONITORING EXECUTION"
    
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring (execution continues)${NC}"
    echo ""
    
    python3 scripts/orchestrate_agent_execution.py --monitor --interval 10
}

show_report() {
    print_header "EXECUTION REPORT"
    python3 scripts/orchestrate_agent_execution.py --report
}

test_execution() {
    print_header "TEST MODE EXECUTION"
    
    echo -e "${YELLOW}This will execute 5 issues per batch (test mode).${NC}"
    echo -e "${YELLOW}Useful for validating the workflow end-to-end.${NC}"
    echo ""
    read -p "Start test execution? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        print_info "Test execution cancelled"
        return 0
    fi
    
    validate_environment || return 1
    
    print_info "Starting test execution..."
    python3 scripts/orchestrate_agent_execution.py --start-all --test --limit 5
}

show_status() {
    print_header "CURRENT EXECUTION STATUS"
    python3 scripts/agent_claim_work.py
}

dry_run() {
    print_header "DRY RUN - EXECUTION PLAN"
    bash .github/autonomous-execution.sh --dry-run
}

main_interactive() {
    while true; do
        print_header "TRIAGE EXECUTION CONTROL CENTER"
        
        show_menu
        
        case $choice in
            1)
                start_execution
                ;;
            2)
                monitor_execution
                ;;
            3)
                show_report
                ;;
            4)
                test_execution
                ;;
            5)
                show_status
                ;;
            6)
                dry_run
                ;;
            7)
                print_info "Exiting"
                exit 0
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# Parse command line arguments for non-interactive mode
if [[ $# -gt 0 ]]; then
    case "$1" in
        start)
            validate_environment || exit 1
            start_execution
            ;;
        monitor)
            monitor_execution
            ;;
        report)
            show_report
            ;;
        test)
            test_execution
            ;;
        status)
            show_status
            ;;
        validate)
            validate_environment
            ;;
        dry-run)
            dry_run
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Usage: $0 [start|monitor|report|test|status|validate|dry-run]"
            exit 1
            ;;
    esac
else
    # Interactive mode if no arguments
    main_interactive
fi
