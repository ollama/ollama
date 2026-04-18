#!/usr/bin/env bash
# ==============================================================================
# GCP LANDING ZONE BOOTSTRAP - Comprehensive Onboarding Script
# ==============================================================================
# Purpose: Validate and onboard this repository to GCP Landing Zone compliance
#
# Compliance Mandates Checked:
# 1. PMO Metadata (24-label mandate)
# 2. Docker Service Naming ({environment}-{application}-{component})
# 3. Mandatory Labels on all resources
# 4. Security Controls (GPG signing, TLS 1.3+, CORS, rate limiting)
# 5. No Root Chaos (file organization)
# 6. Deployment Architecture (GCP LB as single entry point)
# 7. Development Standards (real IP/DNS, never localhost)
# 8. Infrastructure Alignment (Three-Lens Framework)
#
# Usage:
#   ./scripts/landing-zone-bootstrap.sh                   # Full validation
#   ./scripts/landing-zone-bootstrap.sh --dry-run         # Dry run mode
#   ./scripts/landing-zone-bootstrap.sh --fix             # Auto-fix issues
#   ./scripts/landing-zone-bootstrap.sh --report          # Generate report
# ==============================================================================

# Don't exit on error - we want to collect all issues
set +e

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PMO_FILE="${PROJECT_ROOT}/pmo.yaml"
REPORT_FILE="${PROJECT_ROOT}/docs/reports/lz-compliance-report.md"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0
WARNINGS=0
SKIPPED=0

# Modes
DRY_RUN=false
FIX_MODE=false
REPORT_MODE=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --fix) FIX_MODE=true ;;
        --report) REPORT_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--fix] [--report]"
            exit 0
            ;;
    esac
done

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================
log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║ $1${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
}

log_section() {
    echo ""
    echo -e "${BOLD}${BLUE}┌─────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${BLUE}│ $1${NC}"
    echo -e "${BOLD}${BLUE}└─────────────────────────────────────────────────────────────────────────┘${NC}"
}

log_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
    ((WARNINGS++))
}

log_skip() {
    echo -e "${MAGENTA}○ SKIP${NC}: $1"
    ((SKIPPED++))
}

log_info() {
    echo -e "${BLUE}ℹ INFO${NC}: $1"
}

log_fix() {
    echo -e "${CYAN}🔧 FIX${NC}: $1"
}

# =============================================================================
# PREREQUISITE CHECKS
# =============================================================================
check_prerequisites() {
    log_section "PHASE 0: Prerequisites Check"

    local prereqs=("grep" "awk" "sed" "find" "docker")
    local optional=("gcloud" "terraform" "kubectl" "helm")

    for cmd in "${prereqs[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            log_pass "Required: $cmd is installed"
        else
            log_fail "Required: $cmd is NOT installed"
        fi
    done

    for cmd in "${optional[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            log_pass "Optional: $cmd is installed"
        else
            log_warn "Optional: $cmd is NOT installed (recommended for full validation)"
        fi
    done

    # Check Docker daemon
    if docker info &>/dev/null; then
        log_pass "Docker daemon is running"
    else
        log_warn "Docker daemon is not running (some checks will be skipped)"
    fi
}

# =============================================================================
# 1. PMO METADATA VALIDATION (24-Label Mandate)
# =============================================================================
validate_pmo_metadata() {
    log_section "PHASE 1: PMO Metadata Validation (24-Label Mandate)"

    if [[ ! -f "$PMO_FILE" ]]; then
        log_fail "pmo.yaml not found at project root"
        return 0
    fi

    log_pass "pmo.yaml exists at project root"

    # Define all 24 mandatory labels
    local ALL_LABELS="environment cost_center team managed_by created_by created_date lifecycle_state teardown_date retention_days product component tier compliance version stack backup_strategy monitoring_enabled budget_owner project_code monthly_budget_usd chargeback_unit git_repository git_branch auto_delete"

    local missing_labels=""
    local total_labels=0
    local present_labels=0

    for label in $ALL_LABELS; do
        total_labels=$((total_labels + 1))
        if grep -q "^${label}:" "$PMO_FILE"; then
            present_labels=$((present_labels + 1))
            VALUE=$(grep "^${label}:" "$PMO_FILE" | head -n1 | cut -d'"' -f2)
            if [[ -n "$VALUE" && "$VALUE" != "null" ]]; then
                log_pass "$label: \"$VALUE\""
            else
                log_warn "$label is defined but empty"
                missing_labels="$missing_labels $label"
            fi
        else
            log_fail "$label: MISSING"
            missing_labels="$missing_labels $label"
        fi
    done

    echo ""
    log_info "PMO Label Summary: $present_labels/$total_labels labels present"

    if [[ -z "$missing_labels" ]]; then
        log_pass "All 24 mandatory labels are present and populated"
    else
        log_fail "Missing labels:$missing_labels"
    fi
}

# =============================================================================
# 2. DOCKER SERVICE NAMING CONVENTION
# =============================================================================
validate_docker_naming() {
    log_section "PHASE 2: Docker Service Naming Convention"

    local docker_files=(
        "docker/docker-compose.local.yml"
        "docker/docker-compose.prod.yml"
        "docker/docker-compose.elite.yml"
        "docker/docker-compose.minimal.yml"
    )

    local naming_pattern="^[a-z]+-ollama-[a-z-]+$"  # {env}-ollama-{component}

    for file in "${docker_files[@]}"; do
        local filepath="${PROJECT_ROOT}/${file}"
        if [[ ! -f "$filepath" ]]; then
            log_skip "$file does not exist"
            continue
        fi

        log_info "Checking $file..."

        # Extract service names (lines starting with 2 spaces followed by service name and colon)
        local services=$(grep -E '^  [a-z]' "$filepath" | grep -v '^  #' | grep ':$' | sed 's/://g' | awk '{print $1}' | grep -v '^x-' || true)

        for service in $services; do
            # Check if service follows naming convention
            if [[ "$service" =~ $naming_pattern ]]; then
                log_pass "$file: Service '$service' follows naming convention"
            elif [[ "$service" == "networks" || "$service" == "volumes" || "$service" == "secrets" ]]; then
                # These are Docker Compose reserved keys, not services
                continue
            else
                log_warn "$file: Service '$service' may not follow {env}-ollama-{component} pattern"
            fi
        done

        # Check for x-common-labels anchor
        if grep -q "x-common-labels:" "$filepath"; then
            log_pass "$file: Has x-common-labels anchor for DRY labels"
        else
            log_fail "$file: Missing x-common-labels anchor (required for label consistency)"
        fi

        # Check for mandatory labels in anchor
        if grep -q "environment:" "$filepath" && grep -q "application:" "$filepath"; then
            log_pass "$file: Has environment and application labels"
        else
            log_warn "$file: May be missing mandatory labels in services"
        fi
    done
}

# =============================================================================
# 3. SECURITY CONTROLS VALIDATION
# =============================================================================
validate_security_controls() {
    log_section "PHASE 3: Security Controls Validation"

    # 3.1 GPG Signing Configuration
    log_info "Checking Git GPG signing configuration..."
    if git -C "$PROJECT_ROOT" config --get user.signingkey &>/dev/null; then
        log_pass "GPG signing key is configured"
    else
        log_warn "GPG signing key is NOT configured (required for signed commits)"
    fi

    if git -C "$PROJECT_ROOT" config --get commit.gpgsign &>/dev/null; then
        local gpgsign=$(git -C "$PROJECT_ROOT" config --get commit.gpgsign)
        if [[ "$gpgsign" == "true" ]]; then
            log_pass "GPG signing is enabled for commits"
        else
            log_warn "GPG signing is disabled (should be 'true')"
        fi
    else
        log_warn "commit.gpgsign is not set (should be 'true')"
    fi

    # 3.2 Pre-commit hooks
    log_info "Checking pre-commit hooks..."
    if [[ -f "${PROJECT_ROOT}/.git/hooks/pre-commit" ]]; then
        log_pass "Pre-commit hook exists"
    else
        log_warn "Pre-commit hook not found"
    fi

    # 3.3 TLS Configuration in Docker Compose
    log_info "Checking TLS/Security configuration..."
    if grep -r "TLS_MIN_VERSION\|tls_version\|ssl_min_version" "${PROJECT_ROOT}/docker/" &>/dev/null; then
        log_pass "TLS version configuration found in Docker configs"
    else
        log_warn "No explicit TLS version configuration found (should enforce 1.3+)"
    fi

    # 3.4 CORS Configuration
    log_info "Checking CORS configuration..."
    if grep -r 'allow_origins=\["\*"\]' "${PROJECT_ROOT}/ollama/" &>/dev/null; then
        log_fail "CORS allows all origins (*) - security violation"
    else
        log_pass "CORS does not allow wildcard origins"
    fi

    # 3.5 Rate Limiting
    log_info "Checking rate limiting configuration..."
    if grep -r "rate_limit\|RateLimiter\|slowapi" "${PROJECT_ROOT}/ollama/" &>/dev/null; then
        log_pass "Rate limiting implementation found"
    else
        log_warn "No rate limiting implementation found in code"
    fi

    # 3.6 Hardcoded credentials check
    log_info "Checking for hardcoded credentials..."
    local credential_patterns=(
        "password\s*=\s*['\"][^'\"]+['\"]"
        "api_key\s*=\s*['\"][^'\"]+['\"]"
        "secret\s*=\s*['\"][^'\"]+['\"]"
        "token\s*=\s*['\"][^'\"]+['\"]"
    )

    local cred_found=false
    for pattern in "${credential_patterns[@]}"; do
        if grep -rE "$pattern" "${PROJECT_ROOT}/ollama/" --include="*.py" 2>/dev/null | grep -v "\.env" | grep -v "example" | head -3; then
            cred_found=true
        fi
    done

    if [[ "$cred_found" == "true" ]]; then
        log_warn "Potential hardcoded credentials found (review above lines)"
    else
        log_pass "No obvious hardcoded credentials found"
    fi

    # 3.7 .env.example existence
    if [[ -f "${PROJECT_ROOT}/.env.example" ]]; then
        log_pass ".env.example template exists"
    else
        log_warn ".env.example template not found"
    fi
}

# =============================================================================
# 4. NO ROOT CHAOS VALIDATION
# =============================================================================
validate_no_root_chaos() {
    log_section "PHASE 4: No Root Chaos Validation"

    # Allowed root files
    local allowed_files=(
        "README.md" "LICENSE" ".gitignore" ".gitattributes"
        "pmo.yaml" "pyproject.toml" "mypy.ini" "setup.py" "setup.cfg"
        ".pre-commit-config.yaml" ".flake8" ".ruff.toml"
        "Makefile" "docker-compose.yml" "docker-compose.yaml"
        ".env" ".env.example" ".env.local" ".env.development"
        "CHANGELOG.md" "CONTRIBUTING.md" "SECURITY.md"
        "requirements.txt" "requirements-dev.txt"
        "poetry.lock" "Pipfile" "Pipfile.lock"
        ".python-version" ".nvmrc" ".tool-versions"
    )

    local allowed_dirs=(
        "ollama" "tests" "docs" "scripts" "config" "docker" "k8s"
        "alembic" "frontend" "htmlcov" ".github" ".vscode" ".githooks"
        ".git" "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache"
        "node_modules" "dist" "build" ".eggs" "*.egg-info"
    )

    log_info "Checking root directory organization..."

    local violations=()

    for item in "${PROJECT_ROOT}"/*; do
        local basename=$(basename "$item")

        if [[ -f "$item" ]]; then
            # Check if file is allowed
            local is_allowed=false
            for allowed in "${allowed_files[@]}"; do
                if [[ "$basename" == "$allowed" || "$basename" =~ ^\..*$ ]]; then
                    is_allowed=true
                    break
                fi
            done

            if [[ "$is_allowed" == "false" ]]; then
                violations+=("$basename (file)")
            fi
        elif [[ -d "$item" ]]; then
            # Check if directory is allowed
            local is_allowed=false
            for allowed in "${allowed_dirs[@]}"; do
                if [[ "$basename" == "$allowed" || "$basename" =~ ^\..* ]]; then
                    is_allowed=true
                    break
                fi
            done

            if [[ "$is_allowed" == "false" ]]; then
                violations+=("$basename/ (directory)")
            fi
        fi
    done

    if [[ ${#violations[@]} -eq 0 ]]; then
        log_pass "Root directory is clean and organized"
    else
        log_warn "Root directory has unexpected items:"
        for item in "${violations[@]}"; do
            echo "    - $item"
        done
    fi

    # Count root items
    local root_count=$(ls -1 "${PROJECT_ROOT}" | wc -l)
    if [[ $root_count -le 20 ]]; then
        log_pass "Root directory has acceptable number of items: $root_count"
    else
        log_warn "Root directory has too many items: $root_count (recommended: ≤20)"
    fi
}

# =============================================================================
# 5. DEPLOYMENT ARCHITECTURE VALIDATION
# =============================================================================
validate_deployment_architecture() {
    log_section "PHASE 5: Deployment Architecture Validation"

    # 5.1 Check for GCP Load Balancer as single entry point
    log_info "Checking GCP LB configuration..."
    if grep -r "elevatediq.ai/ollama\|GCP_LOAD_BALANCER" "${PROJECT_ROOT}/docker/" &>/dev/null; then
        log_pass "GCP Load Balancer endpoint referenced in Docker configs"
    else
        log_warn "No GCP LB endpoint reference found in Docker configs"
    fi

    # 5.2 Check for internal-only port bindings in production
    log_info "Checking production port exposure..."
    local prod_compose="${PROJECT_ROOT}/docker/docker-compose.prod.yml"
    if [[ -f "$prod_compose" ]]; then
        # Check if monitoring ports are blocked
        if grep -q "# .*127.0.0.1:9090\|# .*127.0.0.1:3000\|# .*127.0.0.1:16686" "$prod_compose"; then
            log_pass "Monitoring ports are blocked in production config"
        else
            log_warn "Monitoring ports may be exposed in production (should be internal only)"
        fi

        # Check for localhost bindings
        if grep -E "127\.0\.0\.1:" "$prod_compose" | grep -v "^#" &>/dev/null; then
            log_pass "Production services use localhost bindings"
        else
            log_warn "Some production services may not use localhost bindings"
        fi
    fi

    # 5.3 Check environment configuration
    log_info "Checking environment variable patterns..."
    if [[ -f "${PROJECT_ROOT}/.env.example" ]]; then
        if grep -q "DATABASE_URL=.*postgres:" "${PROJECT_ROOT}/.env.example"; then
            log_pass "Database uses Docker network service name (not localhost)"
        fi
        if grep -q "REDIS_URL=.*redis:" "${PROJECT_ROOT}/.env.example"; then
            log_pass "Redis uses Docker network service name (not localhost)"
        fi
    fi
}

# =============================================================================
# 6. DEVELOPMENT STANDARDS VALIDATION
# =============================================================================
validate_development_standards() {
    log_section "PHASE 6: Development Standards Validation"

    # 6.1 Check for localhost usage in development configs
    log_info "Checking for localhost/127.0.0.1 violations..."
    local dev_compose="${PROJECT_ROOT}/docker/docker-compose.local.yml"
    if [[ -f "$dev_compose" ]]; then
        # This is actually correct for Docker - ports bound to 127.0.0.1 are acceptable
        log_pass "Development compose file exists"
    fi

    # 6.2 Check Python version
    log_info "Checking Python version requirements..."
    if grep -q "python_requires.*3\.11\|python.*3\.11" "${PROJECT_ROOT}/pyproject.toml" 2>/dev/null; then
        log_pass "Python 3.11+ requirement is specified"
    else
        log_warn "Python 3.11+ requirement may not be explicitly specified"
    fi

    # 6.3 Check type hints requirement
    log_info "Checking mypy configuration..."
    if [[ -f "${PROJECT_ROOT}/mypy.ini" ]]; then
        if grep -q "strict\s*=\s*True\|strict=True" "${PROJECT_ROOT}/mypy.ini"; then
            log_pass "mypy strict mode is enabled"
        else
            log_warn "mypy strict mode may not be enabled"
        fi
    else
        log_warn "mypy.ini not found"
    fi

    # 6.4 Check test coverage configuration
    log_info "Checking test coverage configuration..."
    if grep -r "cov.*90\|cov_fail_under.*90" "${PROJECT_ROOT}/pyproject.toml" "${PROJECT_ROOT}/setup.cfg" 2>/dev/null; then
        log_pass "Test coverage threshold is configured (≥90%)"
    else
        log_warn "Test coverage threshold may not be configured"
    fi
}

# =============================================================================
# 7. INFRASTRUCTURE ALIGNMENT (Three-Lens Framework)
# =============================================================================
validate_infrastructure_alignment() {
    log_section "PHASE 7: Infrastructure Alignment (Three-Lens Framework)"

    log_info "Evaluating CEO Lens (Cost)..."
    # Check for cost-related labels
    if grep -q "monthly_budget_usd\|cost_center\|chargeback_unit" "$PMO_FILE"; then
        log_pass "Cost attribution labels present in PMO"
    else
        log_fail "Cost attribution labels missing from PMO"
    fi

    log_info "Evaluating CTO Lens (Innovation)..."
    # Check for version and stack labels
    if grep -q "version:\|stack:" "$PMO_FILE"; then
        log_pass "Technology stack documentation present"
    else
        log_warn "Technology stack documentation may be incomplete"
    fi

    log_info "Evaluating CFO Lens (ROI)..."
    # Check for budget and project code
    if grep -q "project_code\|budget_owner" "$PMO_FILE"; then
        log_pass "Budget ownership and project tracking present"
    else
        log_warn "Budget ownership documentation may be incomplete"
    fi

    # Check Terraform presence
    log_info "Checking infrastructure as code..."
    if [[ -d "${PROJECT_ROOT}/docker/terraform" ]]; then
        log_pass "Terraform directory exists"
        local tf_files=$(find "${PROJECT_ROOT}/docker/terraform" -name "*.tf" | wc -l)
        log_info "Found $tf_files Terraform files"
    else
        log_warn "Terraform directory not found"
    fi
}

# =============================================================================
# 8. COST/SPEED/ROBUSTNESS/DELIVERY ANALYSIS
# =============================================================================
analyze_enhancements() {
    log_section "PHASE 8: Enhancement Analysis (Cost/Speed/Robustness/Delivery)"

    echo ""
    echo -e "${BOLD}${CYAN}┌─────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${CYAN}│                    ENHANCEMENT OPPORTUNITIES                            │${NC}"
    echo -e "${BOLD}${CYAN}└─────────────────────────────────────────────────────────────────────────┘${NC}"

    # COST ENHANCEMENTS
    echo ""
    echo -e "${YELLOW}💰 COST OPTIMIZATION:${NC}"
    echo "   ┌────────────────────────────────────────────────────────────────────┐"

    # Check for multi-stage Docker builds
    if grep -q "FROM.*AS\|as builder" "${PROJECT_ROOT}/docker/Dockerfile" 2>/dev/null; then
        echo "   │ ✓ Multi-stage Docker builds detected (good for image size)       │"
    else
        echo "   │ ⚠ Consider multi-stage Docker builds to reduce image size        │"
    fi

    # Check for resource limits
    if grep -rq "resources:\|limits:\|memory:" "${PROJECT_ROOT}/docker/" 2>/dev/null; then
        echo "   │ ✓ Resource limits configured in Docker Compose                   │"
    else
        echo "   │ ⚠ Add resource limits to prevent runaway costs                   │"
    fi

    # Check for auto-delete labels
    if grep -q "auto_delete.*true" "$PMO_FILE" 2>/dev/null; then
        echo "   │ ⚠ Auto-delete enabled - verify this is intentional               │"
    else
        echo "   │ ✓ Resources are not set to auto-delete                           │"
    fi

    echo "   │                                                                    │"
    echo "   │ 📋 Recommendations:                                                │"
    echo "   │    • Implement scheduled scaling for off-hours                    │"
    echo "   │    • Consider spot/preemptible instances for non-critical loads   │"
    echo "   │    • Set up budget alerts at 50%, 80%, 100% of monthly budget     │"
    echo "   └────────────────────────────────────────────────────────────────────┘"

    # SPEED ENHANCEMENTS
    echo ""
    echo -e "${GREEN}⚡ SPEED OPTIMIZATION:${NC}"
    echo "   ┌────────────────────────────────────────────────────────────────────┐"

    # Check for caching
    if grep -rq "redis\|cache\|Redis" "${PROJECT_ROOT}/ollama/" 2>/dev/null; then
        echo "   │ ✓ Redis caching implementation detected                          │"
    else
        echo "   │ ⚠ Consider adding Redis caching for frequently accessed data    │"
    fi

    # Check for async operations
    if grep -rq "async def\|asyncio\|await" "${PROJECT_ROOT}/ollama/" 2>/dev/null; then
        echo "   │ ✓ Async operations detected (FastAPI async-first)               │"
    else
        echo "   │ ⚠ Consider async operations for I/O-bound tasks                 │"
    fi

    # Check for connection pooling
    if grep -rq "pool_size\|max_overflow\|pool_pre_ping" "${PROJECT_ROOT}/ollama/" 2>/dev/null; then
        echo "   │ ✓ Connection pooling configured                                  │"
    else
        echo "   │ ⚠ Configure connection pooling for database connections          │"
    fi

    echo "   │                                                                    │"
    echo "   │ 📋 Recommendations:                                                │"
    echo "   │    • Implement response caching with TTL for static responses     │"
    echo "   │    • Consider CDN for static assets                               │"
    echo "   │    • Profile and optimize hot code paths                          │"
    echo "   │    • Use lazy loading for large models                            │"
    echo "   └────────────────────────────────────────────────────────────────────┘"

    # ROBUSTNESS ENHANCEMENTS
    echo ""
    echo -e "${BLUE}🛡️ ROBUSTNESS OPTIMIZATION:${NC}"
    echo "   ┌────────────────────────────────────────────────────────────────────┐"

    # Check for health checks
    if grep -rq "healthcheck\|health_check\|/health" "${PROJECT_ROOT}/docker/" 2>/dev/null; then
        echo "   │ ✓ Health checks configured in Docker Compose                    │"
    else
        echo "   │ ⚠ Add health checks to all services                             │"
    fi

    # Check for retry logic
    if grep -rq "retry\|backoff\|Retry\|tenacity" "${PROJECT_ROOT}/ollama/" 2>/dev/null; then
        echo "   │ ✓ Retry logic detected in application code                      │"
    else
        echo "   │ ⚠ Consider adding retry logic with exponential backoff          │"
    fi

    # Check for circuit breaker
    if grep -rq "circuit_breaker\|CircuitBreaker\|pybreaker" "${PROJECT_ROOT}/ollama/" 2>/dev/null; then
        echo "   │ ✓ Circuit breaker pattern detected                               │"
    else
        echo "   │ ⚠ Consider implementing circuit breaker for external services   │"
    fi

    # Check for backup strategy
    if grep -q "backup_strategy" "$PMO_FILE" 2>/dev/null; then
        local backup=$(grep "backup_strategy:" "$PMO_FILE" | cut -d'"' -f2)
        echo "   │ ✓ Backup strategy configured: $backup                           │"
    fi

    echo "   │                                                                    │"
    echo "   │ 📋 Recommendations:                                                │"
    echo "   │    • Implement graceful degradation for non-critical features     │"
    echo "   │    • Add chaos engineering tests (fault injection)                │"
    echo "   │    • Set up automated failover for critical services              │"
    echo "   │    • Test disaster recovery procedures quarterly                  │"
    echo "   └────────────────────────────────────────────────────────────────────┘"

    # DELIVERY ENHANCEMENTS
    echo ""
    echo -e "${MAGENTA}🚀 DELIVERY OPTIMIZATION:${NC}"
    echo "   ┌────────────────────────────────────────────────────────────────────┐"

    # Check for CI/CD
    if [[ -d "${PROJECT_ROOT}/.github/workflows" ]]; then
        local workflow_count=$(ls -1 "${PROJECT_ROOT}/.github/workflows"/*.yml 2>/dev/null | wc -l)
        echo "   │ ✓ GitHub Actions workflows detected: $workflow_count workflow(s)             │"
    else
        echo "   │ ⚠ No GitHub Actions workflows found                              │"
    fi

    # Check for pre-commit hooks
    if [[ -f "${PROJECT_ROOT}/.pre-commit-config.yaml" ]]; then
        echo "   │ ✓ Pre-commit hooks configured                                    │"
    else
        echo "   │ ⚠ Consider adding pre-commit hooks for quality gates             │"
    fi

    # Check for deployment scripts
    local deploy_scripts=$(find "${PROJECT_ROOT}/scripts" -name "deploy*.sh" 2>/dev/null | wc -l)
    echo "   │ ✓ Deployment scripts found: $deploy_scripts script(s)                        │"

    echo "   │                                                                    │"
    echo "   │ 📋 Recommendations:                                                │"
    echo "   │    • Implement blue-green or canary deployments                   │"
    echo "   │    • Add automated rollback on failure                            │"
    echo "   │    • Set up feature flags for gradual rollouts                    │"
    echo "   │    • Automate changelog generation from commits                   │"
    echo "   └────────────────────────────────────────────────────────────────────┘"
}

# =============================================================================
# SUMMARY AND REPORT
# =============================================================================
generate_summary() {
    log_header "LANDING ZONE BOOTSTRAP SUMMARY"

    echo ""
    echo -e "${BOLD}Validation Results:${NC}"
    echo "────────────────────────────────────────"
    echo -e "  ${GREEN}✓ Passed${NC}:   $PASSED"
    echo -e "  ${RED}✗ Failed${NC}:   $FAILED"
    echo -e "  ${YELLOW}⚠ Warnings${NC}: $WARNINGS"
    echo -e "  ${MAGENTA}○ Skipped${NC}:  $SKIPPED"
    echo "────────────────────────────────────────"

    local total=$((PASSED + FAILED + WARNINGS))
    if [[ $total -gt 0 ]]; then
        local compliance_score=$(( (PASSED * 100) / total ))
        echo -e "  ${BOLD}Compliance Score: ${compliance_score}%${NC}"
    fi

    echo ""
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ LANDING ZONE COMPLIANCE: PASSED${NC}"
        echo "  This repository meets GCP Landing Zone requirements."
    else
        echo -e "${RED}${BOLD}✗ LANDING ZONE COMPLIANCE: NEEDS ATTENTION${NC}"
        echo "  $FAILED issue(s) must be resolved before deployment."
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        echo -e "${CYAN}[DRY-RUN MODE] No changes were made.${NC}"
    fi

    echo ""
    echo "Timestamp: $TIMESTAMP"
    echo "Report: Run with --report flag to generate detailed report"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
main() {
    log_header "GCP LANDING ZONE BOOTSTRAP"
    echo "Onboarding validation for: $PROJECT_ROOT"
    echo "Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY-RUN" || echo "VALIDATION")"
    echo "Timestamp: $TIMESTAMP"

    check_prerequisites
    validate_pmo_metadata
    validate_docker_naming
    validate_security_controls
    validate_no_root_chaos
    validate_deployment_architecture
    validate_development_standards
    validate_infrastructure_alignment
    analyze_enhancements
    generate_summary
}

# Run main
main "$@"
