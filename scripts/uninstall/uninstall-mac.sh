#!/bin/bash
#
# uninstall-mac.sh - Uninstall Ollama from macOS
#
# Usage:
#   ./uninstall-mac.sh [--dry-run]
#
# Options:
#   --dry-run    Show what would be deleted without actually deleting
#

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_dry()   { echo -e "${YELLOW}[DRY-RUN]${NC} Would: $*"; }

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        log_dry "$*"
    else
        "$@"
    fi
}

remove_file() {
    local path="$1"
    if [ -e "$path" ] || [ -L "$path" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_dry "Remove: $path"
        else
            log_info "Removing: $path"
            rm -f "$path"
        fi
    else
        log_info "Not found (skipping): $path"
    fi
}

remove_dir() {
    local path="$1"
    if [ -d "$path" ]; then
        if [ "$DRY_RUN" = true ]; then
            log_dry "Remove directory: $path"
        else
            log_info "Removing directory: $path"
            rm -rf "$path"
        fi
    else
        log_info "Not found (skipping): $path"
    fi
}

echo "========================================="
echo "  Ollama Uninstaller for macOS"
echo "========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Step 1: Stop running Ollama processes
log_info "Step 1: Stopping Ollama processes..."
if pgrep -x "Ollama" >/dev/null 2>&1 || pgrep -f "ollama" >/dev/null 2>&1; then
    run_cmd pkill -x "Ollama" 2>/dev/null || true
    run_cmd pkill -f "ollama" 2>/dev/null || true
    sleep 1
    log_info "Ollama processes stopped."
else
    log_info "No running Ollama processes found."
fi
echo ""

# Step 2: Remove symlink
log_info "Step 2: Removing symlink..."
remove_file "/usr/local/bin/ollama"
echo ""

# Step 3: Remove application bundle
log_info "Step 3: Removing application bundle..."
remove_dir "/Applications/Ollama.app"
echo ""

# Step 4: Ask about model data
log_info "Step 4: Model data..."
if [ -d "$HOME/.ollama" ]; then
    echo "Model directory found at: ~/.ollama"
    if [ "$DRY_RUN" = true ]; then
        log_dry "Would remove: ~/.ollama (contains downloaded models and cache)"
    else
        read -rp "Do you want to remove downloaded models and cache (~/.ollama)? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            remove_dir "$HOME/.ollama"
            log_info "Model data removed."
        else
            log_info "Model data retained at ~/.ollama"
        fi
    fi
else
    log_info "No model directory found at ~/.ollama"
fi
echo ""

# Summary
echo "========================================="
if [ "$DRY_RUN" = true ]; then
    log_info "Dry run complete. No changes were made."
    log_info "Run without --dry-run to perform the actual uninstall."
else
    log_info "Ollama has been uninstalled from your Mac."
    if [ -d "$HOME/.ollama" ]; then
        log_warn "Model data still exists at ~/.ollama"
        log_info "To remove it manually: rm -rf ~/.ollama"
    fi
fi
echo "========================================="
