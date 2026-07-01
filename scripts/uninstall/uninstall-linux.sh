#!/bin/bash
#
# uninstall-linux.sh - Uninstall Ollama from Linux
#
# Usage:
#   sudo ./uninstall-linux.sh [--dry-run]
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
echo "  Ollama Uninstaller for Linux"
echo "========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - No changes will be made"
    echo ""
fi

if [ "$(id -u)" -ne 0 ] && [ "$DRY_RUN" = false ]; then
    log_error "This script requires root privileges. Please run with sudo."
    exit 1
fi

# Step 1: Stop and disable systemd service
log_info "Step 1: Stopping Ollama systemd service..."
if command -v systemctl >/dev/null 2>&1; then
    if systemctl is-active --quiet ollama 2>/dev/null; then
        run_cmd systemctl stop ollama
        log_info "Ollama service stopped."
    else
        log_info "Ollama service not running."
    fi

    if systemctl is-enabled --quiet ollama 2>/dev/null; then
        run_cmd systemctl disable ollama
        log_info "Ollama service disabled."
    else
        log_info "Ollama service not enabled."
    fi

    # Remove systemd service file
    if [ -f "/etc/systemd/system/ollama.service" ]; then
        remove_file "/etc/systemd/system/ollama.service"
        run_cmd systemctl daemon-reload
        log_info "Systemd daemon reloaded."
    else
        log_info "No systemd service file found."
    fi
else
    log_info "systemd not found, skipping service management."
fi
echo ""

# Step 2: Remove binary and symlink
log_info "Step 2: Removing Ollama binary..."
OLLAMA_BIN="$(which ollama 2>/dev/null || true)"
if [ -n "$OLLAMA_BIN" ]; then
    remove_file "$OLLAMA_BIN"
else
    log_info "No ollama binary found in PATH."
fi
echo ""

# Step 3: Remove library files
log_info "Step 3: Removing library files..."
remove_dir "/usr/local/lib/ollama"
echo ""

# Step 4: Remove ollama system user
log_info "Step 4: Removing ollama system user..."
if id "ollama" >/dev/null 2>&1; then
    if [ "$DRY_RUN" = true ]; then
        log_dry "Remove ollama user and home directory"
    else
        log_info "Removing ollama user..."
        userdel -r ollama 2>/dev/null || userdel ollama 2>/dev/null || true
        log_info "Ollama user removed."
    fi
else
    log_info "No ollama user found."
fi
echo ""

# Step 5: Ask about model data
log_info "Step 5: Model data..."
MODEL_DIRS_FOUND=false

if [ -d "/usr/share/ollama" ]; then
    MODEL_DIRS_FOUND=true
    echo "Model directory found at: /usr/share/ollama"
fi

if [ -d "$HOME/.ollama" ]; then
    MODEL_DIRS_FOUND=true
    echo "Model directory found at: ~/.ollama"
fi

if [ "$MODEL_DIRS_FOUND" = true ]; then
    if [ "$DRY_RUN" = true ]; then
        if [ -d "/usr/share/ollama" ]; then
            log_dry "Remove: /usr/share/ollama"
        fi
        if [ -d "$HOME/.ollama" ]; then
            log_dry "Remove: ~/.ollama"
        fi
    else
        read -rp "Do you want to remove downloaded models and cache? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            remove_dir "/usr/share/ollama"
            remove_dir "$HOME/.ollama"
            log_info "Model data removed."
        else
            log_info "Model data retained."
        fi
    fi
else
    log_info "No model directories found."
fi
echo ""

# Summary
echo "========================================="
if [ "$DRY_RUN" = true ]; then
    log_info "Dry run complete. No changes were made."
    log_info "Run without --dry-run to perform the actual uninstall."
else
    log_info "Ollama has been uninstalled from your system."
    if [ -d "/usr/share/ollama" ] || [ -d "$HOME/.ollama" ]; then
        log_warn "Model data still exists."
        log_info "To remove it manually: sudo rm -rf /usr/share/ollama ~/.ollama"
    fi
fi
echo "========================================="
