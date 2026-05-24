#!/bin/bash
# apply_fixes.sh — Apply all gfx12 WMMA fixes to your ollama-ROCM checkout
# Run from repo root after checking out the rdna4-gfx1201 branch.
#
# Usage:
#   chmod +x apply_fixes.sh
#   ./apply_fixes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

echo "=== Ollama ROCM gfx12 WMMA Fix Applier ==="
echo "Repo root: $REPO_ROOT"
echo ""

# ── Step 1: Replace the kernel file ──────────────────────────────────────────
KERNEL_SRC="$REPO_ROOT/ml/backend/ggml/ggml/src/ggml-cuda/fattn-wmma-gfx12.cuh"
KERNEL_FIX="$REPO_ROOT/fattn-wmma-gfx12-fixed.cuh"

if [ ! -f "$KERNEL_FIX" ]; then
    echo "ERROR: fattn-wmma-gfx12-fixed.cuh not found in repo root!"
    echo "Download it from the release assets and place it here."
    exit 1
fi

if [ -f "$KERNEL_SRC" ]; then
    cp "$KERNEL_SRC" "$KERNEL_SRC.bak"
    echo "[OK] Backed up original fattn-wmma-gfx12.cuh"
fi

cp "$KERNEL_FIX" "$KERNEL_SRC"
echo "[OK] Installed fixed fattn-wmma-gfx12.cuh"

# ── Step 2: Patch fattn.cu dispatch ──────────────────────────────────────────
FATTN_CU="$REPO_ROOT/ml/backend/ggml/ggml/src/ggml-cuda/fattn.cu"
FATTN_PATCH="$REPO_ROOT/fattn.cu.patch"

if [ ! -f "$FATTN_PATCH" ]; then
    echo "ERROR: fattn.cu.patch not found in repo root!"
    exit 1
fi

if [ -f "$FATTN_CU" ]; then
    cp "$FATTN_CU" "$FATTN_CU.bak"
    echo "[OK] Backed up original fattn.cu"

    # Apply patch (with fuzz tolerance)
    if patch -p1 --fuzz=3 < "$FATTN_PATCH"; then
        echo "[OK] Patched fattn.cu dispatch logic"
    else
        echo "[WARN] Patch failed cleanly. Attempting manual merge..."
        # The patch may fail due to upstream drift. In that case, manual merge
        # is required. The key changes are:
        # 1. Add gfx12 WMMA include
        # 2. Add gfx12 fast-path in ggml_cuda_get_best_fattn_kernel()
        # 3. Add gfx12 launch in ggml_cuda_flash_attn_ext() case WMMA_F16
        echo "[FAIL] Manual merge required. See fattn.cu.patch for expected changes."
        exit 1
    fi
else
    echo "ERROR: fattn.cu not found at expected path: $FATTN_CU"
    exit 1
fi

# ── Step 3: Verify includes ──────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
if grep -q 'fattn-wmma-gfx12.cuh' "$FATTN_CU"; then
    echo "[OK] fattn-wmma-gfx12.cuh is included in fattn.cu"
else
    echo "[WARN] Include not found in fattn.cu — may need manual fix"
fi

if grep -q 'launch_flash_attn_ext_gfx12' "$FATTN_CU"; then
    echo "[OK] gfx12 launcher is called in fattn.cu"
else
    echo "[WARN] gfx12 launcher call not found — dispatch may be broken"
fi

# ── Step 4: Install Linux build script ───────────────────────────────────────
if [ -f "$REPO_ROOT/build_gfx1201.sh" ]; then
    chmod +x "$REPO_ROOT/build_gfx1201.sh"
    echo "[OK] build_gfx1201.sh is executable"
fi

echo ""
echo "=== All fixes applied! ==="
echo ""
echo "Next steps:"
echo "  1. Review the patched files (diff against .bak if needed)"
echo "  2. Run: ./build_gfx1201.sh   (Linux) or ./build_gfx1201.ps1 (Windows)"
echo "  3. Verify the kernel is executing by running with verbose:"
echo "       OLLAMA_DEBUG=1 ./build/bin/ollama run model 'test' --verbose"
echo "     Look for 'gfx12' in the debug output."
echo ""
echo "If the patch failed, manually apply the changes from fattn.cu.patch"
echo "or open an issue with your upstream llama.cpp commit hash."
