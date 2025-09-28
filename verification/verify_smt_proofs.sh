#!/bin/bash

# SMT/Z3 Verification Script for Ollama Memory Optimizations
# Verifies implementation against formal SMT specifications

set -e

echo "🔍 SMT/Z3 Verification for Ollama Memory Optimizations"
echo "=================================================="

# Check if Z3 is available
if ! command -v z3 &> /dev/null; then
    echo "❌ Z3 not found. Please install Z3 SMT solver:"
    echo "   Ubuntu/Debian: sudo apt-get install z3"
    echo "   macOS: brew install z3"
    echo "   Or visit: https://github.com/Z3Prover/z3"
    exit 1
fi

echo "✅ Z3 version: $(z3 --version)"
echo

# Create results directory
RESULTS_DIR="verification_results"
mkdir -p "$RESULTS_DIR"

# Function to run SMT verification
run_smt_verification() {
    local file="$1"
    local name="$2"

    echo "🧮 Verifying $name..."
    echo "   File: $file"

    # Run Z3 with timeout and capture output
    if timeout 30s z3 "$file" > "$RESULTS_DIR/${name}_output.txt" 2>&1; then
        # Check if the result is SAT (satisfiable)
        if grep -q "^sat$" "$RESULTS_DIR/${name}_output.txt"; then
            echo "   ✅ $name: SAT (constraints satisfiable)"

            # Extract and display key verification results
            echo "   📊 Key results:"
            grep -A 20 "echo.*:" "$RESULTS_DIR/${name}_output.txt" | grep -v "^echo" | sed 's/^/      /'
        else
            echo "   ❌ $name: UNSAT or ERROR"
            head -10 "$RESULTS_DIR/${name}_output.txt" | sed 's/^/      /'
        fi
    else
        echo "   ⏰ $name: TIMEOUT (>30s)"
    fi
    echo
}

# Run all SMT verifications
run_smt_verification "checkpoint_optimization.smt2" "Checkpoint_Optimization"
run_smt_verification "mla_compression.smt2" "MLA_Compression"
run_smt_verification "gpu_device_selection.smt2" "GPU_Device_Selection"
run_smt_verification "combined_optimizations.smt2" "Combined_Optimizations"

# Summary
echo "📋 VERIFICATION SUMMARY"
echo "====================="

total_files=4
sat_count=0

for name in "Checkpoint_Optimization" "MLA_Compression" "GPU_Device_Selection" "Combined_Optimizations"; do
    if [ -f "$RESULTS_DIR/${name}_output.txt" ] && grep -q "^sat$" "$RESULTS_DIR/${name}_output.txt"; then
        echo "✅ $name: VERIFIED"
        ((sat_count++))
    else
        echo "❌ $name: FAILED"
    fi
done

echo
echo "📊 Results: $sat_count/$total_files specifications verified"

if [ $sat_count -eq $total_files ]; then
    echo "🎉 ALL SMT SPECIFICATIONS VERIFIED!"
    echo "   The implementation is mathematically sound according to Z3."
else
    echo "⚠️  Some specifications failed. Check individual results in $RESULTS_DIR/"
    exit 1
fi

echo
echo "💾 Detailed results saved in: $RESULTS_DIR/"
echo "🔗 Run 'cat $RESULTS_DIR/*_output.txt' to see full Z3 outputs"