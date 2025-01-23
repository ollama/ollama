#/bin/sh

# Wrapper script to speed up builds by disabling some permutations and reduce compatibility matrix
# Don't use for release builds, but suitable for local developer iteration

# Only build cuda v12
export OLLAMA_SKIP_CUDA_11_GENERATE=1
# Major versions only
export CUDA_V12_ARCHITECTURES="60;70;80;90"
# Skip ROCm
export OLLAMA_SKIP_ROCM_GENERATE=1
# Disable various less common quants and fattn
export OLLAMA_FAST_BUILD=1

if [ $# -ne 1 ] ; then
    echo "Usage: ./scripts/fast.sh <build_script>"
    exit 1
fi

exec $1