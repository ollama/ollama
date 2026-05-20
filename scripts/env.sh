# Common environment setup across build*.sh scripts

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"
# TODO - consider `docker buildx ls --format=json` to autodiscover platform capability
PLATFORM=${PLATFORM:-"linux/arm64,linux/amd64"}
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}
OLLAMA_COMMON_BUILD_ARGS="--build-arg=GOFLAGS"

# Resolve the TheRock nightly version at build time so the Docker cache key is
# pinned to a specific build rather than the opaque string "latest".
if [ -z "${THEROCK_VERSION:-}" ]; then
    _therock_resolved=$(bash "$(dirname "$0")/fetch_therock_rocm_linux.sh" \
        --target "${THEROCK_TARGET:-multiarch}" \
        --resolve-only 2>/dev/null | awk '/Version:/{print $2}')
    if [ -n "$_therock_resolved" ]; then
        THEROCK_VERSION="$_therock_resolved"
        export THEROCK_VERSION
    fi
fi

add_build_arg() {
    eval "_value=\"\${$1:-}\""
    if [ -n "$_value" ]; then
        OLLAMA_COMMON_BUILD_ARGS="$OLLAMA_COMMON_BUILD_ARGS --build-arg=$1"
    fi
}

for arg in \
    CGO_CFLAGS \
    CGO_CXXFLAGS \
    CMAKEVERSION \
    NINJAVERSION \
    ROCMVERSION \
    JETPACK5VERSION \
    JETPACK6VERSION \
    CUDA12VERSION \
    CUDA13VERSION \
    VULKANVERSION \
    MLX_CUDA_RAM_MB \
    APT_MIRROR \
    OLLAMA_MLX_BUILD_JOBS \
    OLLAMA_MLX_NVCC_THREADS \
    THEROCK_VERSION \
    THEROCK_TARGET
do
    add_build_arg "$arg"
done

# Forward local MLX source overrides as Docker build contexts
if [ -n "${OLLAMA_MLX_SOURCE:-}" ]; then
    OLLAMA_COMMON_BUILD_ARGS="$OLLAMA_COMMON_BUILD_ARGS --build-context local-mlx=$(cd "$OLLAMA_MLX_SOURCE" && pwd)"
fi
if [ -n "${OLLAMA_MLX_C_SOURCE:-}" ]; then
    OLLAMA_COMMON_BUILD_ARGS="$OLLAMA_COMMON_BUILD_ARGS --build-context local-mlx-c=$(cd "$OLLAMA_MLX_C_SOURCE" && pwd)"
fi
echo "Building Ollama"
echo "VERSION=$VERSION"
echo "PLATFORM=$PLATFORM"
