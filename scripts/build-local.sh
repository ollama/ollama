#!/usr/bin/env bash
#
# scripts/build-local.sh
#
# Quick, reproducible local development build for BOTH the Go API layer
# (the ollama binary / server / gRPC+REST adapters) AND the native
# llama-server runner payload (with Metal on Apple Silicon).
#
# This is the recommended workflow for fast iteration on this branch
# (especially gRPC work, adapters, converters, clients, scheduling, etc.).
#
# Usage:
#   ./scripts/build-local.sh                 # full configure + ollama-local (Go + runner)
#   ./scripts/build-local.sh --go-only       # fast Go-only rebuild (once native payload exists)
#   ./scripts/build-local.sh configure       # just cmake -B build
#   ./scripts/build-local.sh build           # build the default target
#   ./scripts/build-local.sh clean           # remove build/
#   ./scripts/build-local.sh --help
#
# After a successful run you will have:
#   ./ollama                 (the Go binary, gitignored)
#   build/lib/ollama/llama-server   (the native runner + Metal libs, under build/)
#
# These are already covered by .gitignore. The binary will discover the
# payload via the updated logic in llm/llama_binary.go (supports the
# build/ layout for harness + OLLAMA_GRPC_HOST dev/testing).
#
# Examples for gRPC vs REST testing:
#   OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve
#   OLLAMA_GRPC_HOST=127.0.0.1:11435 go test -tags=integration -run TestGRPCStreaming ./integration -count=1
#
# See also: docs/development.md and docs/grpc-phased-reliable-approach.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
scripts/build-local.sh - local dev build (Go API + llama-server payload)

Modes / flags:
  (default)          Full configure (if needed) + build ollama-local target
                     (produces ./ollama + build/lib/ollama/llama-server etc.)
  configure          Only run cmake -B build . (idempotent)
  build              Build the default target (ollama-local or ollama-go)
  clean              Remove the build/ directory
  --go-only          Build only the Go layer (ollama-go target). Much faster
                     once the native payload has been built at least once.
  --target <t>       Build a specific cmake target (advanced)
  --preset <name>    Use a specific configure preset from CMakePresets.json
  -j, --parallel N   Override parallelism (default: auto-detect)
  -h, --help         Show this help

Environment variables honored:
  OLLAMA_MLX_BACKENDS   (e.g. "metal_v3;metal_v4" or empty to disable MLX)
  CMAKE_BUILD_TYPE      (default Release via presets / local.cmake)
  Any other vars passed through to cmake.

Examples:
  ./scripts/build-local.sh
  ./scripts/build-local.sh --go-only
  ./scripts/build-local.sh clean
  OLLAMA_MLX_BACKENDS= ./scripts/build-local.sh   # lighter, no MLX
EOF
}

MODE="all"
GO_ONLY=false
TARGET=""
PRESET=""
PARALLEL=""
EXTRA_CMAKE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --go-only) GO_ONLY=true; shift ;;
    --target) TARGET="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    -j|--parallel) PARALLEL="$2"; shift 2 ;;
    configure|build|clean|all)
      MODE="$1"; shift ;;
    *)
      # Pass anything else through to the initial cmake configure
      EXTRA_CMAKE_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$PARALLEL" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    PARALLEL=$(nproc)
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    PARALLEL=$(sysctl -n hw.ncpu)
  else
    PARALLEL=4
  fi
fi

# Basic prereq checks (non-fatal for configure step; cmake will give better errors)
check_prereqs() {
  if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake not found in PATH. Install with: brew install cmake" >&2
    exit 1
  fi
  if command -v ninja >/dev/null 2>&1; then
    echo ">>> ninja detected (recommended, will be used if generator prefers it)"
  fi
}

detect_darwin_metal() {
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo ">>> macOS arm64 detected — default build enables Metal (via OLLAMA_MLX_BACKENDS)."
    echo "    If this is the first build you may need the Metal toolchain:"
    echo "      xcodebuild -downloadComponent MetalToolchain"
    echo "    (cmake will fail with a clear message if it is missing.)"
  fi
}

do_configure() {
  check_prereqs
  detect_darwin_metal

  local configure_cmd=(cmake -B build)
  if [[ -n "$PRESET" ]]; then
    configure_cmd+=(--preset "$PRESET")
  fi
  # Safe expansion under set -u for possibly empty array (common bash gotcha)
  configure_cmd+=(${EXTRA_CMAKE_ARGS[@]+"${EXTRA_CMAKE_ARGS[@]}"})

  echo ">>> Configuring: ${configure_cmd[*]}"
  "${configure_cmd[@]}"
}

do_build() {
  local build_target="ollama-local"
  if [[ -n "$TARGET" ]]; then
    build_target="$TARGET"
  elif [[ "$GO_ONLY" == true ]]; then
    build_target="ollama-go"
  fi

  echo ">>> Building target: ${build_target} (parallel=${PARALLEL})"
  cmake --build build --target "${build_target}" --parallel "${PARALLEL}"
}

do_clean() {
  echo ">>> Cleaning build/ directory (root ./ollama and integration/ollama are gitignored)"
  rm -rf build
  # We intentionally do NOT rm -f ./ollama here — the user may have other copies
  # or want to keep a working one while cleaning the cmake tree.
}

case "$MODE" in
  clean)
    do_clean
    exit 0
    ;;
  configure)
    do_configure
    exit 0
    ;;
  build)
    do_build
    ;;
  all|*)
    if [[ ! -d build || ! -f build/CMakeCache.txt ]]; then
      do_configure
    else
      echo ">>> build/ already configured — skipping full configure (use 'clean' to force)"
    fi
    do_build
    ;;
esac

echo
echo ">>> Build complete."
echo "    Go binary     : $(ls -l ./ollama 2>/dev/null || echo 'not present')"
echo "    llama-server  : $(ls -l build/lib/ollama/llama-server 2>/dev/null || echo 'not present under build/lib/ollama/')"
echo
echo "To run with both REST and gRPC (separate ports, as used in gRPC reports/tests):"
echo "  OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve"
echo
echo "Or SAMEPORT (cmux, more advanced):"
echo "  OLLAMA_GRPC_SAMEPORT=1 ./ollama serve"
echo
echo "Integration gRPC streaming test (requires model + the payload we just built):"
echo "  OLLAMA_GRPC_HOST=127.0.0.1:11435 go test -tags=integration -run TestGRPCStreaming ./integration -count=1 -timeout=5m"
echo
echo "Quality/parity/edge-case comparison (real gRPC data vs REST; tools in streams, mid-gen cancel+token counts, OTEL/pprof/metrics check; SAMEPORT + separate port):"
echo "  # Separate port (standard):"
echo "  OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve &"
echo "  go run scripts/quality-grpc-comparison.go -model llama3.2:1b"
echo "  # SAMEPORT (cmux):"
echo "  OLLAMA_GRPC_SAMEPORT=1 ./ollama serve &"
echo "  go run scripts/quality-grpc-comparison.go -sameport -rest http://127.0.0.1:11434 -grpc 127.0.0.1:11434 -model llama3.2:1b"
echo "  See scripts/quality-grpc-comparison.go header + docs/development.md + docs/grpc-phased-reliable-approach.md for details and expected token matching."
echo
echo "For pure-Go iteration after the native payload exists once, you can also just do:"
echo "  go build -o ollama ."
echo "  OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve"
echo
echo "Tip: ./scripts/build-local.sh --go-only   is fast for Go-only changes (gRPC handlers, clients, converters, etc.)."
