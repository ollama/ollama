#!/usr/bin/env bash
#
# Download and unpack a Linux ROCm SDK tarball from AMD TheRock nightlies.
#
# Examples:
#   ./scripts/fetch_therock_rocm_linux.sh
#   ./scripts/fetch_therock_rocm_linux.sh --target gfx110X
#   ./scripts/fetch_therock_rocm_linux.sh --version 7.14.0a20260519
#
# The default prefix is repo-local and does not require root.

set -euo pipefail

TARGET="multiarch"
VERSION="latest"
ARTIFACT_BASE_URL="https://rocm.nightlies.amd.com/tarball-multi-arch"
RESOLVE_ONLY=false
FORCE=false
PREFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --artifact-base-url) ARTIFACT_BASE_URL="$2"; shift 2 ;;
        --resolve-only) RESOLVE_ONLY=true; shift ;;
        --force) FORCE=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

repo_root() {
    if git rev-parse --show-toplevel 2>/dev/null; then
        return
    fi
    dirname "$(dirname "$(readlink -f "$0")")"
}

if [[ -z "$PREFIX" ]]; then
    ROOT="$(repo_root)"
    PREFIX="${ROOT}/.cache/therock-rocm"
fi

target_artifact_name() {
    case "$1" in
        multiarch) echo "multiarch" ;;
        gfx103X)   echo "gfx103X-all" ;;
        gfx110X)   echo "gfx110X-all" ;;
        gfx120X)   echo "gfx120X-all" ;;
        *)         echo "$1" ;;
    esac
}

ARTIFACT_TARGET="$(target_artifact_name "$TARGET")"

read_artifact_keys() {
    local base_url="$1" prefix="$2"
    curl -fsSL "${base_url}/" \
        | grep -oE "therock-dist-linux-[A-Za-z0-9._-]+\.tar\.gz" \
        | grep "^${prefix}" \
        | sort -u
}

parse_artifact_version() {
    local key="$1" artifact_target="$2"
    local re="^therock-dist-linux-${artifact_target}-([0-9]+\.[0-9]+\.[0-9]+(a|rc)[0-9]+)\.tar\.gz$"
    if [[ "$key" =~ $re ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

version_build_number() {
    local ver="$1"
    if [[ "$ver" =~ (a|rc)([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[2]}"
    else
        echo "0"
    fi
}

resolve_latest() {
    local base_url="$1" artifact_target="$2"
    local prefix="therock-dist-linux-${artifact_target}-"
    local keys latest_key="" latest_build=-1

    echo "Querying ${base_url}/" >&2
    keys="$(read_artifact_keys "$base_url" "${prefix}7")"
    if [[ -z "$keys" ]]; then
        echo "No TheRock Linux tarballs found for target '${artifact_target}' at ${base_url}" >&2
        exit 1
    fi

    local key version build
    while IFS= read -r key; do
        [[ -z "$key" ]] && continue
        version="$(parse_artifact_version "$key" "$artifact_target")"
        [[ -z "$version" ]] && continue
        build="$(version_build_number "$version")"
        if [[ "$build" -gt "$latest_build" ]]; then
            latest_build="$build"
            latest_key="$key"
        fi
    done <<< "$keys"

    if [[ -z "$latest_key" ]]; then
        echo "Could not parse any TheRock Linux tarballs for target '${artifact_target}'" >&2
        exit 1
    fi

    echo "$latest_key"
}

if [[ "$VERSION" == "latest" ]]; then
    ARTIFACT_KEY="$(resolve_latest "$ARTIFACT_BASE_URL" "$ARTIFACT_TARGET")"
    ARTIFACT_VERSION="$(parse_artifact_version "$ARTIFACT_KEY" "$ARTIFACT_TARGET")"
else
    ARTIFACT_KEY="therock-dist-linux-${ARTIFACT_TARGET}-${VERSION}.tar.gz"
    ARTIFACT_VERSION="$VERSION"
fi

ARTIFACT_URL="${ARTIFACT_BASE_URL%/}/${ARTIFACT_KEY}"
ARCHIVE_PATH="${PREFIX}/archives/${ARTIFACT_KEY}"
INSTALL_PATH="${PREFIX}/linux-${TARGET}-${ARTIFACT_VERSION}"
ENV_PATH="${INSTALL_PATH}/ollama-therock-env.sh"
CURRENT_LINK="${PREFIX}/linux-current"

if [[ "$RESOLVE_ONLY" == true ]]; then
    echo "Resolved TheRock ROCm artifact:"
    echo "  Version: ${ARTIFACT_VERSION}"
    echo "  Target:  ${TARGET} (${ARTIFACT_TARGET})"
    echo "  URL:     ${ARTIFACT_URL}"
    echo "  ROCm:    ${INSTALL_PATH}"
    exit 0
fi

ensure_archive() {
    local url="$1" archive_path="$2"
    if [[ -f "$archive_path" && "$FORCE" != true ]]; then
        echo "Using cached archive $archive_path"
        return
    fi
    mkdir -p "$(dirname "$archive_path")"
    local partial="${archive_path}.partial"
    echo "Downloading $url"
    curl --location --fail --retry 5 --retry-delay 5 --continue-at - --output "$partial" "$url"
    mv "$partial" "$archive_path"
}

install_archive() {
    local archive_path="$1" install_path="$2"
    local marker="${install_path}/.ollama-therock-ready"
    if [[ -f "$marker" && "$FORCE" != true ]]; then
        echo "Using existing ROCm install $install_path"
        return
    fi
    local parent tmp
    parent="$(dirname "$install_path")"
    mkdir -p "$parent"
    tmp="${parent}/.$(basename "$install_path").tmp"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    echo "Extracting $archive_path"
    tar -xzf "$archive_path" -C "$tmp"
    # Strip single top-level directory wrapper if present (tarball is currently flat, but be defensive)
    readarray -t children < <(find "$tmp" -maxdepth 1 -mindepth 1)
    if [[ ${#children[@]} -eq 1 && -d "${children[0]}" ]]; then
        find "${children[0]}" -maxdepth 1 -mindepth 1 -exec mv -t "$tmp" {} +
        rmdir "${children[0]}"
    fi
    rm -rf "$install_path"
    mv "$tmp" "$install_path"
    date -Iseconds > "$marker"
}

write_environment_file() {
    local install_path="$1" env_path="$2"
    local hip_llvm_bin="${install_path}/lib/llvm/bin"
    local hip_bin="${install_path}/bin"
    local device_lib="${install_path}/lib/llvm/amdgcn/bitcode"

    # amdclang++ is the preferred HIP compiler in TheRock; fall back to clang++
    local cxx_exe="${hip_llvm_bin}/amdclang++"
    if [[ ! -f "$cxx_exe" ]]; then
        cxx_exe="${hip_llvm_bin}/clang++"
    fi

    {
        printf '# Sourceable environment for TheRock ROCm install at %s\n' "$install_path"
        printf 'export HIP_PATH="%s"\n' "$install_path"
        printf 'export HIP_DIR="%s"\n' "$install_path"
        printf 'export ROCM_PATH="%s"\n' "$install_path"
        printf 'export OLLAMA_THEROCK_ROCM_PATH="%s"\n' "$install_path"
        printf 'export HIP_PLATFORM="amd"\n'
        printf 'export HIP_CLANG_PATH="%s"\n' "$hip_llvm_bin"
        printf 'export HIP_DEVICE_LIB_PATH="%s"\n' "$device_lib"
        printf 'export ROCM_DEVICE_LIB_PATH="%s"\n' "$device_lib"
        printf 'export CC="%s"\n' "${hip_llvm_bin}/clang"
        printf 'export CXX="%s"\n' "$cxx_exe"
        printf 'export HIPCXX="%s"\n' "$cxx_exe"
        printf 'export CMAKE_PREFIX_PATH="%s"\n' "$install_path"
        printf 'export PATH="%s:%s:${PATH}"\n' "$hip_llvm_bin" "$hip_bin"
    } > "$env_path"
}

ensure_archive "$ARTIFACT_URL" "$ARCHIVE_PATH"
install_archive "$ARCHIVE_PATH" "$INSTALL_PATH"
write_environment_file "$INSTALL_PATH" "$ENV_PATH"

ln -sfn "$(basename "$INSTALL_PATH")" "$CURRENT_LINK"

{
    printf '{\n'
    printf '  "target": "%s",\n' "$TARGET"
    printf '  "artifactTarget": "%s",\n' "$ARTIFACT_TARGET"
    printf '  "version": "%s",\n' "$ARTIFACT_VERSION"
    printf '  "url": "%s",\n' "$ARTIFACT_URL"
    printf '  "archive": "%s",\n' "$ARCHIVE_PATH"
    printf '  "installPath": "%s",\n' "$INSTALL_PATH"
    printf '  "envFile": "%s"\n' "$ENV_PATH"
    printf '}\n'
} > "${INSTALL_PATH}/ollama-therock-manifest.json"

echo ""
echo "TheRock ROCm is ready:"
echo "  Version: ${ARTIFACT_VERSION}"
echo "  Target:  ${TARGET} (${ARTIFACT_TARGET})"
echo "  ROCm:    ${INSTALL_PATH}"
echo "  Env:     ${ENV_PATH}"
echo "  Link:    ${CURRENT_LINK} -> $(basename "$INSTALL_PATH")"
