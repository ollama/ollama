#!/usr/bin/env bash

# Prepare MLX runtime libraries for macOS CI unit tests.
#
# Building MLX is expensive, so to enable the MLX-specific unit tests this
# helper finds the newest Ollama release whose MLX_VERSION and MLX_C_VERSION
# match the current checkout, downloads that release's ollama-darwin.tgz, and
# extracts only mlx_metal_v* into build/lib/ollama.
#
# The payload's libollama_xgrammar.dylib is built from Ollama's native
# wrapper (x/mlxrunner/xgrammar/native) and the XGrammar pin recorded in
# cmake/mlx/CMakeLists.txt. When the matched release tag differs from the
# checkout in either, the helper rebuilds just that library from source
# instead of the whole MLX build.
#
# If no release matches the MLX pins (e.g. right after a pin bump), the
# helper builds the minimal MLX payload for this platform: a single Metal
# variant using the superbuild's platform default (metal_v4 on macOS 26.2+
# SDKs, otherwise metal_v3), including a fresh libollama_xgrammar.dylib.

set -euo pipefail

repo="${OLLAMA_MLX_RELEASE_REPO:-ollama/ollama}"
scan_limit="${OLLAMA_MLX_RELEASE_SCAN_LIMIT:-50}"
cache_dir="${OLLAMA_MLX_DARWIN_CACHE:-.cache/mlx-darwin-release}"
target_dir="${OLLAMA_MLX_DARWIN_TARGET:-build/lib/ollama}"
ci_build_dir="${OLLAMA_MLX_CI_BUILD_DIR:-build/mlx-ci}"
tarball="${cache_dir}/ollama-darwin.tgz"
tag_file="${cache_dir}/matched-tag"
pins_file="${cache_dir}/matched-pins"
target_pins_file="${target_dir}/.mlx-release-pins"
tmpdir=""
tmp_tarball=""

cleanup() {
  [ -z "${tmpdir}" ] || rm -rf "${tmpdir}"
  [ -z "${tmp_tarball}" ] || rm -f "${tmp_tarball}"
}
trap cleanup EXIT

warn() {
  if [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "::warning::$*"
  else
    echo "warning: $*" >&2
  fi
}

read_pin() {
  tr -d '[:space:]' <"$1"
}

# Native wrapper sources compiled into libollama_xgrammar.dylib — keep in
# sync with the ollama_xgrammar target in cmake/mlx/CMakeLists.txt.
xgrammar_native_dir=x/mlxrunner/xgrammar/native

# Paths whose content determines libollama_xgrammar.dylib: the build rules
# (pinned version, target sources, compile definitions) and every file under
# the native wrapper directory.
xgrammar_inputs() {
  printf '%s\n' cmake/mlx/CMakeLists.txt
  find "${xgrammar_native_dir}" -type f | sort
}

xgrammar_fingerprint() {
  local file
  {
    xgrammar_inputs
    while IFS= read -r file; do
      cat "${file}" 2>/dev/null || true
    done < <(xgrammar_inputs)
  } | shasum -a 256 | awk '{print $1}'
}

# True when the tag carries the same XGrammar inputs as the checked-out tree.
# Files present in the tree but not at the tag fail the fetch and count as a
# mismatch.
tag_xgrammar_matches() {
  local tag="$1" file
  while IFS= read -r file; do
    if ! curl -fsSL "https://raw.githubusercontent.com/${repo}/${tag}/${file}" 2>/dev/null | cmp -s - "${file}"; then
      return 1
    fi
  done < <(xgrammar_inputs)
  return 0
}

has_payload() {
  local variant
  for variant in "${target_dir}"/mlx_metal_v*; do
    [ -d "${variant}" ] || continue
    [ -f "${variant}/libmlx.dylib" ] && [ -f "${variant}/libmlxc.dylib" ] && return 0
  done
  return 1
}

has_matching_payload() {
  [ -f "${target_pins_file}" ] || return 1
  [ "$(cat "${target_pins_file}")" = "${current_pins}" ] || return 1
  has_payload || return 1
  # Every payload variant must carry libollama_xgrammar.dylib.
  local variant
  for variant in "${target_dir}"/mlx_metal_v*; do
    [ -d "${variant}" ] || continue
    [ -f "${variant}/libollama_xgrammar.dylib" ] || return 1
  done
  return 0
}

extract_payload() {
  local tag="$1"
  tmpdir="$(mktemp -d)"

  tar -xzf "${tarball}" -C "${tmpdir}"
  mkdir -p "${target_dir}"

  rm -rf "${target_dir}"/mlx_metal_v*

  local found=false
  local src dest
  for src in "${tmpdir}"/mlx_metal_v*; do
    [ -d "${src}" ] || continue
    found=true
    dest="${target_dir}/$(basename "${src}")"
    rm -rf "${dest}"
    cp -R "${src}" "${dest}"
  done

  if [ "${found}" != true ] || ! has_payload; then
    echo "Downloaded ${tarball} did not contain a usable MLX Metal payload" >&2
    exit 1
  fi

  echo "${current_pins}" >"${target_pins_file}"
  echo "Prepared MLX Darwin payload from ${repo} ${tag}:"
  find "${target_dir}" -maxdepth 2 -type f \( -name 'libmlx.dylib' -o -name 'libmlxc.dylib' -o -name '*.metallib' \) -print

  rm -rf "${tmpdir}"
  tmpdir=""
}

# Resolve the superbuild's platform-default MLX backend (metal_v3/metal_v4 on
# arm64; empty when the platform has no MLX backend, e.g. x86_64 macOS).
ci_mlx_backend() {
  [ -f "${ci_build_dir}/CMakeCache.txt" ] || return 1
  sed -n 's/^OLLAMA_MLX_BACKENDS:STRING=//p' "${ci_build_dir}/CMakeCache.txt"
}

# Configure the repo-root superbuild and fetch MLX/MLX-C sources at the
# pinned revisions (only the full payload build needs this).
build_ci_sources() {
  cmake -S . -B "${ci_build_dir}" \
    -DOLLAMA_LLAMA_BACKENDS= \
    -DOLLAMA_PAYLOAD_INSTALL_PREFIX="$(dirname "$(dirname "${target_dir}")")"
  cmake --build "${ci_build_dir}" --target ollama-mlx-sources --parallel
}

# Rebuild only libollama_xgrammar.dylib into the extracted payload. The
# target depends only on the pinned XGrammar sources and the native wrapper;
# the Metal toolchain and the superbuild are not involved. MLX is fetched
# only because the cmake/mlx project defines it — nothing from it is built.
build_ci_xgrammar() {
  local lib variant
  local xg_build_dir="${ci_build_dir}/xgrammar"
  local -a configure_args=(-S cmake/mlx -B "${xg_build_dir}" -DOLLAMA_SOURCE_DIR="$(pwd)" -DMLX_BUILD_METAL=OFF)
  if [ -n "${OLLAMA_XGRAMMAR_SOURCE:-}" ]; then
    configure_args+=("-DFETCHCONTENT_SOURCE_DIR_XGRAMMAR=${OLLAMA_XGRAMMAR_SOURCE}")
  fi
  cmake "${configure_args[@]}"
  cmake --build "${xg_build_dir}" --target ollama_xgrammar --parallel
  lib="${xg_build_dir}/lib/ollama/libollama_xgrammar.dylib"
  [ -f "${lib}" ] || {
    echo "ollama_xgrammar build produced no library at ${lib}" >&2
    exit 1
  }
  for variant in "${target_dir}"/mlx_metal_v*; do
    [ -d "${variant}" ] || continue
    cp -f "${lib}" "${variant}/libollama_xgrammar.dylib"
    [ -f "${variant}/libollama_xgrammar.dylib" ] || {
      echo "failed to install ${variant}/libollama_xgrammar.dylib" >&2
      exit 1
    }
  done
  echo "Rebuilt libollama_xgrammar.dylib from source into ${target_dir}"
}

# Build the minimal MLX payload for this platform: one Metal variant,
# whatever the superbuild defaults to here.
build_ci_payload() {
  local backend variant
  build_ci_sources
  backend="$(ci_mlx_backend)"
  case "${backend}" in
    metal_v3 | metal_v4) ;;
    *)
      warn "no MLX backend applicable to this platform; MLX unit tests will be skipped"
      exit 0
      ;;
  esac
  echo "No release carries MLX_VERSION=${current_mlx} MLX_C_VERSION=${current_mlxc}; building the ${backend} payload for unit tests"
  cmake --build "${ci_build_dir}" --target "ollama-mlx-${backend}" --parallel
  for variant in "${target_dir}"/mlx_metal_v*; do
    [ -d "${variant}" ] || continue
    for lib in libmlx.dylib libmlxc.dylib libollama_xgrammar.dylib; do
      [ -f "${variant}/${lib}" ] || {
        echo "built payload is missing ${variant}/${lib}" >&2
        exit 1
      }
    done
  done
  has_payload || {
    echo "built payload is incomplete in ${target_dir}" >&2
    exit 1
  }
  echo "${current_pins}" >"${target_pins_file}"
  echo "Built MLX payload for unit tests:"
  find "${target_dir}" -maxdepth 2 -type f \( -name 'libmlx.dylib' -o -name 'libmlxc.dylib' -o -name 'libollama_xgrammar.dylib' -o -name '*.metallib' \) -print
}

if [ "$(uname -s)" != "Darwin" ]; then
  warn "MLX Darwin payload setup is only supported on macOS"
  exit 0
fi

current_mlx="$(read_pin MLX_VERSION)"
current_mlxc="$(read_pin MLX_C_VERSION)"
# The release tarball only depends on the MLX pins; the extracted payload's
# xgrammar library additionally depends on the tree's XGrammar inputs.
component_pins="${current_mlx} ${current_mlxc}"
current_pins="${component_pins} $(xgrammar_fingerprint)"

if has_matching_payload; then
  echo "MLX payload already present in ${target_dir}"
  exit 0
fi

mkdir -p "${cache_dir}"

if [ -s "${tarball}" ] && [ -f "${tag_file}" ] && [ "$(cat "${pins_file}" 2>/dev/null || true)" = "${component_pins}" ]; then
  extract_payload "$(cat "${tag_file}")"
else
  matched_tag=""
  matched_url=""

  while read -r tag; do
    [ -n "${tag}" ] || continue

    if ! tag_mlx="$(curl -fsSL "https://raw.githubusercontent.com/${repo}/${tag}/MLX_VERSION" | tr -d '[:space:]')"; then
      continue
    fi
    if [ "${tag_mlx}" != "${current_mlx}" ]; then
      continue
    fi

    if ! tag_mlxc="$(curl -fsSL "https://raw.githubusercontent.com/${repo}/${tag}/MLX_C_VERSION" | tr -d '[:space:]')"; then
      continue
    fi
    if [ "${tag_mlxc}" != "${current_mlxc}" ]; then
      continue
    fi

    url="https://github.com/${repo}/releases/download/${tag}/ollama-darwin.tgz"
    if curl -fsIL "${url}" >/dev/null; then
      matched_tag="${tag}"
      matched_url="${url}"
      break
    fi

    echo "MLX pins match ${tag}, but ${url} is not available"
  done < <(
    git ls-remote --tags --refs --sort=-version:refname "https://github.com/${repo}.git" 'v*' |
      awk -v limit="${scan_limit}" '{ sub("refs/tags/", "", $2); print $2; if (limit > 0 && NR >= limit) exit }'
  )

  if [ -z "${matched_tag}" ]; then
    build_ci_payload
    exit 0
  fi

  tmp_tarball="${tarball}.tmp"
  rm -f "${tmp_tarball}"
  curl -fL --retry 3 --retry-delay 2 -o "${tmp_tarball}" "${matched_url}"
  mv "${tmp_tarball}" "${tarball}"
  tmp_tarball=""
  echo "${matched_tag}" >"${tag_file}"
  echo "${component_pins}" >"${pins_file}"

  extract_payload "${matched_tag}"
fi

if tag_xgrammar_matches "$(cat "${tag_file}")"; then
  exit 0
fi

# Replace the payload's libollama_xgrammar.dylib with a build from the
# checked-out tree.
if [ "$(uname -m)" != "arm64" ]; then
  warn "MLX unit tests need a fresh libollama_xgrammar, but only arm64 macOS can build the payload; tests will be skipped"
  exit 0
fi

echo "Release payload XGrammar does not match this checkout; rebuilding libollama_xgrammar from source"
rm -f "${target_dir}"/mlx_metal_v*/libollama_xgrammar.dylib
build_ci_xgrammar