#!/usr/bin/env bash

# Prepare prebuilt MLX Metal runtime libraries for macOS CI unit tests.
#
# Building MLX is expensive, so to enable the MLX-specific unit tests this
# helper finds the newest Ollama release whose MLX_VERSION and MLX_C_VERSION
# match the current checkout, downloads that release's ollama-darwin.tgz, and
# extracts only mlx_metal_v* into build/lib/ollama
#
# If no matching release artifact exists, the helper only emits a warning. That
# covers the expected window after an MLX update lands and before the next release
# publishes a matching Darwin payload.

set -euo pipefail

repo="${OLLAMA_MLX_RELEASE_REPO:-ollama/ollama}"
scan_limit="${OLLAMA_MLX_RELEASE_SCAN_LIMIT:-50}"
cache_dir="${OLLAMA_MLX_DARWIN_CACHE:-.cache/mlx-darwin-release}"
target_dir="${OLLAMA_MLX_DARWIN_TARGET:-build/lib/ollama}"
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
  has_payload
}

extract_payload() {
  local tag="$1"
  tmpdir="$(mktemp -d)"

  tar -xzf "${tarball}" -C "${tmpdir}"
  mkdir -p "${target_dir}"

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

if [ "$(uname -s)" != "Darwin" ]; then
  warn "MLX Darwin payload setup is only supported on macOS"
  exit 0
fi

current_mlx="$(read_pin MLX_VERSION)"
current_mlxc="$(read_pin MLX_C_VERSION)"
current_pins="${current_mlx} ${current_mlxc}"

if has_matching_payload; then
  echo "MLX Darwin payload already present in ${target_dir}"
  exit 0
fi

mkdir -p "${cache_dir}"

if [ -s "${tarball}" ] && [ -f "${tag_file}" ] && [ "$(cat "${pins_file}" 2>/dev/null || true)" = "${current_pins}" ]; then
  extract_payload "$(cat "${tag_file}")"
  exit 0
fi

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
  warn "MLX unit tests are temporarily disabled until a release build publishes ollama-darwin.tgz for MLX_VERSION=${current_mlx} MLX_C_VERSION=${current_mlxc}"
  exit 0
fi

tmp_tarball="${tarball}.tmp"
rm -f "${tmp_tarball}"
curl -fL --retry 3 --retry-delay 2 -o "${tmp_tarball}" "${matched_url}"
mv "${tmp_tarball}" "${tarball}"
tmp_tarball=""
echo "${matched_tag}" >"${tag_file}"
echo "${current_pins}" >"${pins_file}"

extract_payload "${matched_tag}"
