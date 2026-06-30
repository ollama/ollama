#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

assert_version() {
    local input=$1
    local expected=$2
    local actual

    actual=$(printf '%s\n' "$input" | sh "$script_dir/version.sh")
    if [[ "$actual" != "$expected" ]]; then
        printf 'FAIL %s: expected %s, got %s\n' "$input" "$expected" "$actual" >&2
        return 1
    fi
}

assert_version "v0.30.10-0-ge1f7f9c" "0.30.11-0.0.ge1f7f9c"
assert_version "v0.30.10-12-gabcdef0" "0.30.11-0.12.gabcdef0"
assert_version "v0.30.10-12-gabcdef0-dirty" "0.30.11-0.12.gabcdef0.dirty"
assert_version "v0.30.11-rc0-0-ge11eeb3" "0.30.11-0.rc0.0.ge11eeb3"
assert_version "v0.30.11-rc0-5-gd9075ca" "0.30.11-0.rc0.5.gd9075ca"
assert_version "v0.30.11-rc1-5-gABCDEF0-dirty" "0.30.11-0.rc1.5.gABCDEF0.dirty"
assert_version "v0.30.11" "0.30.11"
assert_version "d26a585" "d26a585"
assert_version "d26a585-dirty" "d26a585-dirty"
