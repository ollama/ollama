#!/usr/bin/env bash
#
# Publish a macOS disk image under the stable asset name Ollama.dmg.
#
# The README links to
# https://github.com/maxkongerskov/ollama/releases/latest/download/Ollama.dmg
# which only works when the latest GitHub release has an asset with that
# exact name. Versioned names such as
# Ollama-0.40.5-b11724-ds41.3-darwin-universal.dmg change every release.
#
# The in-app updater never installs a DMG. It stages the versioned .zip
# (macOS) or .exe (Windows). Copying a DMG to Ollama.dmg does not change
# which asset the updater selects.
#
# Usage:
#   scripts/publish_stable_dmg.sh DEST_DIR SOURCE_DMG [VERSIONED_NAME]
#   scripts/publish_stable_dmg.sh --ensure-dist DEST_DIR
#   scripts/publish_stable_dmg.sh --self-test
#
# DEST_DIR SOURCE_DMG copies SOURCE_DMG to DEST_DIR/Ollama.dmg. When
# VERSIONED_NAME is a different file name, also copies SOURCE_DMG to
# DEST_DIR/VERSIONED_NAME. Prints the destination paths, one per line.
#
# --ensure-dist leaves an existing DEST_DIR/Ollama.dmg untouched. Otherwise
# it copies a DMG already in DEST_DIR (preferring a universal build) to
# Ollama.dmg. If DEST_DIR has no DMG, it exits 0 and creates nothing.
#
set -euo pipefail

readonly STABLE_DMG_NAME="Ollama.dmg"

die() { echo "ERROR: $*" >&2; exit 1; }

copy_dmg() {
    local src="$1" dest="$2"
    if [ -e "$dest" ] && [ "$src" -ef "$dest" ]; then
        return 0
    fi
    cp "$src" "$dest"
}

# publish_stable_dmg copies src into dest. It validates every argument
# before writing, so a rejected name does not leave a partial asset behind.
publish_stable_dmg() {
    local dest="$1" src="$2" versioned="${3:-}"
    local stable named

    [ -n "$dest" ] || { echo "ERROR: destination directory is required" >&2; return 1; }
    [ -f "$src" ] || { echo "ERROR: dmg not found: $src" >&2; return 1; }
    case "$src" in
        *.dmg) ;;
        *) echo "ERROR: source is not a .dmg: $src" >&2; return 1 ;;
    esac
    if [ -n "$versioned" ] && [ "$versioned" != "$STABLE_DMG_NAME" ]; then
        case "$versioned" in
            */*) echo "ERROR: versioned DMG name must not contain a path: $versioned" >&2; return 1 ;;
            *.dmg) ;;
            *) echo "ERROR: versioned DMG name must end in .dmg: $versioned" >&2; return 1 ;;
        esac
    fi

    mkdir -p "$dest"
    stable="$dest/$STABLE_DMG_NAME"
    copy_dmg "$src" "$stable"
    printf '%s\n' "$stable"

    if [ -z "$versioned" ] || [ "$versioned" = "$STABLE_DMG_NAME" ]; then
        return 0
    fi
    named="$dest/$versioned"
    copy_dmg "$src" "$named"
    printf '%s\n' "$named"
}

# ensure_stable_dmg makes sure dir/Ollama.dmg exists when dir already
# contains some other DMG. An existing Ollama.dmg is not replaced.
ensure_stable_dmg() {
    local dir="$1" candidate=""
    local -a found=()

    [ -d "$dir" ] || { echo "ERROR: not a directory: $dir" >&2; return 1; }
    if [ -f "$dir/$STABLE_DMG_NAME" ]; then
        printf '%s\n' "$dir/$STABLE_DMG_NAME"
        return 0
    fi

    shopt -s nullglob
    found=("$dir"/*darwin-universal*.dmg)
    if [ "${#found[@]}" -eq 0 ]; then
        found=("$dir"/*-universal.dmg)
    fi
    if [ "${#found[@]}" -eq 0 ]; then
        found=("$dir"/*.dmg)
    fi
    shopt -u nullglob

    if [ "${#found[@]}" -eq 0 ]; then
        echo "no macOS DMG in $dir; stable $STABLE_DMG_NAME not created" >&2
        return 0
    fi
    candidate="${found[0]}"
    publish_stable_dmg "$dir" "$candidate"
}

self_test() {
    local tmp out zip_before root
    tmp="$(mktemp -d)"
    root="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    trap 'rm -rf "$tmp"' RETURN

    printf 'universal-bytes' >"$tmp/src.dmg"
    out="$(publish_stable_dmg "$tmp/out" "$tmp/src.dmg" "Ollama-1.2.3-darwin-universal.dmg")"
    printf '%s\n' "$out" | grep -qx "$tmp/out/$STABLE_DMG_NAME"
    printf '%s\n' "$out" | grep -qx "$tmp/out/Ollama-1.2.3-darwin-universal.dmg"
    cmp -s "$tmp/out/$STABLE_DMG_NAME" "$tmp/out/Ollama-1.2.3-darwin-universal.dmg"
    cmp -s "$tmp/src.dmg" "$tmp/out/$STABLE_DMG_NAME"
    [ "$(printf '%s\n' "$out" | wc -l | tr -d ' ')" = "2" ]

    # The stable name alone must not be emitted twice.
    rm -rf "$tmp/same"
    out="$(publish_stable_dmg "$tmp/same" "$tmp/src.dmg" "$STABLE_DMG_NAME")"
    [ "$out" = "$tmp/same/$STABLE_DMG_NAME" ]
    [ "$(find "$tmp/same" -type f | wc -l | tr -d ' ')" = "1" ]

    # A path or a non-dmg name is rejected, and nothing is published.
    # Run the script itself so a failed publish cannot exit this test early.
    mkdir -p "$tmp/bad"
    if "$root" "$tmp/bad" "$tmp/src.dmg" "../Ollama.dmg" >/dev/null 2>&1; then
        die "self-test: path versioned name was accepted"
    fi
    if [ -e "$tmp/bad/$STABLE_DMG_NAME" ] || [ -e "$tmp/Ollama.dmg" ]; then
        die "self-test: rejected versioned name still published a DMG"
    fi
    if "$root" "$tmp/bad" "$tmp/src.dmg" "notes.txt" >/dev/null 2>&1; then
        die "self-test: non-dmg versioned name was accepted"
    fi
    [ ! -e "$tmp/bad/$STABLE_DMG_NAME" ]
    printf 'not-a-dmg' >"$tmp/src.zip"
    if "$root" "$tmp/bad" "$tmp/src.zip" >/dev/null 2>&1; then
        die "self-test: non-dmg source was accepted"
    fi

    # ensure-dist prefers the universal image and does not touch a zip.
    mkdir -p "$tmp/dist"
    printf 'arm-bytes' >"$tmp/dist/Ollama-1.2.3-darwin-arm64.dmg"
    printf 'uni-bytes' >"$tmp/dist/Ollama-1.2.3-darwin-universal.dmg"
    printf 'zip-bytes' >"$tmp/dist/Ollama-1.2.3-darwin-universal.zip"
    zip_before="$(cat "$tmp/dist/Ollama-1.2.3-darwin-universal.zip")"
    out="$(ensure_stable_dmg "$tmp/dist")"
    [ "$out" = "$tmp/dist/$STABLE_DMG_NAME" ]
    cmp -s "$tmp/dist/$STABLE_DMG_NAME" "$tmp/dist/Ollama-1.2.3-darwin-universal.dmg"
    [ "$(cat "$tmp/dist/Ollama-1.2.3-darwin-universal.zip")" = "$zip_before" ]
    [ "$(cat "$tmp/dist/Ollama-1.2.3-darwin-arm64.dmg")" = "arm-bytes" ]

    # An existing stable DMG is left in place, even if another DMG differs.
    printf 'signed-bytes' >"$tmp/dist/$STABLE_DMG_NAME"
    out="$(ensure_stable_dmg "$tmp/dist")"
    [ "$out" = "$tmp/dist/$STABLE_DMG_NAME" ]
    [ "$(cat "$tmp/dist/$STABLE_DMG_NAME")" = "signed-bytes" ]

    # No DMG: success, and a neighboring zip is unchanged.
    mkdir -p "$tmp/empty"
    printf 'zip-only' >"$tmp/empty/Ollama-1.2.3-darwin-universal.zip"
    out="$(ensure_stable_dmg "$tmp/empty")"
    [ -z "$out" ]
    [ ! -e "$tmp/empty/$STABLE_DMG_NAME" ]
    [ "$(cat "$tmp/empty/Ollama-1.2.3-darwin-universal.zip")" = "zip-only" ]

    # Stable and versioned names are disk images, so the updater's zip/exe
    # filter cannot select them.
    case "$STABLE_DMG_NAME" in
        *.zip|*.exe) die "self-test: stable name looks installable by the updater" ;;
        *.dmg) ;;
        *) die "self-test: stable name is not a dmg" ;;
    esac

    echo "self-test ok"
}

case "${1:-}" in
    --self-test) self_test ;;
    --ensure-dist)
        [ $# -eq 2 ] || die "--ensure-dist needs a directory"
        ensure_stable_dmg "$2"
        ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//' ;;
    *)
        [ $# -eq 2 ] || [ $# -eq 3 ] || die "usage: $0 DEST_DIR SOURCE_DMG [VERSIONED_NAME]"
        publish_stable_dmg "$@"
        ;;
esac
