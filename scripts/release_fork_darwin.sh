#!/bin/bash
#
# Publish a macOS app update for this fork on GitHub releases of
# maxkongerskov/ollama, in the form the in-app updater installs
# (app/updater/updater.go, app/updater/updater_darwin.go):
#
#   - a .zip asset whose name contains "darwin" and the arch (arm64/amd64),
#     or no arch at all for a universal build,
#   - with Ollama.app/ at the root of the zip and nothing else,
#   - a valid code signature on the extracted Ollama.app,
#   - built from exactly the release tag, so the updated app reports that
#     tag as its version and is not offered the same update again.
#
# A DMG passed with --dmg is uploaded under its versioned name and again as
# Ollama.dmg. The README links at releases/latest/download/Ollama.dmg. The
# updater ignores both DMGs and installs the zip.
#
# The release must be a normal (not draft, not prerelease) release, because
# the app only looks at /repos/maxkongerskov/ollama/releases/latest.
#
# This script never talks to ollama/ollama or ollama.com.
#
# Usage: scripts/release_fork_darwin.sh [options]
#   --tag TAG              release tag (default: tag at HEAD)
#   --app PATH             reuse an already built, signed Ollama.app
#                          (default: dist/Ollama.app)
#   --build                build first with scripts/build_darwin.sh (HEAD must be the tag)
#   --archs "arm64"        archs passed to build_darwin.sh -a (default: arm64)
#   --dmg PATH             also upload this DMG as Ollama-<version>-darwin-<arch>.dmg
#                          and as Ollama.dmg (stable README download name)
#   --out DIR              output dir (default: dist/fork-release/TAG)
#   --notarize auto|yes|no notarize + staple the app before zipping (default: auto,
#                          i.e. when credentials are configured)
#   --notary-profile NAME  notarytool keychain profile (or NOTARY_PROFILE env);
#                          otherwise APPLE_ID, APPLE_PASSWORD, APPLE_TEAM_ID env
#   --notes TEXT           release notes for a new release
#   --dry-run              build and verify the zip, but do not notarize or upload
#   --no-go-verify         skip running the updater's own VerifyDownload on the zip
#
set -euo pipefail

readonly REPO="maxkongerskov/ollama"
readonly EXPECTED_TEAM_ID="${EXPECTED_TEAM_ID:-6LZ2DS9JPD}"

status() { echo >&2 ">>> $*"; }
warn() { echo >&2 "WARNING: $*"; }
die() { echo >&2 "ERROR: $*"; exit 1; }
usage() { sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; exit "${1:-0}"; }

TAG=""
APP=""
BUILD=0
ARCHS="arm64"
DMG=""
OUT=""
NOTARIZE="auto"
NOTARY_PROFILE="${NOTARY_PROFILE:-}"
NOTES=""
DRY_RUN=0
GO_VERIFY=1

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --app) APP="$2"; shift 2 ;;
        --build) BUILD=1; shift ;;
        --archs) ARCHS="$2"; shift 2 ;;
        --dmg) DMG="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --notarize) NOTARIZE="$2"; shift 2 ;;
        --notary-profile) NOTARY_PROFILE="$2"; shift 2 ;;
        --notes) NOTES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-go-verify) GO_VERIFY=0; shift ;;
        -h|--help) usage 0 ;;
        *) echo >&2 "unknown option: $1"; usage 1 ;;
    esac
done

[ "$(uname -s)" = "Darwin" ] || die "macOS only"
case "$NOTARIZE" in auto|yes|no) ;; *) die "--notarize must be auto, yes or no" ;; esac

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Only ever publish to this fork.
[ "$REPO" = "maxkongerskov/ollama" ] || die "refusing to publish to $REPO"
ORIGIN_URL="$(git remote get-url origin)"
case "$ORIGIN_URL" in
    https://github.com/maxkongerskov/ollama|https://github.com/maxkongerskov/ollama.git|git@github.com:maxkongerskov/ollama.git) ;;
    *) die "origin is $ORIGIN_URL, expected github.com/maxkongerskov/ollama" ;;
esac
unset GH_REPO GH_HOST
gh_fork() { gh "$@" --repo "$REPO"; }

if [ -z "$TAG" ]; then
    TAG="$(git describe --tags --exact-match HEAD 2>/dev/null)" ||
        die "HEAD is not tagged; pass --tag (e.g. git tag v0.35.1-b11325-glm5next-foo && git push origin v0.35.1-b11325-glm5next-foo)"
fi
VERSION="${TAG#v}"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || die "tag $TAG does not exist locally"
if ! git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
    if [ "$DRY_RUN" = 1 ]; then
        warn "tag $TAG is not pushed to origin yet (required before publishing)"
    else
        die "tag $TAG is not pushed to origin; run: git push origin $TAG"
    fi
fi

OUT="${OUT:-$ROOT/dist/fork-release/$TAG}"

if [ "$BUILD" = 1 ]; then
    [ -z "$APP" ] || die "use either --build or --app"
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TAG^{commit}")" ] ||
        die "--build needs HEAD at $TAG (git checkout $TAG)"
    [ -n "${APPLE_IDENTITY:-}" ] ||
        warn "APPLE_IDENTITY is not set; build_darwin.sh will produce an unsigned app the updater rejects"
    BUILD_START="$(date +%s)"
    status "building Ollama.app ($ARCHS) with scripts/build_darwin.sh"
    ./scripts/build_darwin.sh -a "$ARCHS"
    if [ -z "$DMG" ] && [ -f dist/Ollama.dmg ] &&
        [ "$(stat -f %m dist/Ollama.dmg)" -ge "$BUILD_START" ]; then
        DMG="dist/Ollama.dmg"
    fi
fi
APP="${APP:-$ROOT/dist/Ollama.app}"
[ -d "$APP" ] || die "app not found: $APP (build it, or pass --app)"
APP="$(cd "$(dirname "$APP")" && pwd)/$(basename "$APP")"

# The app must be built from exactly this tag. The updater compares the
# release tag to the running version minus the git describe suffix; a
# mismatch would make the updated app keep offering the same update.
APP_VERSION="$(plutil -extract CFBundleShortVersionString raw "$APP/Contents/Info.plist")"
APP_TAG="$(printf '%s' "$APP_VERSION" | sed -E -e 's/^v//' -e 's/-[0-9]+-g[0-9a-f]+(-dirty)?$//' -e 's/-dirty$//')"
[ "$APP_TAG" = "$VERSION" ] ||
    die "$APP is version $APP_VERSION (tag $APP_TAG), not $TAG; build from the tag or pass the matching --tag"
case "$APP_VERSION" in
    *-dirty) warn "app was built from a dirty tree ($APP_VERSION)" ;;
esac
case "$APP_VERSION" in
    "$VERSION"|"$VERSION"-0-g*) ;;
    *) warn "app was built from commits after $TAG ($APP_VERSION)" ;;
esac

status "checking code signature of $APP"
codesign --verify --deep --strict "$APP" || die "$APP has an invalid signature"
TEAM_ID="$(codesign -dv "$APP" 2>&1 | sed -n 's/^TeamIdentifier=//p')"
[ -n "$TEAM_ID" ] && [ "$TEAM_ID" != "not set" ] ||
    die "$APP is not signed with a Developer ID (ad-hoc/unsigned)"
[ "$TEAM_ID" = "$EXPECTED_TEAM_ID" ] ||
    die "$APP is signed by team $TEAM_ID, expected $EXPECTED_TEAM_ID (set EXPECTED_TEAM_ID to override)"

APP_ARCHS="$(lipo -archs "$APP/Contents/MacOS/Ollama")"
case "$APP_ARCHS" in
    arm64) ARCH="arm64" ;;
    x86_64) ARCH="amd64" ;;
    *arm64*x86_64*|*x86_64*arm64*) ARCH="universal" ;;
    *) die "unexpected app archs: $APP_ARCHS" ;;
esac
BASENAME="Ollama-$VERSION-darwin-$ARCH"
ZIP="$OUT/$BASENAME.zip"

rm -rf "$OUT"
mkdir -p "$OUT/stage"
status "staging a copy of the app in $OUT/stage"
ditto "$APP" "$OUT/stage/Ollama.app"
STAGED="$OUT/stage/Ollama.app"

# --norsrc, not --sequesterRsrc: the updater rejects any zip entry outside
# Ollama.app/, and --sequesterRsrc adds __MACOSX/ entries.
make_zip() { rm -f "$2"; ditto -c -k --norsrc --keepParent "$1" "$2"; }

HAVE_NOTARY_CREDS=0
if [ -n "$NOTARY_PROFILE" ] || { [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_PASSWORD:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; }; then
    HAVE_NOTARY_CREDS=1
fi
if xcrun stapler validate "$STAGED" >/dev/null 2>&1; then
    status "app is already notarized and stapled"
elif [ "$NOTARIZE" = "no" ] || { [ "$NOTARIZE" = "auto" ] && [ "$HAVE_NOTARY_CREDS" = 0 ]; }; then
    status "not notarizing (the in-app updater only needs a valid signature; a DMG download still needs notarization for Gatekeeper)"
elif [ "$HAVE_NOTARY_CREDS" = 0 ]; then
    die "--notarize yes needs --notary-profile or APPLE_ID/APPLE_PASSWORD/APPLE_TEAM_ID"
elif [ "$DRY_RUN" = 1 ]; then
    status "dry run: would notarize and staple $STAGED"
else
    status "notarizing $STAGED"
    make_zip "$STAGED" "$OUT/notarize.zip"
    if [ -n "$NOTARY_PROFILE" ]; then
        xcrun notarytool submit "$OUT/notarize.zip" --wait --timeout 30m --keychain-profile "$NOTARY_PROFILE"
    else
        xcrun notarytool submit "$OUT/notarize.zip" --wait --timeout 30m \
            --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID"
    fi
    rm -f "$OUT/notarize.zip"
    xcrun stapler staple "$STAGED"
fi

status "creating $ZIP"
make_zip "$STAGED" "$ZIP"

status "verifying $ZIP"
BAD_ENTRIES="$(zipinfo -1 "$ZIP" | grep -v -e '^Ollama\.app/' || true)"
[ -z "$BAD_ENTRIES" ] || die "zip has entries outside Ollama.app/: $BAD_ENTRIES"
mkdir -p "$OUT/verify"
unzip -q "$ZIP" -d "$OUT/verify"
codesign --verify --deep --strict "$OUT/verify/Ollama.app" || die "extracted Ollama.app fails codesign --verify"
[ "$(plutil -extract CFBundleShortVersionString raw "$OUT/verify/Ollama.app/Contents/Info.plist")" = "$APP_VERSION" ] ||
    die "extracted app version mismatch"
rm -rf "$OUT/verify"

if [ "$GO_VERIFY" = 1 ] && command -v go >/dev/null 2>&1; then
    # Run the updater's own download verification on the zip, staged the way
    # DownloadNewRelease stages it.
    status "verifying with app/updater.VerifyDownload"
    mkdir -p "$OUT/_goverify/stage/etag" "$OUT/_goverify/cmd"
    cp "$ZIP" "$OUT/_goverify/stage/etag/"
    cat >"$OUT/_goverify/cmd/main.go" <<'GO'
package main

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/app/updater"
)

func main() {
	updater.UpdateStageDir = os.Args[1]
	if err := updater.VerifyDownload(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
GO
    go run "$OUT/_goverify/cmd/main.go" "$OUT/_goverify/stage" || die "updater VerifyDownload rejected $ZIP"
    rm -rf "$OUT/_goverify"
fi
rm -rf "$OUT/stage"

ASSETS=("$ZIP")
if [ -n "$DMG" ]; then
    [ -f "$DMG" ] || die "dmg not found: $DMG"
    # Versioned name plus Ollama.dmg. The updater installs $BASENAME.zip only;
    # Ollama.dmg is the stable URL in the README. Fail the release if either
    # copy is missing, instead of publishing a versioned DMG alone.
    dmg_list="$("$ROOT/scripts/publish_stable_dmg.sh" "$OUT" "$DMG" "$BASENAME.dmg")" \
        || die "failed to stage DMG assets"
    stable_ok=0
    versioned_ok=0
    while IFS= read -r asset; do
        [ -n "$asset" ] || continue
        [ -f "$asset" ] || die "staged DMG missing: $asset"
        ASSETS+=("$asset")
        case "$(basename "$asset")" in
            Ollama.dmg) stable_ok=1 ;;
            "$BASENAME.dmg") versioned_ok=1 ;;
        esac
    done <<< "$dmg_list"
    [ "$stable_ok" = 1 ] || die "stable Ollama.dmg was not staged"
    [ "$versioned_ok" = 1 ] || die "versioned DMG $BASENAME.dmg was not staged"
fi
(cd "$OUT" && shasum -a 256 "${ASSETS[@]##*/}" >sha256sum.txt)
ASSETS+=("$OUT/sha256sum.txt")

status "assets for $REPO $TAG:"
for f in "${ASSETS[@]}"; do ls -l "$f" >&2; done

if [ -n "$DMG" ]; then
    NOTES="${NOTES:-Ollama $VERSION ($REPO). The app updates itself from the $BASENAME.zip asset. Ollama.dmg is the same disk image as $BASENAME.dmg, for a stable download URL.}"
else
    NOTES="${NOTES:-Ollama $VERSION ($REPO). The app updates itself from the $BASENAME.zip asset.}"
fi
if [ "$DRY_RUN" = 1 ]; then
    status "dry run: not uploading. Would run:"
    if gh_fork release view "$TAG" >/dev/null 2>&1; then
        echo "  gh release upload $TAG ${ASSETS[*]} --repo $REPO --clobber"
    else
        echo "  gh release create $TAG ${ASSETS[*]} --repo $REPO --verify-tag --latest --title $TAG --notes '$NOTES'"
    fi
    exit 0
fi

if gh_fork release view "$TAG" >/dev/null 2>&1; then
    status "uploading to existing release $TAG on $REPO"
    gh_fork release upload "$TAG" "${ASSETS[@]}" --clobber
else
    status "creating release $TAG on $REPO"
    gh_fork release create "$TAG" "${ASSETS[@]}" --verify-tag --latest --title "$TAG" --notes "$NOTES"
fi

LATEST="$(gh api "repos/$REPO/releases/latest" --jq .tag_name)"
if [ "$LATEST" != "$TAG" ]; then
    warn "latest release on $REPO is $LATEST, not $TAG; the app only installs the latest release (gh release edit $TAG --repo $REPO --draft=false --prerelease=false --latest)"
fi
status "done: apps older than $TAG pick this up on their next update check (hourly, and 3s after launch)"
