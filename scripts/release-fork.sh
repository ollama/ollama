#!/bin/bash
# One command on the release Mac. It does not run from CI or on every push.
#
#   bash scripts/release-fork.sh patch
#   bash scripts/release-fork.sh v0.40.5-b11724-ds41.4
#
# patch bumps the latest vX.Y.Z-bNNNNN-ds41[.N] tag. An explicit tag must be
# strictly newer. The script runs the Go gate, the macOS old-to-new rehearsal,
# builds a universal signed app, notarizes with the notarytool keychain
# profile, staples, writes the versioned zip (what the app installs), the
# versioned DMG, the stable Ollama.dmg, sha256sum.txt, and an ed25519
# signature next to each of those. It uploads them to a GitHub Release and
# marks that release latest. Any failure before that mark deletes the tag and
# the draft and publishes nothing.
#
#   bash scripts/release-fork.sh --dry-run patch
#   bash scripts/release-fork.sh --self-test
#
# --dry-run computes the tag and runs the Go gate. It does not tag, build, or
# upload. --self-test is the Linux-runnable part of the update rehearsal.
#
# OLLAMA_SIGN_ID, OLLAMA_NOTARY_PROFILE (default AC_PASSWORD), and
# OLLAMA_UPDATE_KEY (default ~/.config/ollama-fork/update-private.pem)
# override the Developer ID, notary profile, and ed25519 private key.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="maxkongerskov/ollama"
TEAM="6LZ2DS9JPD"
NOTARY_PROFILE="${OLLAMA_NOTARY_PROFILE:-AC_PASSWORD}"
UPDATE_KEY="${OLLAMA_UPDATE_KEY:-$HOME/.config/ollama-fork/update-private.pem}"

die() { echo "ERROR: $*" >&2; exit 1; }
status() { echo "▶ $*"; }

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-1}"
}

origin_path() {
  local url="$1"
  url="${url%.git}"
  case "$url" in
    *://*) url="${url#*://}" ;;
  esac
  url="${url#*@}"
  printf '%s\n' "$url"
}

go_gate() {
  local dirty
  dirty="$(gofmt -l app/updater app/updatetrust app/forkversion scripts/updatesign scripts/forkversion)"
  if [ -n "$dirty" ]; then
    echo "gofmt would change:" >&2
    printf '%s\n' "$dirty" >&2
    exit 1
  fi
  go test -count=1 ./app/updatetrust/ ./app/forkversion/ ./app/updater/
  go vet ./app/updatetrust/ ./app/forkversion/ ./app/updater/
  bash scripts/publish_stable_dmg.sh --self-test >/dev/null
}

cli_checks() {
  local latest next
  latest="$(go run ./scripts/forkversion latest)"
  next="$(go run ./scripts/forkversion next patch)"
  if [ -z "$latest" ] || [ -z "$next" ] || [ "$latest" = "$next" ]; then
    die "forkversion next did not move past ${latest:-<none>}"
  fi
  if go run ./scripts/forkversion next "$latest" >/dev/null 2>&1; then
    die "forkversion accepted the current tag $latest"
  fi
  echo "forkversion: $latest -> $next"
}

MODE=""
SPEC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE="dry"; shift ;;
    --self-test) MODE="self"; shift ;;
    -h|--help) usage 0 ;;
    --) shift; break ;;
    -*) die "unknown option: $1" ;;
    *)
      [ -z "$SPEC" ] || die "unexpected extra argument: $1"
      SPEC="$1"
      shift
      ;;
  esac
done

if [ "$MODE" = "self" ]; then
  [ -z "$SPEC" ] || die "--self-test does not take a version"
  status "Go tests, vet, and version CLI"
  go_gate
  cli_checks
  echo "self-test ok"
  exit 0
fi

[ -n "$SPEC" ] || usage 1

if [ "$MODE" != "dry" ] && [ "$(uname -s)" != "Darwin" ]; then
  die "A release runs on macOS so it can sign, notarize, and staple. Use --dry-run or --self-test here."
fi

status "resolving $SPEC"
if ! git fetch origin --tags; then
  if [ "$MODE" = "dry" ]; then
    echo "WARNING: git fetch origin --tags failed. The next tag is based on local tags only." >&2
  else
    die "git fetch origin --tags failed"
  fi
fi
TAG="$(go run ./scripts/forkversion next "$SPEC")"
VERSION="${TAG#v}"
echo "Next release: $TAG"

if [ "$MODE" = "dry" ]; then
  if [ -n "$(git status --porcelain)" ]; then
    echo "WARNING: working tree is not clean. A real release refuses this." >&2
  fi
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if [ "$branch" = "HEAD" ]; then
    echo "WARNING: detached HEAD. A real release refuses this." >&2
  fi
  if [ ! -f "$UPDATE_KEY" ]; then
    echo "WARNING: no update signing key at $UPDATE_KEY. A real release refuses this. Run bash scripts/generate-update-key.sh on the Mac and commit app/updatetrust/public_key.go." >&2
  fi
  status "Go gate (no tag, build, or upload)"
  go_gate
  cat <<EOF
Dry run only. A real release on the Mac would then:
  1. Require a clean tree on a branch whose origin is $REPO.
  2. Require Developer ID Application team $TEAM, notarytool profile $NOTARY_PROFILE, gh auth, and $UPDATE_KEY matching the embedded public key.
  3. Run bash scripts/e2e-update-mac.sh (old ds41.900 -> ds41.901, plus a tampered zip that must be rejected).
  4. Tag $TAG, build a universal app with scripts/build_darwin.sh, notarize, staple, and check spctl.
  5. Write dist/fork-release/$TAG/ with Ollama-${VERSION}-darwin-universal.zip, the versioned DMG, Ollama.dmg, sha256sum.txt, and a .ed25519 sidecar for each.
  6. Push the tag and publish a GitHub Release only after every check passes.
Nothing was tagged, built, or uploaded.
EOF
  exit 0
fi

if [ -n "$(git status --porcelain)" ]; then
  die "Working tree is not clean. Commit or stash before releasing."
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "HEAD" ]; then
  die "Detached HEAD. Check out a branch first."
fi
ORIGIN="$(origin_path "$(git remote get-url origin)")"
case "$ORIGIN" in
  "github.com/$REPO"|"github.com:$REPO") ;;
  *) die "origin is not $REPO (refusing to publish)" ;;
esac
unset GH_REPO GH_HOST

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  die "Tag $TAG already exists locally"
fi
REMOTE_TAG="$(git ls-remote --tags origin "refs/tags/$TAG")" || die "Could not reach origin to check whether $TAG is already tagged."
if printf '%s\n' "$REMOTE_TAG" | grep -q "refs/tags/${TAG}$"; then
  die "Tag $TAG already exists on origin"
fi

status "preflight (aborting here publishes nothing)"
SIGN_ID="${OLLAMA_SIGN_ID:-}"
if [ -z "$SIGN_ID" ]; then
  SIGN_ID="$(security find-identity -v -p codesigning | awk -v team="$TEAM" 'index($0, team) && /Developer ID Application/ { print $2; exit }')"
fi
[ -n "$SIGN_ID" ] || die "No Developer ID Application identity for team $TEAM"
xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null
gh auth status
[ -f "$UPDATE_KEY" ] || die "Missing $UPDATE_KEY. Run bash scripts/generate-update-key.sh and commit app/updatetrust/public_key.go."
go run ./scripts/updatesign check-embedded "$UPDATE_KEY"

status "Go gate"
go_gate

status "macOS update rehearsal"
bash scripts/e2e-update-mac.sh

status "tagging $TAG locally until publish succeeds"
git tag -a "$TAG" -m "Ollama fork $TAG"

PUBLISHED=0
TAG_PUSHED=0
cleanup() {
  status_code=$?
  if [ "$PUBLISHED" = 1 ]; then
    exit "$status_code"
  fi
  echo "Release failed (exit $status_code). Removing unpublished tag $TAG." >&2
  git tag -d "$TAG" >/dev/null 2>&1 || true
  if [ "$TAG_PUSHED" = 1 ]; then
    echo "Removing remote tag and draft release $TAG." >&2
    git push origin ":refs/tags/$TAG" || true
    gh release delete "$TAG" --repo "$REPO" --yes --cleanup-tag || true
  fi
  exit "$status_code"
}
trap cleanup EXIT

status "building universal app (notarization required)"
APPLE_IDENTITY="$SIGN_ID" NOTARY_PROFILE="$NOTARY_PROFILE" ./scripts/build_darwin.sh
[ -d dist/Ollama.app ] || die "build did not produce dist/Ollama.app"
[ -f dist/Ollama.dmg ] || die "build did not produce dist/Ollama.dmg"

APP_VERSION="$(plutil -extract CFBundleShortVersionString raw dist/Ollama.app/Contents/Info.plist)"
APP_TAG="$(printf '%s' "$APP_VERSION" | sed -E -e 's/^v//' -e 's/-[0-9]+-g[0-9a-f]+(-dirty)?$//' -e 's/-dirty$//')"
[ "$APP_TAG" = "$VERSION" ] || die "built app is $APP_VERSION (tag $APP_TAG), not $TAG"

APP_ARCHS="$(lipo -archs dist/Ollama.app/Contents/MacOS/Ollama)"
case "$APP_ARCHS" in
  *arm64*x86_64*|*x86_64*arm64*) ARCH="universal" ;;
  *) die "release build is not universal ($APP_ARCHS)" ;;
esac
BASENAME="Ollama-${VERSION}-darwin-${ARCH}"
OUT="dist/fork-release/$TAG"
rm -rf "$OUT"
mkdir -p "$OUT"
ditto -c -k --norsrc --keepParent dist/Ollama.app "$OUT/$BASENAME.zip"
dmg_list="$(./scripts/publish_stable_dmg.sh "$OUT" dist/Ollama.dmg "$BASENAME.dmg")" || die "failed to stage DMG assets"
printf '%s\n' "$dmg_list" | grep -qx "$OUT/Ollama.dmg"
printf '%s\n' "$dmg_list" | grep -qx "$OUT/$BASENAME.dmg"
if [ ! -f "$OUT/Ollama.dmg" ] || [ ! -f "$OUT/$BASENAME.dmg" ]; then
  die "DMG assets were not staged"
fi

status "checking Apple signature, staple, and Gatekeeper"
TEAM_ID="$(codesign -dv dist/Ollama.app 2>&1 | sed -n 's/^TeamIdentifier=//p')"
[ "$TEAM_ID" = "$TEAM" ] || die "app team is ${TEAM_ID:-unset}, expected $TEAM"
codesign --verify --deep --strict dist/Ollama.app
xcrun stapler validate dist/Ollama.app
spctl --assess --type execute --verbose=4 dist/Ollama.app
codesign --verify --deep --strict dist/Ollama.dmg
xcrun stapler validate dist/Ollama.dmg
spctl --assess --type open --verbose=4 dist/Ollama.dmg

command -v zipinfo >/dev/null 2>&1 || die "zipinfo is required"
BAD_ENTRIES="$(zipinfo -1 "$OUT/$BASENAME.zip" | grep -v -e '^Ollama\.app/' || true)"
[ -z "$BAD_ENTRIES" ] || die "zip has entries outside Ollama.app/: $BAD_ENTRIES"

status "signing update artifacts"
go run ./scripts/updatesign sign "$UPDATE_KEY" \
  "$OUT/$BASENAME.zip" \
  "$OUT/$BASENAME.dmg" \
  "$OUT/Ollama.dmg"
(
  cd "$OUT"
  shasum -a 256 \
    "$BASENAME.zip" "$BASENAME.zip.ed25519" \
    "$BASENAME.dmg" "$BASENAME.dmg.ed25519" \
    Ollama.dmg Ollama.dmg.ed25519 \
    >sha256sum.txt
)
go run ./scripts/updatesign sign "$UPDATE_KEY" "$OUT/sha256sum.txt"
go run ./scripts/updatesign verify \
  "$OUT/$BASENAME.zip" \
  "$OUT/$BASENAME.dmg" \
  "$OUT/Ollama.dmg" \
  "$OUT/sha256sum.txt"

status "verifying the zip with the app updater"
rm -rf "$OUT/_goverify"
mkdir -p "$OUT/_goverify/stage/etag" "$OUT/_goverify/cmd"
cp "$OUT/$BASENAME.zip" "$OUT/_goverify/stage/etag/"
cp "$OUT/$BASENAME.zip.ed25519" "$OUT/_goverify/stage/etag/"
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
go run "$OUT/_goverify/cmd/main.go" "$OUT/_goverify/stage"
rm -rf "$OUT/_goverify"
mkdir -p "$OUT/verify"
unzip -q "$OUT/$BASENAME.zip" -d "$OUT/verify"
codesign --verify --deep --strict "$OUT/verify/Ollama.app"
rm -rf "$OUT/verify"

NOTES="Ollama ${VERSION} (${REPO}).

The app installs ${BASENAME}.zip after checking its ed25519 signature and the Apple code signature.
Ollama.dmg is the same disk image as ${BASENAME}.dmg, for the stable download URL."

status "pushing $TAG"
git push -u origin HEAD
git push origin "$TAG"
TAG_PUSHED=1

upload_one() {
  local f="$1"
  local attempt=1
  local delay=4
  while [ "$attempt" -le 5 ]; do
    if gh release upload "$TAG" --repo "$REPO" --clobber "$f"; then
      return 0
    fi
    echo "Upload failed (attempt ${attempt}) for $(basename "$f"). Retrying in ${delay}s." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
  echo "Giving up on $(basename "$f")" >&2
  return 1
}

status "publishing GitHub Release $TAG"
gh release create "$TAG" --repo "$REPO" --draft --title "$TAG" --notes "$NOTES"
FILES=(
  "$OUT/$BASENAME.zip"
  "$OUT/$BASENAME.zip.ed25519"
  "$OUT/$BASENAME.dmg"
  "$OUT/$BASENAME.dmg.ed25519"
  "$OUT/Ollama.dmg"
  "$OUT/Ollama.dmg.ed25519"
  "$OUT/sha256sum.txt"
  "$OUT/sha256sum.txt.ed25519"
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || die "missing asset $f"
  status "upload $(basename "$f")"
  upload_one "$f"
done

ASSET_NAMES="$(gh release view "$TAG" --repo "$REPO" --json assets --jq '.assets[].name')"
for f in "${FILES[@]}"; do
  base="$(basename "$f")"
  printf '%s\n' "$ASSET_NAMES" | grep -qx "$base" || die "Missing on the release: $base"
done
gh release edit "$TAG" --repo "$REPO" --draft=false --prerelease=false --latest --title "$TAG" --notes "$NOTES"
LATEST="$(gh api "repos/$REPO/releases/latest" --jq .tag_name)"
[ "$LATEST" = "$TAG" ] || die "latest release is $LATEST, not $TAG"

PUBLISHED=1
trap - EXIT
echo "Published $TAG on $REPO. Installed apps pick up ${BASENAME}.zip on their next update check."
