#!/bin/sh

# Build script for the Intel (x86_64) macOS 12+ fork of Ollama.
#
# Produces a single-arch amd64 .app bundle and a DMG named
# Ollama-intel-macos12-<version>.dmg. This lives on the intel-macos12
# branch alongside the JSC lookbehind polyfill and the 326KB code-split
# UI bundle. It is independent of scripts/build_darwin.sh, which targets
# the Apple Silicon / macOS 14+ build on main.
#
# Usage:
#   ./scripts/build_darwin_intel.sh [build app]
#
# Outputs:
#   dist/Ollama.app/                 (amd64-only)
#   dist/Ollama-intel-macos12-<ver>.dmg
#
# Environment overrides:
#   VERSION    - override the version string (default: git describe)
#   DMG_SKIP_FINDER - set to 1 to package without Finder AppleScript

set -e

export GOCACHE="${GOCACHE:-$(pwd)/build/go-cache}"

VOL_NAME=${VOL_NAME:-"Ollama Intel"}
export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export CGO_CFLAGS="-O3 -mmacosx-version-min=12.0"
export CGO_CXXFLAGS="-O3 -mmacosx-version-min=12.0"
export CGO_LDFLAGS="-mmacosx-version-min=12.0"

status() { echo >&2 ">>> $@"; }
usage() {
    echo "usage: $(basename $0) [build app]"
    exit 1
}

mkdir -p dist

_build_darwin_intel() {
    status "Building darwin amd64 (Intel, macOS 12+)"
    # MLX is deliberately disabled on this fork target: it is an Apple-Silicon
    # optimized backend, this build is Intel-only with no usable Metal GPU,
    # and MLX's current LAPACK usage does not compile against this machine's
    # Xcode 14 SDK. GGML/llama.cpp (used by every locally pulled model) is
    # unaffected and remains the only backend built here.
    SOURCE_BUILD=build/darwin-sources
    cmake -S . -B "$SOURCE_BUILD" -DOLLAMA_MLX_BACKENDS= -DOLLAMA_LLAMA_BACKENDS=
    cmake --build "$SOURCE_BUILD" --target ollama-llama-cpp-source
    LLAMA_CPP_SHARED_SRC="$(pwd)/$SOURCE_BUILD/_deps/llama_cpp-src"

    INSTALL_PREFIX=dist/darwin-amd64/
    BUILD_DIR=build/darwin-amd64

    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=x86_64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
        -DOLLAMA_PAYLOAD_INSTALL_PREFIX=$INSTALL_PREFIX \
        -DOLLAMA_GO_OUTPUT=$INSTALL_PREFIX/ollama \
        -DOLLAMA_VERSION="$VERSION" \
        -DOLLAMA_MLX_BACKENDS= \
        -DOLLAMA_LLAMA_BACKENDS= \
        -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=$LLAMA_CPP_SHARED_SRC

    GOOS=darwin GOARCH=amd64 CGO_ENABLED=1 \
        CGO_CFLAGS="-O3 -mmacosx-version-min=12.0" \
        CGO_LDFLAGS="-ldl -lc++ -framework Accelerate -mmacosx-version-min=12.0" \
        cmake --build "$BUILD_DIR" --target ollama-local --parallel
}

_build_macapp_intel() {
    if ! command -v npm &> /dev/null; then
        echo "npm is not installed. Please install Node.js and npm first."
        exit 1
    fi

    cd app/ui/app
    npm ci
    npm run build
    cd ../../..

    rm -rf dist/Ollama.app
    cp -a ./app/darwin/Ollama.app dist/Ollama.app
    touch dist/Ollama.app

    GOARCH=amd64 CGO_ENABLED=1 GOOS=darwin CGO_CFLAGS="-O3 -mmacosx-version-min=12.0" CGO_LDFLAGS="-mmacosx-version-min=12.0" \
        go build -o dist/darwin-app-amd64 \
        -ldflags="-s -w -X=github.com/ollama/ollama/app/version.Version=${VERSION}" \
        ./app/cmd/app

    mkdir -p dist/Ollama.app/Contents/MacOS
    cp dist/darwin-app-amd64 dist/Ollama.app/Contents/MacOS/Ollama
    rm -f dist/darwin-app-amd64

    # Mock Squirrel.framework
    mkdir -p dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/
    cp -a dist/Ollama.app/Contents/MacOS/Ollama dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel
    ln -s ../Squirrel dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/ShipIt
    cp -a ./app/cmd/squirrel/Info.plist dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Resources/Info.plist
    ln -s A dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/Current
    ln -s Versions/Current/Resources dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Resources
    ln -s Versions/Current/Squirrel dist/Ollama.app/Contents/Frameworks/Squirrel.framework/Squirrel

    plutil -replace CFBundleShortVersionString -string "$VERSION" dist/Ollama.app/Contents/Info.plist
    plutil -replace CFBundleVersion -string "$VERSION" dist/Ollama.app/Contents/Info.plist

    # Resources
    mkdir -p dist/Ollama.app/Contents/Resources
    cp -a dist/darwin-amd64/ollama dist/Ollama.app/Contents/Resources/ollama
    cp dist/darwin-amd64/lib/ollama/llama-server dist/Ollama.app/Contents/Resources/ 2>/dev/null || true
    cp dist/darwin-amd64/lib/ollama/llama-quantize dist/Ollama.app/Contents/Resources/ 2>/dev/null || true
    if [ -d dist/darwin-amd64/lib/ollama ]; then
        cp -a dist/darwin-amd64/lib/ollama/. dist/Ollama.app/Contents/Resources/
    fi
    chmod a+x dist/Ollama.app/Contents/Resources/ollama

    rm -f dist/Ollama-intel-macos12-${VERSION}.dmg
    (
        cd dist
        set --
        if [ "${DMG_SKIP_FINDER:-0}" = 1 ]; then
            set -- --skip-jenkins
        fi
        ../scripts/create-dmg.sh "$@" \
        --volname "${VOL_NAME}" \
        --volicon ../app/darwin/Ollama.app/Contents/Resources/icon.icns \
        --background ../app/assets/background.png \
        --window-pos 200 120 \
        --window-size 800 400 \
        --icon-size 128 \
        --icon "Ollama.app" 200 190 \
        --hide-extension "Ollama.app" \
        --app-drop-link 600 190 \
        "Ollama-intel-macos12-${VERSION}.dmg" \
        "Ollama.app" \
    ; )
    rm -f dist/rw*.dmg

    echo "Done: dist/Ollama-intel-macos12-${VERSION}.dmg"
}

if [ "$#" -eq 0 ]; then
    _build_darwin_intel
    _build_macapp_intel
    exit 0
fi

for CMD in "$@"; do
    case $CMD in
        build) _build_darwin_intel ;;
        app) _build_macapp_intel ;;
        *) usage ;;
    esac
done
