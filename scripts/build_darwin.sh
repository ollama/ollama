#!/bin/sh

set -e

status() { echo >&2 ">>> $@"; }
usage() {
    echo "usage: $(basename $0) [build [sign]]"
    exit 1
}

export VERSION=${VERSION:-$(git describe --tags --dirty)}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=${VERSION#v}\" \"-X=github.com/ollama/ollama/server.mode=release\"'"
export CGO_CPPFLAGS='-mmacosx-version-min=11.3'

ARCHS="arm64 amd64"
while getopts "a:h" OPTION; do
    case $OPTION in
        a) ARCHS=$OPTARG ;;
        h) usage ;;
    esac
done

shift $(( $OPTIND - 1 ))

_build_darwin() {
    for ARCH in $ARCHS; do
        status "Building darwin $ARCH"
        GOOS=darwin GOARCH=$ARCH CGO_ENABLED=1 go build -o dist/darwin/$ARCH/ .

        if [ "$ARCH" = "amd64" ]; then
            status "Building darwin $ARCH dynamic backends"
            cmake -B build/darwin/$ARCH -DCMAKE_OSX_DEPLOYMENT_TARGET=11.3 -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64
            cmake --build build/darwin/$ARCH --target ggml-cpu -j
            cp -a build/darwin/$ARCH/lib/ollama/* dist/darwin/$ARCH/
        fi
    done
}

_sign_darwin() {
    status "Creating universal binary..."
    lipo -create -output dist/darwin/ollama dist/darwin/*/ollama

    if [ -z "$APPLE_IDENTITY" ]; then
        status "No APPLE_IDENTITY set, skipping code signing"
        return
    fi

    for F in dist/darwin/ollama dist/darwin/amd64/lib*; do
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime $F
    done

    # create a temporary zip for notarization
    TEMP=$(mktemp -u).zip
    ditto -c -k --keepParent dist/darwin/ollama "$TEMP"
    xcrun notarytool submit dist/darwin/temp.zip --wait --timeout 10m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
    rm -f "$TEMP"

    # create a universal tarball
    tar -cf dist/ollama-darwin.tar --strip-components 2 dist/darwin/ollama
    tar -rf dist/ollama-darwin.tar --strip-components 3 dist/darwin/amd64/lib*
    gzip -9vc <dist/ollama-darwin.tar >dist/ollama-darwin.tgz
}

_build_macapp() {
    # build and optionally sign the mac app
    npm install --prefix macapp
    if [ -n "$APPLE_IDENTITY" ]; then
        npm run --prefix macapp make:sign
    else
        npm run --prefix macapp make
    fi

    mv ./macapp/out/make/zip/darwin/universal/Ollama-darwin-universal-$VERSION.zip dist/Ollama-darwin.zip
}

if [ "$#" -eq 0 ]; then
    _build_darwin
    _sign_darwin
    _build_macapp
    exit 0
fi

for CMD in "$@"; do
    case $CMD in
        build) _build_darwin ;;
        sign) _sign_darwin ;;
        macapp) _build_macapp ;;
        *) usage ;;
    esac
done
