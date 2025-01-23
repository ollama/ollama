#!/bin/sh

set -e

. $(dirname $0)/env.sh

mkdir -p dist

# These require Xcode v13 or older to target MacOS v11
# If installed to an alternate location use the following to enable
# export SDKROOT=/Applications/Xcode_12.5.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
# export DEVELOPER_DIR=/Applications/Xcode_12.5.1.app/Contents/Developer
export CGO_LDFLAGS=""
export CGO_CPPFLAGS=-mmacosx-version-min=11.3

# Generate the universal ollama binary for stand-alone usage: metal + avx
echo "Building binary"
echo "Building darwin arm64"
GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 go build -o dist/ollama-darwin-arm64
echo "Building darwin amd64 with AVX enabled"
GOOS=darwin GOARCH=amd64 CGO_ENABLED=1 GO_CPPFLAGS="-DGGML_AVX" go build -o dist/ollama-darwin-amd64
lipo -create -output dist/ollama-darwin dist/ollama-darwin-arm64 dist/ollama-darwin-amd64

# sign the binary and rename it
if [ -n "$APPLE_IDENTITY" ]; then
    codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/ollama-darwin
else
    echo "WARNING: Skipping code signing - set APPLE_IDENTITY"
fi
ditto -c -k --keepParent dist/ollama-darwin dist/temp.zip
if [ -n "$APPLE_IDENTITY" ]; then
    xcrun notarytool submit dist/temp.zip --wait --timeout 10m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
fi
rm -f dist/temp.zip
cp dist/ollama-darwin dist/ollama

cmake -B build -DCMAKE_OSX_DEPLOYMENT_TARGET=11.3 -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_OSX_ARCHITECTURES=x86_64
cmake --build build -j
if [ -n "$APPLE_IDENTITY" ]; then
    for lib in build/lib/*; do
        echo "Signing $lib"
        codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime "$lib"
    done
else
    echo "WARNING: Skipping library code signing - set APPLE_IDENTITY" 
fi

# build and optionally sign the mac app
npm install --prefix macapp
if [ -n "$APPLE_IDENTITY" ]; then
    npm run --prefix macapp make:sign
else 
    npm run --prefix macapp make
fi
cp ./macapp/out/make/zip/darwin/universal/Ollama-darwin-universal-$VERSION.zip dist/Ollama-darwin.zip
