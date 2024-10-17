#!/bin/sh

set -e

. $(dirname $0)/env.sh

mkdir -p dist

for TARGETARCH in arm64 amd64; do
    if [ -n "${OLLAMA_NEW_RUNNERS}" ]; then
        echo "Building Go runner darwin $TARGETARCH"
        rm -rf llama/build
        GOOS=darwin ARCH=$TARGETARCH GOARCH=$TARGETARCH make -C llama -j 8
    else
        echo "Building C++ runner darwin $TARGETARCH"
        rm -rf llm/build
        GOOS=darwin GOARCH=$TARGETARCH go generate ./...
    fi
    # These require Xcode v13 or older to target MacOS v11
    # If installed to an alternate location use the following to enable
    # export SDKROOT=/Applications/Xcode_12.5.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    # export DEVELOPER_DIR=/Applications/Xcode_12.5.1.app/Contents/Developer
    export CGO_CFLAGS=-mmacosx-version-min=11.3
    export CGO_CXXFLAGS=-mmacosx-version-min=11.3
    export CGO_LDFLAGS=-mmacosx-version-min=11.3
    CGO_ENABLED=1 GOOS=darwin GOARCH=$TARGETARCH go build -trimpath -o dist/ollama-darwin-$TARGETARCH
    CGO_ENABLED=1 GOOS=darwin GOARCH=$TARGETARCH go build -trimpath -cover -o dist/ollama-darwin-$TARGETARCH-cov
done

lipo -create -output dist/ollama dist/ollama-darwin-arm64 dist/ollama-darwin-amd64
rm -f dist/ollama-darwin-arm64 dist/ollama-darwin-amd64
if [ -n "$APPLE_IDENTITY" ]; then
    codesign --deep --force --options=runtime --sign "$APPLE_IDENTITY" --timestamp dist/ollama
else
    echo "Skipping code signing - set APPLE_IDENTITY"
fi
chmod +x dist/ollama

# build and optionally sign the mac app
npm install --prefix macapp
if [ -n "$APPLE_IDENTITY" ]; then
    npm run --prefix macapp make:sign
else 
    npm run --prefix macapp make
fi
cp macapp/out/make/zip/darwin/universal/Ollama-darwin-universal-$VERSION.zip dist/Ollama-darwin.zip

# sign the binary and rename it
if [ -n "$APPLE_IDENTITY" ]; then
    codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/ollama
else
    echo "WARNING: Skipping code signing - set APPLE_IDENTITY"
fi
ditto -c -k --keepParent dist/ollama dist/temp.zip
if [ -n "$APPLE_IDENTITY" ]; then
    xcrun notarytool submit dist/temp.zip --wait --timeout 10m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
fi
mv dist/ollama dist/ollama-darwin
rm -f dist/temp.zip
