#!/bin/sh

set -e

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

mkdir -p dist

for TARGETARCH in arm64 amd64; do
    rm -rf llm/llama.cpp/build
    GOOS=darwin GOARCH=$TARGETARCH go generate ./...
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
