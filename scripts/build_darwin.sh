#!/bin/sh

set -e

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

mkdir -p dist

for TARGETARCH in arm64 amd64; do
    rm -rf llm/llama.cpp/build
    GOOS=darwin GOARCH=$TARGETARCH go generate ./...
    CGO_ENABLED=1 GOOS=darwin GOARCH=$TARGETARCH go build -trimpath -o dist/ollama-darwin-$TARGETARCH
    CGO_ENABLED=1 GOOS=darwin GOARCH=$TARGETARCH go build -C app -trimpath -o ../dist/ollama-app-darwin-$TARGETARCH
done

lipo -create -output dist/ollama dist/ollama-darwin-arm64 dist/ollama-darwin-amd64
lipo -create -output dist/ollama-app dist/ollama-app-darwin-arm64 dist/ollama-app-darwin-amd64
rm -f dist/ollama-darwin-* dist/ollama-app-darwin-*

# create the mac app
rm -rf dist/Ollama.app
cp -R app/darwin/Ollama.app dist/
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" dist/Ollama.app/Contents/Info.plist
mkdir -p dist/Ollama.app/Contents/MacOS
mv dist/ollama-app dist/Ollama.app/Contents/MacOS/Ollama
cp dist/ollama dist/Ollama.app/Contents/Resources/ollama

# sign and notarize the app
if [ -n "$APPLE_IDENTITY" ]; then
    codesign -f --timestamp --options=runtime --sign "$APPLE_IDENTITY" --identifier ai.ollama.ollama dist/Ollama.app/Contents/MacOS/Ollama
    codesign -f --timestamp --options=runtime --sign "$APPLE_IDENTITY" --identifier ai.ollama.ollama dist/Ollama.app/Contents/Resources/ollama
    codesign -f --timestamp --options=runtime --sign "$APPLE_IDENTITY" --identifier ai.ollama.ollama dist/Ollama.app
    ditto -c -k --keepParent dist/Ollama.app dist/Ollama-darwin.zip
    rm -rf dist/Ollama.app
    xcrun notarytool submit dist/Ollama-darwin.zip --wait --timeout 10m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
    unzip dist/Ollama-darwin.zip -d dist
    rm -f dist/Ollama-darwin.zip
    xcrun stapler staple "dist/Ollama.app"
    ditto -c -k --keepParent dist/Ollama.app dist/Ollama-darwin.zip
    rm -rf dist/Ollama.app
else
    echo "Skipping code signing - set APPLE_IDENTITY"
fi

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
