#!/bin/sh

set -e

. $(dirname $0)/env.sh

mkdir -p dist

# These require Xcode v13 or older to target MacOS v11
# If installed to an alternate location use the following to enable
# export SDKROOT=/Applications/Xcode_12.5.1.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
# export DEVELOPER_DIR=/Applications/Xcode_12.5.1.app/Contents/Developer
export CGO_CPPFLAGS=-mmacosx-version-min=11.3

rm -rf llama/build dist/darwin-*
echo "Building darwin arm64"
GOOS=darwin ARCH=arm64 GOARCH=arm64 make -j 8 dist
echo "Building darwin amd64 with AVX enabled"
GOOS=darwin ARCH=amd64 GOARCH=amd64 CUSTOM_CPU_FLAGS="avx" make -j 8 dist


lipo -create -output dist/ollama dist/darwin-arm64/bin/ollama dist/darwin-amd64/bin/ollama
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
