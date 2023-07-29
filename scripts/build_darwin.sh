#!/bin/bash

mkdir -p dist

# build universal binary
CGO_ENABLED=1 GOARCH=arm64 go build -o dist/ollama_arm64
CGO_ENABLED=1 GOARCH=amd64 go build -o dist/ollama_amd64
lipo -create -output dist/ollama dist/ollama_arm64 dist/ollama_amd64
rm dist/ollama_amd64 dist/ollama_arm64
codesign --deep --force --options=runtime --sign "$APPLE_IDENTITY" --timestamp dist/ollama

# build and sign the mac app
npm run --prefix app make:sign
cp app/out/make/zip/darwin/universal/Ollama-darwin-universal-${VERSION:-0.0.0}.zip dist/Ollama-darwin.zip

# sign the binary and rename it
codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime dist/ollama
ditto -c -k --keepParent dist/ollama dist/temp.zip
xcrun notarytool submit dist/temp.zip --wait --timeout 10m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
mv dist/ollama dist/ollama-darwin
rm dist/temp.zip
