#!/bin/bash

mkdir -p dist

# build and sign the universal binary
CGO_ENABLED=1 GOARCH=arm64 go build -o dist/ollama_arm64
CGO_ENABLED=1 GOARCH=amd64 go build -o dist/ollama_amd64
lipo -create -output dist/ollama dist/ollama_arm64 dist/ollama_amd64
rm dist/ollama_amd64 dist/ollama_arm64
codesign --deep --force --options=runtime --sign "$APPLE_IDENTITY" --timestamp ./dist/ollama
xcrun altool --notarize-app --username="$APPLE_ID" --password "$APPLE_PASSWORD" --file ./dist/ollama

# build and sign the mac app
npm run --prefix app make:sign
cp app/out/make/zip/darwin/universal/Ollama-darwin-universal-${VERSION:-0.0.0}.zip dist/Ollama-darwin.zip

# rename the cli after its been packaged
mv dist/ollama dist/ollama-darwin
