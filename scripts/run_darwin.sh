#!/bin/bash

set -e

cp -R app/darwin/Ollama.app $TMPDIR/Ollama.app
mkdir -p $TMPDIR/Ollama.app/Contents/Resources $TMPDIR/Ollama.app/Contents/MacOS
go build -o $TMPDIR/Ollama.app/Contents/Resources/ollama .
go build -C app -o $TMPDIR/Ollama.app/Contents/MacOS/Ollama .
$TMPDIR/Ollama.app/Contents/MacOS/Ollama
