package llm

import "embed"

// Darwin payloads separated by architecture to avoid duplicate payloads when cross compiling

//go:embed build/darwin/arm64/*/bin/*
var libEmbed embed.FS
