package llm

import (
	"embed"
)

//go:embed llama.cpp/ggml-metal.metal llama.cpp/build/darwin/arm64/*/lib/*.dylib*
var libEmbed embed.FS
