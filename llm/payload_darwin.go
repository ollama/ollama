package llm

import (
	"embed"
)

//go:embed llama.cpp/ggml-metal.metal llama.cpp/build/darwin/*/lib/*.so
var libEmbed embed.FS
