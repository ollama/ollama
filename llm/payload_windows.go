package llm

import (
	"embed"
)

//go:embed llama.cpp/build/windows/*/lib/*.dll
var libEmbed embed.FS
