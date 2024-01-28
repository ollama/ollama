package llm

import (
	"embed"
)

//go:embed llama.cpp/build/linux/*/*/lib/*.so*
var libEmbed embed.FS
