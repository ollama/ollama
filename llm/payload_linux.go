package llm

import (
	"embed"
)

//go:embed llama.cpp/build/linux/*/*/lib/*
var libEmbed embed.FS
