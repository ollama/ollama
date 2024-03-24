package llm

import (
	"embed"
)

//go:embed llama.cpp/build/linux/*/*/lib/*
//go:embed neural_speed/build/linux/*/*/lib/*
var libEmbed embed.FS
