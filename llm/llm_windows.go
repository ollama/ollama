package llm

import "embed"

//go:embed build/windows/*/*/bin/*
var libEmbed embed.FS
