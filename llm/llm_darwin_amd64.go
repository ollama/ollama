package llm

import (
	"embed"
)

//go:embed build/darwin/x86_64/*/bin/*
var libEmbed embed.FS
