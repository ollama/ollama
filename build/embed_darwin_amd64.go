package build

import "embed"

// Darwin payloads separated by architecture to avoid duplicate payloads when cross compiling

//go:embed darwin/amd64/*
var EmbedFS embed.FS
