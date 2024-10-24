package build

import "embed"

//go:embed linux/amd64/*
var EmbedFS embed.FS
