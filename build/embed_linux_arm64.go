package build

import "embed"

//go:embed linux/arm64/*
var EmbedFS embed.FS
