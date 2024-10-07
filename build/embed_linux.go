package build

import "embed"

//go:embed linux/*
var EmbedFS embed.FS
