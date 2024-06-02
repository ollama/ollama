//go:build dragonfly || freebsd || netbsd || openbsd

package llm

import "embed"

//go:embed build/bsd/*/*/bin/*
var libEmbed embed.FS
