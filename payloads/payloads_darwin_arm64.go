package payloads

import "embed"

// Darwin payloads separated by architecture to avoid duplicate payloads when cross compiling

//go:embed build/darwin/arm64/*/*
var libEmbed embed.FS
