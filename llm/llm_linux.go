//go:generate scripts/build_linux.sh
package llm

import "embed"

//go:embed build/linux/*/*/bin/*
var libEmbed embed.FS
