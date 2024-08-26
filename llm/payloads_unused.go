//go:build !linux && !darwin

package llm

import "embed"

// unused on windows
var libEmbed embed.FS
