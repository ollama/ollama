//go:build !linux && !darwin

package build

import "embed"

// unused on windows
var EmbedFS embed.FS
