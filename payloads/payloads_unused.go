//go:build !linux && !darwin

package payloads

import "embed"

// unused on windows
var libEmbed embed.FS
