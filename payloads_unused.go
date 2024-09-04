//go:build !linux && !darwin

package main

import "embed"

// unused on windows
var libEmbed embed.FS
