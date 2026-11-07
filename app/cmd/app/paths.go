package main

import (
	"path/filepath"
	"strings"
)

// pathWithinDir reports whether path resolves to a location inside dir
// (or equal to it). It uses filepath.Rel rather than a string prefix so
// sibling directories such as /ApplicationsFoo are not mistaken for
// children of /Applications.
func pathWithinDir(path, dir string) bool {
	rel, err := filepath.Rel(dir, path)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
