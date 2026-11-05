//go:build !windows

package manifest

import "path/filepath"

func resolveLinks(path string) (string, error) {
	return filepath.EvalSymlinks(path)
}
