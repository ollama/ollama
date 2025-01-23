package server

import (
	"os"
	"path/filepath"
	"strings"
)

// fixBlobs walks the provided dir and replaces (":") to ("-") in the file
// prefix. (e.g. sha256:1234 -> sha256-1234)
func fixBlobs(dir string) error {
	return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		baseName := filepath.Base(path)
		typ, sha, ok := strings.Cut(baseName, ":")
		if ok && typ == "sha256" {
			newPath := filepath.Join(filepath.Dir(path), typ+"-"+sha)
			if err := os.Rename(path, newPath); err != nil {
				return err
			}
		}
		return nil
	})
}
