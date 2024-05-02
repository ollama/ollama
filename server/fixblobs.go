package server

import (
	"os"
	"path/filepath"
	"runtime"
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

// fixManifests walks the provided dir and replaces (":") to ("%") for all
// manifest files on non-Windows systems.
func fixManifests(dir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	return filepath.Walk(dir, func(oldPath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		var partNum int
		newPath := []byte(oldPath)
		for i := len(newPath) - 1; i >= 0; i-- {
			if partNum > 3 {
				break
			}
			if partNum == 3 {
				if newPath[i] == ':' {
					newPath[i] = '%'
					break
				}
				continue
			}
			if newPath[i] == '/' {
				partNum++
			}
		}

		newDir, _ := filepath.Split(string(newPath))
		if err := os.MkdirAll(newDir, 0o755); err != nil {
			return err
		}

		return os.Rename(oldPath, string(newPath))
	})
}
