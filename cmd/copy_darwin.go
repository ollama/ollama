package cmd

import (
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

func localCopy(src, target string) error {
	dirPath := filepath.Dir(target)

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return err
	}

	err := unix.Clonefile(src, target, 0)
	if err != nil {
		return err
	}

	return nil
}
