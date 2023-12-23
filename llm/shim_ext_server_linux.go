package llm

import (
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"strings"
)

//go:embed llama.cpp/gguf/build/*/*/lib/*.so
var libEmbed embed.FS

func updatePath(dir string) {
	pathComponents := strings.Split(os.Getenv("PATH"), ":")
	for _, comp := range pathComponents {
		if comp == dir {
			return
		}
	}
	newPath := strings.Join(append(pathComponents, dir), ":")
	log.Printf("Updating PATH to %s", newPath)
	os.Setenv("PATH", newPath)
}

func verifyDriverAccess() error {
	// Only check ROCm access if we have the dynamic lib loaded
	if _, rocmPresent := AvailableShims["rocm"]; rocmPresent {
		// Verify we have permissions - either running as root, or we have group access to the driver
		fd, err := os.OpenFile("/dev/kfd", os.O_RDWR, 0666)
		if err != nil {
			if errors.Is(err, fs.ErrPermission) {
				return fmt.Errorf("Radeon card detected, but permissions not set up properly.  Either run ollama as root, or add you user account to the render group.")
			} else if errors.Is(err, fs.ErrNotExist) {
				// expected behavior without a radeon card
				return nil
			}

			return fmt.Errorf("failed to check permission on /dev/kfd: %w", err)
		}
		fd.Close()
	}
	return nil
}
