package gpu

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var (
	lock        sync.Mutex
	payloadsDir = ""
)

func PayloadsDir() (string, error) {
	lock.Lock()
	defer lock.Unlock()
	if payloadsDir == "" {
		tmpDir, err := os.MkdirTemp("", "ollama")
		if err != nil {
			return "", fmt.Errorf("failed to generate tmp dir: %w", err)
		}
		payloadsDir = tmpDir
	}
	return payloadsDir, nil
}

func Cleanup() {
	lock.Lock()
	defer lock.Unlock()
	if payloadsDir != "" {
		slog.Debug("cleaning up payloads dir " + payloadsDir)
		err := os.RemoveAll(payloadsDir)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to cleanup tmp dir: %s", err))
		}
	}
}

func UpdatePath(dir string) {
	if runtime.GOOS == "windows" {
		tmpDir := filepath.Dir(dir)
		pathComponents := strings.Split(os.Getenv("PATH"), ";")
		i := 0
		for _, comp := range pathComponents {
			if strings.EqualFold(comp, dir) {
				return
			}
			// Remove any other prior paths to our temp dir
			if !strings.HasPrefix(strings.ToLower(comp), strings.ToLower(tmpDir)) {
				pathComponents[i] = comp
				i++
			}
		}
		newPath := strings.Join(append([]string{dir}, pathComponents...), ";")
		slog.Info(fmt.Sprintf("Updating PATH to %s", newPath))
		os.Setenv("PATH", newPath)
	}
	// linux and darwin rely on rpath
}
