package gpu

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/jmorganca/ollama/version"
)

func AssetsDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	baseDir := filepath.Join(home, ".ollama", "assets")
	libDirs, err := os.ReadDir(baseDir)
	if err == nil {
		for _, d := range libDirs {
			if d.Name() == version.Version {
				continue
			}
			// Special case the rocm dependencies, which are handled by the installer
			if d.Name() == "rocm" {
				continue
			}
			slog.Debug("stale lib detected, cleaning up " + d.Name())
			err = os.RemoveAll(filepath.Join(baseDir, d.Name()))
			if err != nil {
				slog.Warn(fmt.Sprintf("unable to clean up stale library %s: %s", filepath.Join(baseDir, d.Name()), err))
			}
		}
	}
	return filepath.Join(baseDir, version.Version), nil
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
