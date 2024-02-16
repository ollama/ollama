package llm

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/jmorganca/ollama/version"
)

func parseDurationMs(ms float64) time.Duration {
	dur, err := time.ParseDuration(fmt.Sprintf("%fms", ms))
	if err != nil {
		panic(err)
	}

	return dur
}

func libDir() (string, error) {
	baseDir := ""
	if ollamaHome, exists := os.LookupEnv("OLLAMA_HOME"); exists {
		baseDir = filepath.Join(ollamaHome, "libs")
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		baseDir = filepath.Join(home, ".ollama", "libs")
	}
	libDirs, err := os.ReadDir(baseDir)
	if err == nil {
		for _, d := range libDirs {
			if d.Name() == version.Version {
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
