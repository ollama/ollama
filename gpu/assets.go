package gpu

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/ollama/ollama/envconfig"
)

var (
	lock        sync.Mutex
	payloadsDir = ""
)

func PayloadsDir() (string, error) {
	lock.Lock()
	defer lock.Unlock()
	var err error
	if payloadsDir == "" {
		runnersDir := envconfig.RunnersDir()

		if runnersDir != "" {
			payloadsDir = runnersDir
			return payloadsDir, nil
		}

		// The remainder only applies on non-windows where we still carry payloads in the main executable
		cleanupTmpDirs()
		tmpDir := envconfig.TmpDir()
		if tmpDir == "" {
			tmpDir, err = os.MkdirTemp("", "ollama")
			if err != nil {
				return "", fmt.Errorf("failed to generate tmp dir: %w", err)
			}
		} else {
			err = os.MkdirAll(tmpDir, 0o755)
			if err != nil {
				return "", fmt.Errorf("failed to generate tmp dir %s: %w", tmpDir, err)
			}
		}

		// Track our pid so we can clean up orphaned tmpdirs
		n := filepath.Join(tmpDir, "ollama.pid")
		if err := os.WriteFile(n, []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
			return "", fmt.Errorf("failed to write pid file %s: %w", n, err)
		}

		// We create a distinct subdirectory for payloads within the tmpdir
		// This will typically look like /tmp/ollama3208993108/runners on linux
		payloadsDir = filepath.Join(tmpDir, "runners")
	}
	return payloadsDir, nil
}

// Best effort to clean up prior tmpdirs
func cleanupTmpDirs() {
	matches, err := filepath.Glob(filepath.Join(os.TempDir(), "ollama*", "ollama.pid"))
	if err != nil {
		return
	}

	for _, match := range matches {
		raw, err := os.ReadFile(match)
		if errors.Is(err, os.ErrNotExist) {
			slog.Debug("not a ollama runtime directory, skipping", "path", match)
			continue
		} else if err != nil {
			slog.Warn("could not read ollama.pid, skipping", "path", match, "error", err)
			continue
		}

		pid, err := strconv.Atoi(string(raw))
		if err != nil {
			slog.Warn("invalid pid, skipping", "path", match, "error", err)
			continue
		}

		p, err := os.FindProcess(pid)
		if err == nil && !errors.Is(p.Signal(syscall.Signal(0)), os.ErrProcessDone) {
			slog.Warn("process still running, skipping", "pid", pid, "path", match)
			continue
		}

		if err := os.Remove(match); err != nil {
			slog.Warn("could not cleanup stale pidfile", "path", match, "error", err)
		}

		runners := filepath.Join(filepath.Dir(match), "runners")
		if err := os.RemoveAll(runners); err != nil {
			slog.Warn("could not cleanup stale runners", "path", runners, "error", err)
		}

		if err := os.Remove(filepath.Dir(match)); err != nil {
			slog.Warn("could not cleanup stale tmpdir", "path", filepath.Dir(match), "error", err)
		}
	}
}

func Cleanup() {
	lock.Lock()
	defer lock.Unlock()
	runnersDir := envconfig.RunnersDir()
	if payloadsDir != "" && runnersDir == "" && runtime.GOOS != "windows" {
		// We want to fully clean up the tmpdir parent of the payloads dir
		tmpDir := filepath.Clean(filepath.Join(payloadsDir, ".."))
		slog.Debug("cleaning up", "dir", tmpDir)
		err := os.RemoveAll(tmpDir)
		if err != nil {
			// On windows, if we remove too quickly the llama.dll may still be in-use and fail to remove
			time.Sleep(1000 * time.Millisecond)
			err = os.RemoveAll(tmpDir)
			if err != nil {
				slog.Warn("failed to clean up", "dir", tmpDir, "err", err)
			}
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
		slog.Info("updating", "PATH", newPath)
		os.Setenv("PATH", newPath)
	}
	// linux and darwin rely on rpath
}
