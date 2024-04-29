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
		runnersDir := os.Getenv("OLLAMA_RUNNERS_DIR")
		// On Windows we do not carry the payloads inside the main executable
		if runtime.GOOS == "windows" && runnersDir == "" {
			appExe, err := os.Executable()
			if err != nil {
				slog.Error("failed to lookup executable path", "error", err)
				return "", err
			}

			cwd, err := os.Getwd()
			if err != nil {
				slog.Error("failed to lookup working directory", "error", err)
				return "", err
			}

			var paths []string
			for _, root := range []string{appExe, cwd} {
				paths = append(paths,
					filepath.Join(root),
					filepath.Join(root, "windows-"+runtime.GOARCH),
					filepath.Join(root, "dist", "windows-"+runtime.GOARCH),
				)
			}

			// Try a few variations to improve developer experience when building from source in the local tree
			for _, p := range paths {
				candidate := filepath.Join(p, "ollama_runners")
				_, err := os.Stat(candidate)
				if err == nil {
					runnersDir = candidate
					break
				}
			}
			if runnersDir == "" {
				err = fmt.Errorf("unable to locate llm runner directory.  Set OLLAMA_RUNNERS_DIR to the location of 'ollama_runners'")
				slog.Error("incomplete distribution", "error", err)
				return "", err
			}
		}
		if runnersDir != "" {
			payloadsDir = runnersDir
			return payloadsDir, nil
		}

		// The remainder only applies on non-windows where we still carry payloads in the main executable
		cleanupTmpDirs()
		tmpDir := os.Getenv("OLLAMA_TMPDIR")
		if tmpDir == "" {
			tmpDir, err = os.MkdirTemp("", "ollama")
			if err != nil {
				return "", fmt.Errorf("failed to generate tmp dir: %w", err)
			}
		} else {
			err = os.MkdirAll(tmpDir, 0755)
			if err != nil {
				return "", fmt.Errorf("failed to generate tmp dir %s: %w", tmpDir, err)
			}
		}

		// Track our pid so we can clean up orphaned tmpdirs
		pidFilePath := filepath.Join(tmpDir, "ollama.pid")
		pidFile, err := os.OpenFile(pidFilePath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.ModePerm)
		if err != nil {
			return "", err
		}
		if _, err := pidFile.Write([]byte(fmt.Sprint(os.Getpid()))); err != nil {
			return "", err
		}

		// We create a distinct subdirectory for payloads within the tmpdir
		// This will typically look like /tmp/ollama3208993108/runners on linux
		payloadsDir = filepath.Join(tmpDir, "runners")
	}
	return payloadsDir, nil
}

// Best effort to clean up prior tmpdirs
func cleanupTmpDirs() {
	dirs, err := filepath.Glob(filepath.Join(os.TempDir(), "ollama*"))
	if err != nil {
		return
	}
	for _, d := range dirs {
		info, err := os.Stat(d)
		if err != nil || !info.IsDir() {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(d, "ollama.pid"))
		if err == nil {
			pid, err := strconv.Atoi(string(raw))
			if err == nil {
				if proc, err := os.FindProcess(int(pid)); err == nil && !errors.Is(proc.Signal(syscall.Signal(0)), os.ErrProcessDone) {
					// Another running ollama, ignore this tmpdir
					continue
				}
			}
		} else {
			slog.Debug("failed to open ollama.pid", "path", d, "error", err)
		}
		err = os.RemoveAll(d)
		if err != nil {
			slog.Debug("unable to cleanup stale tmpdir", "path", d, "error", err)
		}
	}
}

func Cleanup() {
	lock.Lock()
	defer lock.Unlock()
	runnersDir := os.Getenv("OLLAMA_RUNNERS_DIR")
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
