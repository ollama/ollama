//go:build darwin

package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

var (
	pidFile       = filepath.Join(os.Getenv("HOME"), "Library", "Application Support", "Ollama", "ollama.pid")
	serverLogPath = filepath.Join(os.Getenv("HOME"), ".ollama", "logs", "server.log")
)

func commandContext(ctx context.Context, name string, arg ...string) *exec.Cmd {
	return exec.CommandContext(ctx, name, arg...)
}

func terminate(proc *os.Process) error {
	return proc.Signal(os.Interrupt)
}

func terminated(pid int) (bool, error) {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false, fmt.Errorf("failed to find process: %v", err)
	}

	err = proc.Signal(syscall.Signal(0))
	if err != nil {
		if errors.Is(err, os.ErrProcessDone) || errors.Is(err, syscall.ESRCH) {
			return true, nil
		}

		return false, fmt.Errorf("error signaling process: %v", err)
	}

	return false, nil
}

// reapServers kills all ollama processes except our own
func reapServers() error {
	// Get our own PID to avoid killing ourselves
	currentPID := os.Getpid()

	// Use pkill to kill ollama processes
	// -x matches the whole command name exactly
	// We'll get the list first, then kill selectively
	cmd := exec.Command("pgrep", "-x", "ollama")
	output, err := cmd.Output()
	if err != nil {
		// No ollama processes found
		slog.Debug("no ollama processes found")
		return nil //nolint:nilerr
	}

	pidsStr := strings.TrimSpace(string(output))
	if pidsStr == "" {
		return nil
	}

	pids := strings.Split(pidsStr, "\n")
	for _, pidStr := range pids {
		pidStr = strings.TrimSpace(pidStr)
		if pidStr == "" {
			continue
		}

		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			slog.Debug("failed to parse PID", "pidStr", pidStr, "err", err)
			continue
		}
		if pid == currentPID {
			continue
		}

		proc, err := os.FindProcess(pid)
		if err != nil {
			slog.Debug("failed to find process", "pid", pid, "err", err)
			continue
		}

		if err := proc.Signal(syscall.SIGTERM); err != nil {
			// Try SIGKILL if SIGTERM fails
			if err := proc.Signal(syscall.SIGKILL); err != nil {
				slog.Warn("failed to stop external ollama process", "pid", pid, "err", err)
				continue
			}
		}

		slog.Info("stopped external ollama process", "pid", pid)
	}

	return nil
}
