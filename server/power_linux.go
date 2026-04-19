//go:build linux

package server

import (
	"errors"
	"log/slog"
	"os"
	"os/exec"
	"sync"
)

var (
	inhibitMu   sync.Mutex
	inhibitProc *os.Process
)

// platformPreventSleep uses systemd-inhibit to prevent system sleep on Linux.
// This works on systems with systemd (most modern Linux distributions).
func platformPreventSleep() error {
	inhibitMu.Lock()
	defer inhibitMu.Unlock()

	if inhibitProc != nil {
		return nil // Already inhibiting
	}

	// Check if systemd-inhibit is available
	path, err := exec.LookPath("systemd-inhibit")
	if err != nil {
		slog.Debug("systemd-inhibit not available, sleep prevention not supported on this system")
		return nil // Not an error, just not supported
	}

	// Start systemd-inhibit with a blocking command (cat without input will block forever)
	// The inhibit lock is held as long as this process runs
	cmd := exec.Command(path,
		"--what=idle:sleep",
		"--who=ollama",
		"--why=Ollama is processing requests",
		"--mode=block",
		"cat",
	)

	if err := cmd.Start(); err != nil {
		return errors.New("failed to start systemd-inhibit: " + err.Error())
	}

	inhibitProc = cmd.Process
	return nil
}

// platformAllowSleep stops the systemd-inhibit process, allowing system sleep.
func platformAllowSleep() error {
	inhibitMu.Lock()
	defer inhibitMu.Unlock()

	if inhibitProc == nil {
		return nil // Not currently inhibiting
	}

	if err := inhibitProc.Kill(); err != nil {
		return errors.New("failed to stop systemd-inhibit: " + err.Error())
	}

	// Wait for the process to exit to avoid zombies
	_, _ = inhibitProc.Wait()
	inhibitProc = nil
	return nil
}
