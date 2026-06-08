//go:build linux

package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

// PreventSystemSleep prevents the system from sleeping during inference on Linux
// Uses systemd-inhibit to acquire an inhibitor lock
// Returns the process ID of the inhibition process
func PreventSystemSleep() uint32 {
	// Try using systemd-inhibit first (preferred method)
	cmd := exec.Command(
		"systemd-inhibit",
		"--what=sleep",
		"--who=ollama",
		"--why=model inference in progress",
		"--mode=block",
		fmt.Sprintf("/proc/%d/fd/1", os.Getpid()),
	)
	
	err := cmd.Start()
	if err == nil {
		return uint32(cmd.Process.Pid)
	}
	
	// Fallback: try using dbus-send via org.freedesktop.login1
	// This is a simpler approach using the systemd login manager
	cmd = exec.Command(
		"dbus-send",
		"--system",
		"--print-reply",
		"/org/freedesktop/login1",
		"org.freedesktop.login1.Manager.Inhibit",
		"s:sleep",
		"s:ollama",
		"s:model inference in progress",
		"s:block",
	)
	
	err = cmd.Start()
	if err == nil {
		return uint32(cmd.Process.Pid)
	}
	
	// If both fail, return 0 (no active process)
	return 0
}

// AllowSystemSleep removes the sleep inhibitor on Linux
// Pass the PID returned from PreventSystemSleep
func AllowSystemSleep(pid uint32) {
	if pid == 0 {
		return
	}
	
	// Kill the inhibition process
	proc, err := os.FindProcess(int(pid))
	if err != nil {
		return
	}
	proc.Signal(syscall.SIGTERM)
}

// IsSystemSleepPreventionAvailable returns true if sleep prevention tools are available
func IsSystemSleepPreventionAvailable() bool {
	// Check for systemd-inhibit first
	_, err := exec.LookPath("systemd-inhibit")
	if err == nil {
		return true
	}
	
	// Check for dbus-send as fallback
	_, err = exec.LookPath("dbus-send")
	return err == nil
}
