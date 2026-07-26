//go:build darwin

package cmd

import (
	"fmt"
	"os"
	"os/exec"
)

// PreventSystemSleep prevents the system from sleeping during inference on macOS
// Returns the process ID of the caffeinate command, or -1 if it fails
// Must be stopped with AllowSystemSleep(pid)
func PreventSystemSleep() uint32 {
	// Use macOS caffeinate command to prevent sleep
	cmd := exec.Command("caffeinate", "-i", "-w", fmt.Sprintf("%d", os.Getpid()))
	
	// Start the process but don't wait for it
	// caffeinate will run and prevent sleep while the parent process is alive
	err := cmd.Start()
	if err != nil {
		return 0
	}
	
	// Return the process ID
	return uint32(cmd.Process.Pid)
}

// AllowSystemSleep stops the caffeinate process to allow system sleep
// Pass the PID returned from PreventSystemSleep
func AllowSystemSleep(pid uint32) {
	if pid == 0 {
		return
	}
	
	// Kill the caffeinate process
	proc, err := os.FindProcess(int(pid))
	if err != nil {
		return
	}
	proc.Kill()
}

// IsSystemSleepPreventionAvailable returns true if caffeinate is available on macOS
func IsSystemSleepPreventionAvailable() bool {
	_, err := exec.LookPath("caffeinate")
	return err == nil
}
