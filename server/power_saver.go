// Package server provides the Ollama API server
package server

import (
	"os/exec"
	"sync"
	"runtime"
)

// Power management to prevent sleep during inference

var (
	preventSleepMu sync.Mutex
	preventSleepCnt int
	isActive       bool
)

// AcquirePowerLock prevents the system from sleeping while processing
// Inference requests. Uses platform-specific commands.
func AcquirePowerLock() {
	preventSleepMu.Lock()
	defer preventSleepMu.Unlock()
	preventSleepCnt++
	
	if preventSleepCnt == 1 && !isActive {
		isActive = true
		if runtime.GOOS == "darwin" {
			// macOS: use caffeinate to prevent idle sleep
			exec.Command("sh", "-c", "caffeinate -i -s &").Start()
		} else if runtime.GOOS == "linux" {
			// Linux: use systemd-inhibit
			exec.Command("sh", "-c", "systemd-inhibit --what=idle sleep infinity &").Start()
		}
	}
}

// ReleasePowerLock allows the system to sleep again
func ReleasePowerLock() {
	preventSleepMu.Lock()
	defer preventSleepMu.Unlock()
	preventSleepCnt--
	
	if preventSleepCnt <= 0 {
		preventSleepCnt = 0
		if isActive {
			isActive = false
			// Kill the power management processes
			exec.Command("pkill", "-f", "caffeinate").Run()
			exec.Command("pkill", "-f", "systemd-inhibit").Run()
		}
	}
}
