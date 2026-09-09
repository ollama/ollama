//go:build windows

package server

import (
	"log/slog"
	"runtime"
	"time"

	"golang.org/x/sys/windows"
)

var (
	kernel32                 = windows.NewLazySystemDLL("kernel32.dll")
	pSetThreadExecutionState = kernel32.NewProc("SetThreadExecutionState")
)

const (
	esSystemRequired = 0x00000001
	esContinuous     = 0x80000000

	// refreshInterval resets the idle timer well before Windows' minimum
	// sleep timeout (typically 60s), keeping the system awake continuously.
	refreshInterval = 30 * time.Second
)

func setExecutionState(flags uintptr) bool {
	ret, _, err := pSetThreadExecutionState.Call(flags)
	if ret == 0 {
		slog.Warn("SetThreadExecutionState failed", "error", err)
		return false
	}
	return true
}

// preventSleep prevents the OS from sleeping during inference.
// It periodically resets the system idle timer so the computer
// stays awake for the full duration of the request.
// Returns a function that restores normal sleep behavior when called.
func preventSleep() func() {
	stop := make(chan struct{})

	go func() {
		// Lock this goroutine to its OS thread so SetThreadExecutionState
		// remains effective — the API is per-thread on Windows.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		setExecutionState(uintptr(esContinuous | esSystemRequired))

		ticker := time.NewTicker(refreshInterval)
		defer ticker.Stop()

		for {
			select {
			case <-stop:
				setExecutionState(uintptr(esContinuous))
				return
			case <-ticker.C:
				setExecutionState(uintptr(esContinuous | esSystemRequired))
			}
		}
	}()

	return func() {
		close(stop)
	}
}
