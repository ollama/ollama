//go:build windows

package cmd

import (
	"syscall"
	"unsafe"
)

// Windows constants for SetThreadExecutionState
const (
	ES_CONTINUOUS       = 0x80000000
	ES_SYSTEM_REQUIRED  = 0x00000001
	ES_DISPLAY_REQUIRED = 0x00000002
)

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	procSetThreadExecutionState = kernel32.NewProc("SetThreadExecutionState")
)

// PreventSystemSleep prevents the system from sleeping during inference
// Returns the previous state which should be restored with RestoreSystemState
func PreventSystemSleep() uint32 {
	var prevState uint32
	ret, _, _ := procSetThreadExecutionState.Call(
		uintptr(ES_CONTINUOUS | ES_SYSTEM_REQUIRED),
	)
	prevState = uint32(ret)
	return prevState
}

// AllowSystemSleep re-enables system sleep after inference
// Pass the previous state returned from PreventSystemSleep
func AllowSystemSleep(prevState uint32) {
	procSetThreadExecutionState.Call(uintptr(prevState))
}

// IsSystemSleepPreventionAvailable returns true on Windows
func IsSystemSleepPreventionAvailable() bool {
	return true
}
