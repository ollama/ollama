//go:build windows

package server

import (
	"syscall"
)

var (
	kernel32                  = syscall.NewLazyDLL("kernel32.dll")
	procSetThreadExecutionState = kernel32.NewProc("SetThreadExecutionState")
)

// Execution state flags for SetThreadExecutionState
const (
	esSystemRequired  = 0x00000001
	esContinuous      = 0x80000000
)

func platformPreventSleep() error {
	// ES_CONTINUOUS | ES_SYSTEM_REQUIRED prevents the system from sleeping
	// until we call SetThreadExecutionState with just ES_CONTINUOUS
	_, _, err := procSetThreadExecutionState.Call(uintptr(esContinuous | esSystemRequired))
	if err != syscall.Errno(0) {
		return err
	}
	return nil
}

func platformAllowSleep() error {
	// Clear the ES_SYSTEM_REQUIRED flag to allow sleep again
	_, _, err := procSetThreadExecutionState.Call(uintptr(esContinuous))
	if err != syscall.Errno(0) {
		return err
	}
	return nil
}
