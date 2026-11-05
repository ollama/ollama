//go:build windows

package server

import "golang.org/x/sys/windows"

const (
	esContinuous     = 0x80000000
	esSystemRequired = 0x00000001
)

var (
	modkernel32                 = windows.NewLazySystemDLL("kernel32.dll")
	procSetThreadExecutionState = modkernel32.NewProc("SetThreadExecutionState")
)

// platformAcquirePowerAssertion implements the windows side of issue #4072.
// ES_SYSTEM_REQUIRED resets the idle timer; ES_CONTINUOUS keeps the state
// until explicitly cleared. Display sleep is intentionally left alone.
var platformAcquirePowerAssertion = func() (func(), bool) {
	r1, _, _ := procSetThreadExecutionState.Call(esContinuous | esSystemRequired)
	if r1 == 0 {
		return nil, false
	}
	return func() {
		procSetThreadExecutionState.Call(esContinuous)
	}, true
}
